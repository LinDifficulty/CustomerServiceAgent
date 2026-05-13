from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import utc_now

# 默认追踪数据存储目录
DEFAULT_TRACE_DIR = "traces"
# 预览文本时最大保留的字符数，超出部分会被截断
MAX_PREVIEW_CHARS = 300
# 敏感信息被掩盖后的替代文本
REDACTED_VALUE = "[redacted]"
# 默认需要脱敏的键名元组，匹配时忽略大小写和连字符/下划线差异
DEFAULT_REDACT_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


def _json_safe(value: Any) -> Any:
    # 递归地将任意Python值转换为JSON可序列化的类型
    # 处理Path、set等JSON原生不支持的Python类型
    # 对字典递归转换，确保所有值都是JSON安全类型
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    # 对列表/元组/集合递归转换每个元素
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    # Path对象转为字符串
    if isinstance(value, Path):
        return str(value)
    # 原生JSON类型直接返回
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    # 其他类型兜底转为字符串
    return str(value)


def _should_redact_key(key: str, redact_keys: tuple[str, ...]) -> bool:
    # 判断一个键名是否包含敏感词汇，需要被脱敏
    # 先将键名统一为小写、将连字符替换为下划线，再做子串匹配
    normalized = key.lower().replace("-", "_")
    return any(item in normalized for item in redact_keys)


def _redact_sensitive(value: Any, redact_keys: tuple[str, ...]) -> Any:
    # 递归地对数据中的敏感字段进行脱敏处理
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            text_key = str(key)
            # 如果键名匹配敏感词，值替换为 [redacted]
            if _should_redact_key(text_key, redact_keys):
                result[text_key] = REDACTED_VALUE
            else:
                # 否则继续递归处理嵌套值
                result[text_key] = _redact_sensitive(item, redact_keys)
        return result
    if isinstance(value, list | tuple | set):
        return [_redact_sensitive(item, redact_keys) for item in value]
    return value


def preview_text(text: Any, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    # 对长文本进行截断预览，防止追踪记录中存储过大的文本内容
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def summarize_result(item: dict, *, include_content: bool = False) -> dict:
    # 将单个检索结果条目压缩为精简摘要，仅保留关键评分和元信息
    # include_content=True 时额外附加内容预览
    # 提取各类评分：综合分、向量分、BM25分、融合分、重排序分、多向量分
    summary = {
        "score": item.get("score"),
        "vector_score": item.get("vector_score"),
        "bm25_score": item.get("bm25_score"),
        "hybrid_score": item.get("hybrid_score"),
        "rerank_score": item.get("rerank_score"),
        "multi_vector_scores": item.get("multi_vector_scores"),
        "best_vector_type": item.get("best_vector_type"),
        "source": item.get("source"),
        # doc_id 可能在顶层也可能在 metadata 嵌套中
        "doc_id": item.get("doc_id") or item.get("metadata", {}).get("doc_id"),
        "chunk_index": item.get("metadata", {}).get("chunk_index"),
        "retrieval_mode": item.get("retrieval_mode"),
        "matched_queries": item.get("matched_queries"),
    }
    if include_content:
        # 可选择性地附加内容预览（截断后）
        summary["content_preview"] = preview_text(item.get("content", ""))
    return summary


@dataclass
class TraceRecorder:
    """Append-only JSONL trace writer for local RAG and Agent runs."""

    trace_dir: str | Path = DEFAULT_TRACE_DIR  # 追踪文件存放目录
    run_id: str | None = None  # 本次运行的唯一ID，不传则自动生成
    enabled: bool = True  # 是否启用追踪记录
    default_tags: dict[str, Any] = field(default_factory=dict)  # 每条事件默认附加的标签
    redact_keys: tuple[str, ...] = DEFAULT_REDACT_KEYS  # 需要脱敏的键名
    event_sinks: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)  # 事件输出的额外接收器

    def __post_init__(self) -> None:
        # 初始化后自动生成 run_id（如果未提供）
        self.run_id = self.run_id or uuid.uuid4().hex
        # 确保 trace_dir 是 Path 对象
        self.trace_dir = Path(self.trace_dir)
        # 追踪文件路径：traces/<run_id>.jsonl
        self.path = self.trace_dir / f"{self.run_id}.jsonl"
        # 统一脱敏键为小写，方便后续匹配
        self.redact_keys = tuple(item.lower() for item in self.redact_keys)
        if self.enabled:
            # 如果启用了追踪，确保目录存在
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def event(
        self,
        event_type: str,  # 事件类型，如 "span_start", "span_end", "metric", "rag_call" 等
        name: str,  # 事件名称，便于筛选和聚合，如 "retrieve", "agent_call"
        payload: dict[str, Any] | None = None,  # 附加数据载荷
        *,
        parent_id: str | None = None,  # 父事件ID，用于构建事件树
        span_id: str | None = None,  # 当前 span 的ID
        elapsed_ms: float | None = None,  # 耗时（毫秒）
        level: str = "info",  # 日志级别：info / warning / error
    ) -> dict:
        # 构建事件记录，包含运行ID、事件ID、时间戳等字段
        record = {
            "run_id": self.run_id,
            "event_id": uuid.uuid4().hex,
            "timestamp": utc_now(),
            "type": event_type,
            "name": name,
            "level": level,
            "parent_id": parent_id,
            "span_id": span_id,
            "elapsed_ms": elapsed_ms,
            "tags": _redact_sensitive(self.default_tags, self.redact_keys),  # 脱敏处理标签
            "payload": _redact_sensitive(payload or {}, self.redact_keys),  # 脱敏处理载荷
        }
        self.write(record)  # 写入 JSONL 文件
        self._notify_sinks(record)  # 通知所有附加的事件接收器
        return record

    def metric(
        self,
        name: str,  # 指标名称
        value: int | float,  # 指标数值
        *,
        unit: str = "",  # 单位，如 "ms", "count"
        tags: dict[str, Any] | None = None,  # 指标的附加标签
        payload: dict[str, Any] | None = None,  # 额外载荷
        level: str = "info",  # 日志级别
    ) -> dict:
        # 构建指标专用的 payload 结构
        metric_payload = {
            "metric_name": name,
            "value": float(value),
            "unit": unit,
            "metric_tags": tags or {},
            **(payload or {}),
        }
        # 指标事件统一使用 "metric" 类型
        return self.event("metric", name, metric_payload, level=level)

    @contextmanager
    def span(
        self,
        name: str,  # Span 名称，如 "embed_query", "bm25_search"
        payload: dict[str, Any] | None = None,  # 开始时的附加数据
        *,
        parent_id: str | None = None,  # 父 span 的 ID，用于构建调用链
    ) -> Iterator[str]:
        # Span 上下文管理器，自动记录开始、结束和异常
        # 使用方式：with recorder.span("name") as span_id: ...
        span_id = uuid.uuid4().hex  # 生成唯一的 span ID
        start = time.perf_counter()  # 高精度计时起点
        # 记录 span 开始事件
        self.event(
            "span_start",
            name,
            payload,
            parent_id=parent_id,
            span_id=span_id,
        )
        try:
            yield span_id  # 交出 span_id 供调用方使用
        except Exception as error:
            # 发生异常时，计算耗时并记录错误事件
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.event(
                "span_error",
                name,
                {"error": repr(error)},
                parent_id=parent_id,
                span_id=span_id,
                elapsed_ms=elapsed_ms,
                level="error",
            )
            raise  # 重新抛出异常，不吞没错误
        else:
            # 正常结束时，计算耗时并记录结束事件
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.event(
                "span_end",
                name,
                {},
                parent_id=parent_id,
                span_id=span_id,
                elapsed_ms=elapsed_ms,
            )

    def write(self, record: dict[str, Any]) -> None:
        # 将一条事件记录追加写入 JSONL 文件
        if not self.enabled:
            return  # 如果追踪被禁用，直接跳过
        with self.path.open("a", encoding="utf-8") as file:
            # 每行一条 JSON，确保中文不乱码 (ensure_ascii=False)
            file.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def add_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        # 注册额外的事件接收器（如回调函数），每条事件写入后都会调用
        self.event_sinks.append(sink)

    def _notify_sinks(self, record: dict[str, Any]) -> None:
        # 通知所有已注册的事件接收器
        for sink in self.event_sinks:
            try:
                sink(record)
            except Exception:
                # 可观测性接收器绝不能影响主业务流程，异常静默忽略
                continue


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    # 从 JSONL 追踪文件中加载所有事件记录
    records: list[dict[str, Any]] = []
    trace_path = Path(path)
    if not trace_path.exists():
        return records  # 文件不存在时返回空列表
    # 逐行读取，每行是一个 JSON 对象
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue  # 跳过空行
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # 跳过无法解析的行
        if isinstance(record, dict):
            records.append(record)
    return records


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact operational summary from JSONL trace records.
    从追踪事件列表中构建精简的运营摘要。
    """
    # 统计各级别事件数量：info / warning / error
    levels = Counter(str(item.get("level") or "info") for item in records)
    # 统计各事件类型数量：span_start / span_end / metric 等
    event_types = Counter(str(item.get("type") or "") for item in records)
    # 统计各事件名称数量，取 top 10
    names = Counter(str(item.get("name") or "") for item in records)
    # 收集所有有效耗时（毫秒）
    elapsed_values = [float(item["elapsed_ms"]) for item in records if isinstance(item.get("elapsed_ms"), int | float)]
    # 收集所有时间戳
    timestamps = [str(item.get("timestamp")) for item in records if item.get("timestamp")]
    return {
        "event_count": len(records),  # 事件总数
        "levels": dict(sorted(levels.items())),  # 各级别分布
        "event_types": dict(sorted(event_types.items())),  # 各类型分布
        "top_names": dict(names.most_common(10)),  # 最频繁的10个事件名
        "warning_count": levels.get("warning", 0),  # 警告事件数
        "error_count": levels.get("error", 0),  # 错误事件数
        "elapsed_ms_total": sum(elapsed_values),  # 总耗时（毫秒）
        "first_timestamp": min(timestamps) if timestamps else None,  # 最早时间戳
        "last_timestamp": max(timestamps) if timestamps else None,  # 最晚时间戳
    }
