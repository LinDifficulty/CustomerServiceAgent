from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def coerce_message_content(content: Any) -> str:
    """Normalize LangChain message content to a plain string.
    将 LangChain 消息内容规范化为纯字符串格式。

    Handles str, list-of-dicts (multi-part messages), and arbitrary values.
    处理字符串、字典列表（多部分消息）和任意值类型。
    """
    if isinstance(content, str):
        # 如果已经是字符串，直接返回
        return content
    if isinstance(content, list):
        # 如果是列表（LangChain多部分消息格式），逐个提取文本
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                # 字典类型且有 text 字段，提取其文本值
                parts.append(str(item["text"]))
            else:
                # 其他类型直接转字符串
                parts.append(str(item))
        # 用换行符合并所有部分
        return "\n".join(parts)
    # 其他类型直接转字符串作为兜底
    return str(content)


def parse_json_object(raw_response: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response string.
    从 LLM 返回的字符串中尽力提取 JSON 对象。
    LLM 经常在 JSON 前后附加说明文字，本函数会尝试从响应中定位并解析 JSON。
    """
    # 候选解析目标列表，优先尝试提取后的内容
    candidates = [raw_response]
    # 先尝试从 markdown 代码块（```json ... ```）中提取 JSON
    json_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    if json_fence:
        candidates.insert(0, json_fence.group(1))
    # 查找第一个 { 和最后一个 }，提取中间的 JSON 片段
    json_block = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if json_block:
        candidates.insert(0, json_block.group(0))

    # 逐个尝试解析候选字符串
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue  # 解析失败则尝试下一个候选
        if isinstance(payload, dict):
            return payload  # 成功解析为字典对象则返回
    # 所有尝试都失败时返回空字典作为兜底
    return {}


def coerce_bool(value: Any, *, default: bool = False, strict: bool = False) -> bool:
    """Coerce a value to bool, with support for common string representations.
    将任意值强制转为布尔值，支持常见的中英文字符串表示。
    例如 "是"、"true"、"yes"、"1" 等都视为 True。

    When strict=True, raises ValueError for unrecognized values instead of falling back.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1", "是", "需要"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        if strict:
            raise ValueError(f"unrecognized boolean string: {value!r}")
        return bool(value)
    if value is None:
        return default
    if strict and not isinstance(value, (bool, str)):
        raise ValueError(f"unexpected type for boolean: {type(value).__name__}")
    return bool(value)


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def cache_key_or_none(cache: Any, category: str, payload: dict[str, Any]) -> str | None:
    """Generate a cache key, or return None if no cache is configured."""
    if cache is None:
        return None
    return cache.make_key(category, payload)


def normalize_vector_score(score: float) -> float:
    """将 FAISS 内积分数归一化到 [0.0, 1.0] 区间。

    L2 归一化向量的内积范围是 [-1, 1]，通过 (score+1)/2 映射到 [0, 1]。
    """
    return max(0.0, min(1.0, (score + 1) / 2))


def trace_retry_failure(
    trace_recorder: Any,
    category: str,
    name: str,
    provider: str,
    model_name: str,
    event: dict[str, Any],
) -> None:
    """记录 LLM 调用重试失败事件。

    如果 trace_recorder 为 None 则静默跳过。
    level 根据 event 中的 will_retry 字段自动选择 warning 或 error。
    """
    if trace_recorder is None:
        return
    trace_recorder.event(
        category,
        name,
        {"provider": provider, "model_name": model_name, **event},
        level="warning" if event.get("will_retry") else "error",
    )


async def call_async_fallback(obj: Any, async_attr: str, sync_attr: str, *args: Any, **kwargs: Any) -> Any:
    """优先调用对象的异步方法，不存在时通过 asyncio.to_thread 回退到同步版本。"""
    method = getattr(obj, async_attr, None)
    if method is not None:
        return await method(*args, **kwargs)
    sync_fn = getattr(obj, sync_attr)
    return await asyncio.to_thread(sync_fn, *args, **kwargs)


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory.

    Args:
        name: The prompt filename (e.g. "system_prompt.txt").

    Returns:
        The prompt content as a string, with leading/trailing whitespace stripped.
    """
    file_path = _PROMPTS_DIR / name
    return file_path.read_text(encoding="utf-8").strip()
