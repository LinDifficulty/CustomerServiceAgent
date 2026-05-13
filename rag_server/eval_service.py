from __future__ import annotations

import json
from dataclasses import dataclass  # 用于定义不可变数据类
from pathlib import Path
from typing import Any

from .rag_service import RAGService
from .trace_service import TraceRecorder, summarize_result


# 检索评测用例的数据类 —— 描述一条评测期望：给定查询，预期匹配哪些文档
@dataclass(frozen=True)
class RetrievalEvalCase:
    """单条检索评测用例，包含查询和多种匹配条件。"""

    id: str  # 用例唯一标识
    query: str  # 评测查询文本
    expected_sources: list[str]  # 期望命中的文档来源路径列表
    expected_doc_ids: list[str]  # 期望命中的文档 ID 列表
    expected_substrings: list[str]  # 期望检索结果内容中出现的子串列表


def load_retrieval_eval_dataset(path: str | Path) -> list[RetrievalEvalCase]:
    """从 JSON 或 JSONL 文件加载检索评测数据集。"""
    eval_path = Path(path)
    text = eval_path.read_text(encoding="utf-8")

    # 根据文件后缀区分 JSONL（逐行 JSON）和普通 JSON
    if eval_path.suffix.lower() == ".jsonl":
        # JSONL 格式：每行一个 JSON 对象，跳过空行和注释行（以 # 开头）
        raw_items = [
            json.loads(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        # JSON 格式：顶层可以是数组，或者包含 "cases" 键的对象
        payload = json.loads(text)
        raw_items = payload.get("cases", payload) if isinstance(payload, dict) else payload

    cases: list[RetrievalEvalCase] = []
    for index, item in enumerate(raw_items):
        # 跳过非字典类型的数据条目
        if not isinstance(item, dict):
            continue
        # 查询文本不能为空
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        # 构建评测用例对象，每个期望字段都经过规范化处理
        cases.append(
            RetrievalEvalCase(
                id=str(item.get("id") or f"case_{index + 1}"),  # 无 id 时自动生成
                query=query,
                expected_sources=_normalize_string_list(item.get("expected_sources")),
                expected_doc_ids=_normalize_string_list(item.get("expected_doc_ids")),
                expected_substrings=_normalize_string_list(item.get("expected_substrings")),
            )
        )
    return cases


def evaluate_retrieval_dataset(
    rag: RAGService,
    dataset_path: str | Path,
    *,
    top_k: int = 3,
    candidate_top_k: int | None = None,
    use_bm25: bool | None = None,
    use_rerank: bool | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> dict:
    """对 RAG 服务执行确定性检索评测，返回包含逐条结果和汇总指标的报告。"""
    # 加载评测数据集
    cases = load_retrieval_eval_dataset(dataset_path)
    case_reports = []  # 存储每条用例的评测结果

    # 逐条遍历评测用例
    for case in cases:
        # 调用 RAG 检索服务获取 top_k 结果
        results = rag.search(
            case.query,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            use_bm25=use_bm25,
            use_rerank=use_rerank,
        )

        # 计算三种匹配方式的最早命中排名：
        #   match_rank: 综合匹配（source 和 substring 同时满足）
        #   source_rank: 仅按文档来源匹配
        #   substring_rank: 仅按内容子串匹配
        match_rank = _first_match_rank(case, results)
        source_rank = _first_source_rank(case, results)
        substring_rank = _first_substring_rank(case, results)

        # 构建单条用例的评测报告
        report = {
            "id": case.id,
            "query": case.query,
            "hit": match_rank is not None,  # 是否有综合命中
            "rank": match_rank,  # 综合命中排名（1-based）
            "reciprocal_rank": 0.0 if match_rank is None else 1.0 / match_rank,  # 倒数排名
            "source_hit": source_rank is not None,  # 是否有来源命中
            "source_rank": source_rank,  # 来源命中排名
            "substring_hit": substring_rank is not None,  # 是否有子串命中
            "substring_rank": substring_rank,  # 子串命中排名
            "expected_sources": case.expected_sources,  # 记录期望条件，便于人工复核
            "expected_doc_ids": case.expected_doc_ids,
            "expected_substrings": case.expected_substrings,
            "results": [
                # 对每一条检索结果生成摘要（包含内容）
                summarize_result(item, include_content=True)
                for item in results
            ],
        }
        case_reports.append(report)

        # 如果启用了追踪，记录单条用例的评测事件
        if trace_recorder is not None:
            trace_recorder.event("eval", "eval.retrieval_case", report)

    # 汇总所有用例的评测指标（hit_rate, mrr 等）
    summary = _summarize_cases(case_reports)

    # 构建完整的评测报告
    payload = {
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "case_count": len(case_reports),
        "summary": summary,
        "cases": case_reports,
    }

    # 如果启用了追踪，记录整个评测的汇总事件
    if trace_recorder is not None:
        trace_recorder.event(
            "eval",
            "eval.retrieval_summary",
            {
                "dataset_path": str(dataset_path),
                "top_k": top_k,
                "candidate_top_k": candidate_top_k,
                **summary,
            },
        )
    return payload


def write_eval_report(report: dict, output_path: str | Path) -> None:
    """将评测报告写入 JSON 文件，自动创建父目录。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_string_list(value: Any) -> list[str]:
    """将各种输入格式统一规范化为字符串列表。
    支持 None -> []、字符串 -> [字符串]、列表 -> 去除空白后非空字符串的列表。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    # 过滤掉空白字符串
    return [str(item).strip() for item in value if str(item).strip()]


def _first_match_rank(case: RetrievalEvalCase, results: list[dict]) -> int | None:
    """返回综合匹配（source 和 substring 同时满足）的最早命中排名（1-based），无命中返回 None。"""
    for rank, item in enumerate(results, start=1):
        if _matches_expected(case, item):
            return rank
    return None


def _first_source_rank(case: RetrievalEvalCase, results: list[dict]) -> int | None:
    """返回仅按文档来源匹配的最早命中排名。如果用例未定义来源期望，返回 None。"""
    # 如果用例没有指定任何来源或文档 ID 期望，则无法判断来源匹配
    if not case.expected_sources and not case.expected_doc_ids:
        return None
    for rank, item in enumerate(results, start=1):
        if _matches_source(case, item):
            return rank
    return None


def _first_substring_rank(case: RetrievalEvalCase, results: list[dict]) -> int | None:
    """返回仅按内容子串匹配的最早命中排名。如果用例未定义子串期望，返回 None。"""
    if not case.expected_substrings:
        return None
    for rank, item in enumerate(results, start=1):
        if _matches_substring(case, item):
            return rank
    return None


def _matches_expected(case: RetrievalEvalCase, item: dict) -> bool:
    """综合匹配：同时检查来源匹配和子串匹配，两者之中只检查用例中已定义的那些。
    逻辑是：如果用例定义了来源期望，则必须来源匹配；如果定义了子串期望，则必须子串匹配。
    """
    # 判断用例是否定义了来源/文档ID匹配条件
    source_expected = bool(case.expected_sources or case.expected_doc_ids)
    # 判断用例是否定义了子串匹配条件
    substring_expected = bool(case.expected_substrings)

    # 如果没定义来源期望，来源匹配自动通过；如果没定义子串期望，子串匹配自动通过
    source_ok = not source_expected or _matches_source(case, item)
    substring_ok = not substring_expected or _matches_substring(case, item)

    return source_ok and substring_ok


def _matches_source(case: RetrievalEvalCase, item: dict) -> bool:
    """检查检索结果是否匹配用例期望的文档来源。
    匹配方式：比较 doc_id 是否在预期列表中，或比较文件来源路径。
    """
    # 尝试从多个字段提取 doc_id
    doc_id = str(item.get("doc_id") or item.get("metadata", {}).get("doc_id") or "")
    if doc_id and doc_id in case.expected_doc_ids:
        return True

    # 规范化路径斜杠后，检查来源路径的后缀是否匹配（支持相对路径匹配）
    source = _normalize_path_string(str(item.get("source") or ""))
    return any(source.endswith(_normalize_path_string(expected_source)) for expected_source in case.expected_sources)


def _matches_substring(case: RetrievalEvalCase, item: dict) -> bool:
    """检查检索结果的内容中是否包含任意期望的子串。"""
    content = str(item.get("content") or "")
    return any(expected in content for expected in case.expected_substrings)


def _normalize_path_string(value: str) -> str:
    """规范化路径字符串：将反斜杠转换为正斜杠，去除首尾空白。"""
    return value.replace("\\", "/").strip()


def _summarize_cases(case_reports: list[dict]) -> dict:
    """汇总所有用例的评测指标，计算 hit_rate、mrr、source_hit_rate、substring_hit_rate。"""
    count = len(case_reports)
    # 如果没有评测用例，各项指标直接返回 0
    if count == 0:
        return {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "source_hit_rate": 0.0,
            "substring_hit_rate": 0.0,
        }

    return {
        # hit_rate: 综合命中用例数 / 总用例数
        "hit_rate": sum(1 for item in case_reports if item["hit"]) / count,
        # mrr (Mean Reciprocal Rank): 各用例倒数排名的平均值
        "mrr": sum(float(item["reciprocal_rank"]) for item in case_reports) / count,
        # source_hit_rate: 来源命中用例数 / 总用例数
        "source_hit_rate": (sum(1 for item in case_reports if item["source_hit"]) / count),
        # substring_hit_rate: 子串命中用例数 / 总用例数
        "substring_hit_rate": (sum(1 for item in case_reports if item["substring_hit"]) / count),
    }
