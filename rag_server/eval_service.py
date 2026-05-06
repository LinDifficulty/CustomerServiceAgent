from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rag_service import RAGService
from .trace_service import TraceRecorder, summarize_result


@dataclass(frozen=True)
class RetrievalEvalCase:
    id: str
    query: str
    expected_sources: list[str]
    expected_doc_ids: list[str]
    expected_substrings: list[str]


def load_retrieval_eval_dataset(path: str | Path) -> list[RetrievalEvalCase]:
    """Load retrieval eval cases from JSON or JSONL."""
    eval_path = Path(path)
    text = eval_path.read_text(encoding="utf-8")
    if eval_path.suffix.lower() == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        payload = json.loads(text)
        raw_items = payload.get("cases", payload) if isinstance(payload, dict) else payload

    cases: list[RetrievalEvalCase] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        cases.append(
            RetrievalEvalCase(
                id=str(item.get("id") or f"case_{index + 1}"),
                query=query,
                expected_sources=_normalize_string_list(item.get("expected_sources")),
                expected_doc_ids=_normalize_string_list(item.get("expected_doc_ids")),
                expected_substrings=_normalize_string_list(
                    item.get("expected_substrings")
                ),
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
    """Run a deterministic retrieval eval over a RAGService instance."""
    cases = load_retrieval_eval_dataset(dataset_path)
    case_reports = []

    for case in cases:
        results = rag.search(
            case.query,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            use_bm25=use_bm25,
            use_rerank=use_rerank,
        )
        match_rank = _first_match_rank(case, results)
        source_rank = _first_source_rank(case, results)
        substring_rank = _first_substring_rank(case, results)
        report = {
            "id": case.id,
            "query": case.query,
            "hit": match_rank is not None,
            "rank": match_rank,
            "reciprocal_rank": 0.0 if match_rank is None else 1.0 / match_rank,
            "source_hit": source_rank is not None,
            "source_rank": source_rank,
            "substring_hit": substring_rank is not None,
            "substring_rank": substring_rank,
            "expected_sources": case.expected_sources,
            "expected_doc_ids": case.expected_doc_ids,
            "expected_substrings": case.expected_substrings,
            "results": [
                summarize_result(item, include_content=True) for item in results
            ],
        }
        case_reports.append(report)
        if trace_recorder is not None:
            trace_recorder.event("eval", "eval.retrieval_case", report)

    summary = _summarize_cases(case_reports)
    payload = {
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "case_count": len(case_reports),
        "summary": summary,
        "cases": case_reports,
    }
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
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_match_rank(case: RetrievalEvalCase, results: list[dict]) -> int | None:
    for rank, item in enumerate(results, start=1):
        if _matches_expected(case, item):
            return rank
    return None


def _first_source_rank(case: RetrievalEvalCase, results: list[dict]) -> int | None:
    if not case.expected_sources and not case.expected_doc_ids:
        return None
    for rank, item in enumerate(results, start=1):
        if _matches_source(case, item):
            return rank
    return None


def _first_substring_rank(case: RetrievalEvalCase, results: list[dict]) -> int | None:
    if not case.expected_substrings:
        return None
    for rank, item in enumerate(results, start=1):
        if _matches_substring(case, item):
            return rank
    return None


def _matches_expected(case: RetrievalEvalCase, item: dict) -> bool:
    source_expected = bool(case.expected_sources or case.expected_doc_ids)
    substring_expected = bool(case.expected_substrings)
    source_ok = not source_expected or _matches_source(case, item)
    substring_ok = not substring_expected or _matches_substring(case, item)
    return source_ok and substring_ok


def _matches_source(case: RetrievalEvalCase, item: dict) -> bool:
    doc_id = str(item.get("doc_id") or item.get("metadata", {}).get("doc_id") or "")
    if doc_id and doc_id in case.expected_doc_ids:
        return True

    source = _normalize_path_string(str(item.get("source") or ""))
    return any(
        source.endswith(_normalize_path_string(expected_source))
        for expected_source in case.expected_sources
    )


def _matches_substring(case: RetrievalEvalCase, item: dict) -> bool:
    content = str(item.get("content") or "")
    return any(expected in content for expected in case.expected_substrings)


def _normalize_path_string(value: str) -> str:
    return value.replace("\\", "/").strip()


def _summarize_cases(case_reports: list[dict]) -> dict:
    count = len(case_reports)
    if count == 0:
        return {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "source_hit_rate": 0.0,
            "substring_hit_rate": 0.0,
        }
    return {
        "hit_rate": sum(1 for item in case_reports if item["hit"]) / count,
        "mrr": sum(float(item["reciprocal_rank"]) for item in case_reports) / count,
        "source_hit_rate": (
            sum(1 for item in case_reports if item["source_hit"]) / count
        ),
        "substring_hit_rate": (
            sum(1 for item in case_reports if item["substring_hit"]) / count
        ),
    }
