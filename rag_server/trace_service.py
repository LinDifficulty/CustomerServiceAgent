from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

DEFAULT_TRACE_DIR = "traces"
MAX_PREVIEW_CHARS = 300
REDACTED_VALUE = "[redacted]"
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _should_redact_key(key: str, redact_keys: tuple[str, ...]) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(item in normalized for item in redact_keys)


def _redact_sensitive(value: Any, redact_keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            text_key = str(key)
            if _should_redact_key(text_key, redact_keys):
                result[text_key] = REDACTED_VALUE
            else:
                result[text_key] = _redact_sensitive(item, redact_keys)
        return result
    if isinstance(value, list | tuple | set):
        return [_redact_sensitive(item, redact_keys) for item in value]
    return value


def preview_text(text: Any, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def summarize_result(item: dict, *, include_content: bool = False) -> dict:
    summary = {
        "score": item.get("score"),
        "vector_score": item.get("vector_score"),
        "bm25_score": item.get("bm25_score"),
        "hybrid_score": item.get("hybrid_score"),
        "rerank_score": item.get("rerank_score"),
        "multi_vector_scores": item.get("multi_vector_scores"),
        "best_vector_type": item.get("best_vector_type"),
        "source": item.get("source"),
        "doc_id": item.get("doc_id") or item.get("metadata", {}).get("doc_id"),
        "chunk_index": item.get("metadata", {}).get("chunk_index"),
        "retrieval_mode": item.get("retrieval_mode"),
        "matched_queries": item.get("matched_queries"),
    }
    if include_content:
        summary["content_preview"] = preview_text(item.get("content", ""))
    return summary


@dataclass
class TraceRecorder:
    """Append-only JSONL trace writer for local RAG and Agent runs."""

    trace_dir: str | Path = DEFAULT_TRACE_DIR
    run_id: str | None = None
    enabled: bool = True
    default_tags: dict[str, Any] = field(default_factory=dict)
    redact_keys: tuple[str, ...] = DEFAULT_REDACT_KEYS
    event_sinks: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_id = self.run_id or uuid.uuid4().hex
        self.trace_dir = Path(self.trace_dir)
        self.path = self.trace_dir / f"{self.run_id}.jsonl"
        self.redact_keys = tuple(item.lower() for item in self.redact_keys)
        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def event(
        self,
        event_type: str,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        parent_id: str | None = None,
        span_id: str | None = None,
        elapsed_ms: float | None = None,
        level: str = "info",
    ) -> dict:
        record = {
            "run_id": self.run_id,
            "event_id": uuid.uuid4().hex,
            "timestamp": _utc_now(),
            "type": event_type,
            "name": name,
            "level": level,
            "parent_id": parent_id,
            "span_id": span_id,
            "elapsed_ms": elapsed_ms,
            "tags": _redact_sensitive(self.default_tags, self.redact_keys),
            "payload": _redact_sensitive(payload or {}, self.redact_keys),
        }
        self.write(record)
        self._notify_sinks(record)
        return record

    def metric(
        self,
        name: str,
        value: int | float,
        *,
        unit: str = "",
        tags: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        level: str = "info",
    ) -> dict:
        metric_payload = {
            "metric_name": name,
            "value": float(value),
            "unit": unit,
            "metric_tags": tags or {},
            **(payload or {}),
        }
        return self.event("metric", name, metric_payload, level=level)

    @contextmanager
    def span(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        parent_id: str | None = None,
    ) -> Iterator[str]:
        span_id = uuid.uuid4().hex
        start = time.perf_counter()
        self.event(
            "span_start",
            name,
            payload,
            parent_id=parent_id,
            span_id=span_id,
        )
        try:
            yield span_id
        except Exception as error:
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
            raise
        else:
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
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def add_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        self.event_sinks.append(sink)

    def _notify_sinks(self, record: dict[str, Any]) -> None:
        for sink in self.event_sinks:
            try:
                sink(record)
            except Exception:
                # Observability sinks must never break the main user flow.
                continue


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    trace_path = Path(path)
    if not trace_path.exists():
        return records
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact operational summary from JSONL trace records."""
    levels = Counter(str(item.get("level") or "info") for item in records)
    event_types = Counter(str(item.get("type") or "") for item in records)
    names = Counter(str(item.get("name") or "") for item in records)
    elapsed_values = [
        float(item["elapsed_ms"])
        for item in records
        if isinstance(item.get("elapsed_ms"), int | float)
    ]
    timestamps = [
        str(item.get("timestamp"))
        for item in records
        if item.get("timestamp")
    ]
    return {
        "event_count": len(records),
        "levels": dict(sorted(levels.items())),
        "event_types": dict(sorted(event_types.items())),
        "top_names": dict(names.most_common(10)),
        "warning_count": levels.get("warning", 0),
        "error_count": levels.get("error", 0),
        "elapsed_ms_total": sum(elapsed_values),
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }
