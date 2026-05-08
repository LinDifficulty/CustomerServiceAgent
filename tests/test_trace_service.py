from __future__ import annotations

import tempfile
import unittest

from rag_server.trace_service import (
    REDACTED_VALUE,
    TraceRecorder,
    load_trace,
    summarize_trace,
)


class TraceServiceTests(unittest.TestCase):
    def test_writes_redacted_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                trace_dir=temp_dir,
                run_id="test-run",
                default_tags={"user_id": "u1", "api_key": "secret"},
            )
            recorder.event(
                "runtime",
                "startup",
                {
                    "Authorization": "Bearer token",
                    "nested": {"password": "pw", "safe": "value"},
                },
            )

            records = load_trace(recorder.path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tags"]["api_key"], REDACTED_VALUE)
        self.assertEqual(records[0]["payload"]["Authorization"], REDACTED_VALUE)
        self.assertEqual(records[0]["payload"]["nested"]["password"], REDACTED_VALUE)
        self.assertEqual(records[0]["payload"]["nested"]["safe"], "value")

    def test_metric_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(trace_dir=temp_dir, run_id="summary")
            recorder.event("agent", "agent.start", {})
            recorder.event("agent", "agent.warning", {}, level="warning")
            recorder.metric("latency_ms", 12.5, unit="ms")
            records = load_trace(recorder.path)

        summary = summarize_trace(records)
        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["warning_count"], 1)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["event_types"]["metric"], 1)
        self.assertIn("latency_ms", summary["top_names"])

    def test_event_sinks_receive_records_when_file_trace_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            seen = []
            recorder = TraceRecorder(
                trace_dir=temp_dir,
                run_id="sink",
                enabled=False,
                event_sinks=[seen.append],
            )

            record = recorder.event("rag", "rag.search", {"query": "尺码"})

        self.assertEqual(seen, [record])


if __name__ == "__main__":
    unittest.main()
