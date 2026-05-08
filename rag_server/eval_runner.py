from __future__ import annotations

import argparse
import json

from .eval_service import evaluate_retrieval_dataset, write_eval_report
from .rag_service import RAGService
from .trace_service import DEFAULT_TRACE_DIR, TraceRecorder, load_trace, summarize_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evals for RAGService.")
    parser.add_argument(
        "--dataset",
        default="evals/retrieval_eval.jsonl",
        help="JSON or JSONL retrieval eval dataset path.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="RAG index data directory. Defaults to data.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of final retrieval results to evaluate.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=10,
        help="Number of candidates before rerank. Defaults to 10.",
    )
    parser.add_argument(
        "--bm25",
        choices=["on", "off"],
        default="on",
        help="Enable or disable BM25 during eval.",
    )
    parser.add_argument(
        "--cross-encoder",
        choices=["on", "off"],
        default="on",
        help="Enable or disable CrossEncoder reranking during eval.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--trace",
        choices=["on", "off"],
        default="on",
        help="Enable or disable JSONL tracing for eval runs. Defaults to on.",
    )
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=f"Directory for JSONL trace files. Defaults to {DEFAULT_TRACE_DIR}.",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Fail the eval run if hit_rate is below this value.",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Fail the eval run if mrr is below this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_recorder = (
        TraceRecorder(
            trace_dir=args.trace_dir,
            default_tags={"entrypoint": "eval_runner", "dataset": args.dataset},
        )
        if args.trace == "on"
        else None
    )
    rag = RAGService(
        data_dir=args.data_dir,
        default_use_bm25=args.bm25 == "on",
        default_use_rerank=args.cross_encoder == "on",
        trace_recorder=trace_recorder,
    )
    report = evaluate_retrieval_dataset(
        rag,
        args.dataset,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
        use_bm25=args.bm25 == "on",
        use_rerank=args.cross_encoder == "on",
        trace_recorder=trace_recorder,
    )
    if args.output:
        write_eval_report(report, args.output)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if trace_recorder is not None:
        print(f"trace: {trace_recorder.path}")
        print(
            "trace_summary: "
            + json.dumps(
                summarize_trace(load_trace(trace_recorder.path)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if args.output:
        print(f"report: {args.output}")

    failures = []
    if (
        args.min_hit_rate is not None
        and report["summary"]["hit_rate"] < args.min_hit_rate
    ):
        failures.append(
            f"hit_rate {report['summary']['hit_rate']:.4f} < {args.min_hit_rate:.4f}"
        )
    if args.min_mrr is not None and report["summary"]["mrr"] < args.min_mrr:
        failures.append(f"mrr {report['summary']['mrr']:.4f} < {args.min_mrr:.4f}")
    if failures:
        raise SystemExit("Eval failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
