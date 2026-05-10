from __future__ import annotations

import argparse
import json

# 导入评测核心函数：执行检索评测、写入评测报告
from .eval_service import evaluate_retrieval_dataset, write_eval_report
# 导入模型默认配置常量
from .model_factory import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_PROVIDER,
)
from .rag_service import RAGService
# 导入追踪服务：用于记录评测过程中的事件日志
from .trace_service import DEFAULT_TRACE_DIR, TraceRecorder, load_trace, summarize_trace


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回命名空间对象。"""
    parser = argparse.ArgumentParser(description="Run retrieval evals for RAGService.")
    # --dataset: 评测数据集路径，支持 JSON 或 JSONL 格式
    parser.add_argument(
        "--dataset",
        default="evals/retrieval_eval.jsonl",
        help="JSON or JSONL retrieval eval dataset path.",
    )
    # --data-dir: RAG 索引数据目录，包含 FAISS 索引和文档
    parser.add_argument(
        "--data-dir",
        default="data",
        help="RAG index data directory. Defaults to data.",
    )
    # --top-k: 最终返回给评测的检索结果数量
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of final retrieval results to evaluate.",
    )
    # --candidate-top-k: 重排序前的候选结果数量
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=10,
        help="Number of candidates before rerank. Defaults to 10.",
    )
    # --bm25: 是否在评测中启用 BM25 关键词检索
    parser.add_argument(
        "--bm25",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable BM25 during eval.",
    )
    # --cross-encoder: 是否启用 CrossEncoder 重排序
    parser.add_argument(
        "--cross-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CrossEncoder reranking during eval.",
    )
    # --embedding-provider: 嵌入模型的提供方
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Provider used by the embedding model.",
    )
    # --embedding-model: 嵌入模型的名称
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name.",
    )
    # --reranker-provider: 重排序模型的提供方
    parser.add_argument(
        "--reranker-provider",
        default=DEFAULT_RERANKER_PROVIDER,
        help="Provider used by reranking.",
    )
    # --reranker-model: 重排序模型的名称
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Reranker model name.",
    )
    # --reranker-device: 重排序模型运行设备（cpu/cuda/mps）
    parser.add_argument(
        "--reranker-device",
        default=None,
        help="Optional reranker device, such as cpu, cuda, or mps.",
    )
    # --reranker-batch-size: 重排序时每批处理的数量
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
        help="Batch size used by reranker predict().",
    )
    # --output: 可选，将评测报告输出为 JSON 文件
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report output path.",
    )
    # --trace: 是否启用 JSONL 追踪日志
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable JSONL tracing for eval runs.",
    )
    # --trace-dir: 追踪日志文件的存储目录
    parser.add_argument(
        "--trace-dir",
        default=DEFAULT_TRACE_DIR,
        help=f"Directory for JSONL trace files. Defaults to {DEFAULT_TRACE_DIR}.",
    )
    # --min-hit-rate: 最低命中率阈值，低于该值时评测视为失败
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Fail the eval run if hit_rate is below this value.",
    )
    # --min-mrr: 最低 MRR 阈值，低于该值时评测视为失败
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Fail the eval run if mrr is below this value.",
    )
    return parser.parse_args()


def main() -> None:
    """主函数：解析参数、初始化服务、运行评测、输出报告。"""
    args = parse_args()

    # 如果启用追踪，创建 TraceRecorder 实例；否则设为 None
    trace_recorder = (
        TraceRecorder(
            trace_dir=args.trace_dir,
            default_tags={"entrypoint": "eval_runner", "dataset": args.dataset},
        )
        if args.trace
        else None
    )

    # 初始化 RAG 服务：加载 FAISS 索引并配置检索参数
    rag = RAGService(
        data_dir=args.data_dir,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        reranker_provider=args.reranker_provider,
        reranker_model_name=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        default_use_bm25=args.bm25,
        default_use_rerank=args.cross_encoder,
        trace_recorder=trace_recorder,
    )

    # 对数据集中的每条查询执行检索评测，返回包含逐条结果和汇总指标的报告
    report = evaluate_retrieval_dataset(
        rag,
        args.dataset,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
        use_bm25=args.bm25,
        use_rerank=args.cross_encoder,
        trace_recorder=trace_recorder,
    )

    # 如果指定了 output 路径，将报告写入 JSON 文件
    if args.output:
        write_eval_report(report, args.output)

    # 在控制台打印评测汇总指标
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    # 如果启用了追踪，打印追踪文件路径及其摘要信息
    if trace_recorder is not None:
        print(f"trace: {trace_recorder.path}")
        print(
            "trace_summary: "
            + json.dumps(
                # 加载追踪文件并生成摘要（统计各事件类型的数量等）
                summarize_trace(load_trace(trace_recorder.path)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if args.output:
        print(f"report: {args.output}")

    # 根据 min-hit-rate 和 min-mrr 阈值检测评测是否达标，未达标则抛出 SystemExit
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

    # 如果有任何指标未达标，以非零退出码终止程序
    if failures:
        raise SystemExit("Eval failed: " + "; ".join(failures))


# 模块入口：直接运行此文件时调用 main()
if __name__ == "__main__":
    main()
