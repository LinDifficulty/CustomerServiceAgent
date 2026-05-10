from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 定位项目根目录（本文件位于 examples/ 子目录下，向上两级即为项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 确保项目根目录在 sys.path 中，以便导入 rag_server 包
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# noqa: E402 用于抑制 import-not-at-top-of-file 的 flake8 警告
# 因为在修改 sys.path 之后才能正确导入这些模块
from rag_server import RAGService  # noqa: E402
from rag_server.model_factory import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,  # 默认 Embedding 模型名称
    DEFAULT_EMBEDDING_PROVIDER,  # 默认 Embedding 提供商（如 dashscope）
)
from rag_server.rag_service import SUPPORTED_EXTENSIONS  # noqa: E402  # 支持的文件扩展名（.txt, .md, .pdf）


def build_arg_parser() -> argparse.ArgumentParser:
    # 构建命令行参数解析器，支持自定义文档目录、数据目录和 Embedding 配置
    parser = argparse.ArgumentParser(description="Ingest local docs into the RAG index.")
    parser.add_argument(
        "--docs-dir",
        default="docs",  # 默认从项目根目录下的 docs/ 读取文档
        help="Directory containing .txt, .md, or .pdf documents.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",  # 默认数据存储在项目根目录下的 data/
        help="Directory where FAISS and metadata files are stored.",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",  # 开启后会保留索引中已有但 docs-dir 中已删除的文档
        help="Keep indexed documents that are no longer present in docs-dir.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,  # 来自 model_factory 的默认值
        help="Provider used by the embedding model.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,  # 来自 model_factory 的默认值
        help="Embedding model name.",
    )
    return parser


def _project_relative(path: Path, project_root: Path) -> str:
    # 将绝对路径转为相对于项目根目录的 POSIX 格式路径，便于输出展示
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        # 如果路径不在项目根目录下（如符号链接指向外部），返回原始路径
        return str(path)


def main(argv: list[str] | None = None) -> None:
    # 主函数：扫描文档目录、创建 RAGService、同步文档到知识库索引
    args = build_arg_parser().parse_args(argv)  # 解析命令行参数
    project_root = PROJECT_ROOT
    os.chdir(project_root)  # 切换到项目根目录，确保相对路径正常工作

    # 处理文档目录路径：支持 ~ 展开和相对路径转绝对路径
    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.is_absolute():
        docs_dir = project_root / docs_dir  # 相对路径基于项目根目录
    docs_dir = docs_dir.resolve()  # 解析符号链接等，得到规范路径

    # 递归扫描文档目录，筛选出支持的文件类型（.txt, .md, .pdf）
    # 将路径转为相对于项目根目录的格式，便于在索引中显示
    file_paths = [
        _project_relative(path, project_root)
        for path in sorted(docs_dir.rglob("*"))  # 递归遍历、按文件名排序
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS  # 只保留支持的文件类型
    ]
    if not file_paths:
        # 如果没有找到任何支持的文档，直接退出并报错
        raise SystemExit(f"No supported documents found in {docs_dir}")

    # 创建 RAGService 实例，传入数据目录和 Embedding 配置
    rag = RAGService(
        data_dir=args.data_dir,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    # 同步文档：新增或更新已存在的文档，可选是否删除不再存在的文档
    # remove_missing=True（默认）会清理索引中已删除的文档
    result = rag.sync_documents(
        file_paths,
        remove_missing=not args.keep_missing,  # --keep-missing 开启时保留缺失文档
    )
    # 以格式化 JSON 输出同步结果（包含 added/updated/removed 统计）
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 直接执行脚本时启动文档批量入库流程
    main()
