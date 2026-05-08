from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_server import RAGService
from rag_server.model_factory import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_PROVIDER
from rag_server.rag_service import SUPPORTED_EXTENSIONS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local docs into the RAG index.")
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Directory containing .txt, .md, or .pdf documents.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where FAISS and metadata files are stored.",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep indexed documents that are no longer present in docs-dir.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Provider used by the embedding model.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name.",
    )
    return parser


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    project_root = PROJECT_ROOT
    os.chdir(project_root)

    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.is_absolute():
        docs_dir = project_root / docs_dir
    docs_dir = docs_dir.resolve()

    file_paths = [
        _project_relative(path, project_root)
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not file_paths:
        raise SystemExit(f"No supported documents found in {docs_dir}")

    rag = RAGService(
        data_dir=args.data_dir,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    result = rag.sync_documents(
        file_paths,
        remove_missing=not args.keep_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
