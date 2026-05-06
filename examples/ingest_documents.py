from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow running this example directly from the repository root.
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_server import RAGService

if __name__ == "__main__":
    rag_service = RAGService()
    result = rag_service.add_documents(
        [
            "docs/尺码推荐.txt",
            "docs/颜色选择.txt",
            "docs/洗涤养护.txt",
        ]
    )
    print(result)
