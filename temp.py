from rag_service import RAGService

if __name__ == "__main__":
    rag_service = RAGService()
    rag_service.add_documents(["docs/test.txt"])