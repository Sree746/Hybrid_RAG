from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever


def main():

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    print(
        f"Documents in database: "
        f"{vector_store.count()}"
    )

    retriever = BM25Retriever(
        vector_store=vector_store
    )

    query = input(
        "\nEnter your question: "
    )

    results = retriever.retrieve(
        query=query,
        top_k=10
    )

    print("\n" + "=" * 80)
    print("BM25 RETRIEVAL RESULTS")
    print("=" * 80)

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['metadata'].get('source')}"
        )

        print(
            f"Chunk: "
            f"{result['metadata'].get('chunk_index')}"
        )

        print("\nText:")

        print(
            result["text"]
        )

        print("-" * 80)


if __name__ == "__main__":
    main()