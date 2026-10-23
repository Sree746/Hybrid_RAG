from src.embeddings.embedder import EmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.dense_retriever import DenseRetriever


def main():

    # -------------------------------------------------
    # Load embedding model
    # -------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # -------------------------------------------------
    # Connect to existing ChromaDB
    # -------------------------------------------------

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    print(
        f"\nDocuments in database: "
        f"{vector_store.count()}"
    )

    # -------------------------------------------------
    # Create retriever
    # -------------------------------------------------

    retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # -------------------------------------------------
    # User query
    # -------------------------------------------------

    query = input(
        "\nEnter your question: "
    )

    # -------------------------------------------------
    # Retrieve top 10
    # -------------------------------------------------

    results = retriever.retrieve(
        query=query,
        top_k=10
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("\n" + "=" * 80)
    print("DENSE RETRIEVAL RESULTS")
    print("=" * 80)

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
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
