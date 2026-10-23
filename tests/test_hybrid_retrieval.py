from src.embeddings.embedder import EmbeddingModel

from src.vectorstore.chroma_store import (
    ChromaVectorStore
)

from src.retrieval.dense_retriever import (
    DenseRetriever
)

from src.retrieval.bm25_retriever import (
    BM25Retriever
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever
)


def main():

    # =================================================
    # Embedding model
    # =================================================

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # =================================================
    # ChromaDB
    # =================================================

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    # =================================================
    # Dense retriever
    # =================================================

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # =================================================
    # BM25 retriever
    # =================================================

    sparse_retriever = BM25Retriever(
        vector_store=vector_store
    )

    # =================================================
    # Hybrid retriever
    # =================================================

    hybrid_retriever = HybridRetriever(

        dense_retriever=dense_retriever,

        sparse_retriever=sparse_retriever,

        dense_weight=0.7,

        sparse_weight=0.3,

        rrf_k=60
    )

    # =================================================
    # Query
    # =================================================

    query = input(
        "\nEnter your question: "
    )

    # =================================================
    # Retrieve
    # =================================================

    results = hybrid_retriever.retrieve(

        query=query,

        dense_top_k=10,

        sparse_top_k=10,

        final_top_k=20
    )

    # =================================================
    # Display
    # =================================================

    print("\n" + "=" * 80)

    print(
        "HYBRID RETRIEVAL RESULTS"
    )

    print("=" * 80)

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
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