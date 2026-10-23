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

from src.retrieval.reranker import (
    Reranker
)

from src.retrieval.pipeline import (
    RetrievalPipeline
)


def main():

    # =================================================
    # 1. Embedding model
    # =================================================

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # =================================================
    # 2. Vector store
    # =================================================

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    print(
        f"\nDocuments in database: "
        f"{vector_store.count()}"
    )

    # =================================================
    # 3. Dense retriever
    # =================================================

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # =================================================
    # 4. BM25 retriever
    # =================================================

    sparse_retriever = BM25Retriever(
        vector_store=vector_store
    )

    # =================================================
    # 5. Hybrid retriever
    # =================================================

    hybrid_retriever = HybridRetriever(

        dense_retriever=dense_retriever,

        sparse_retriever=sparse_retriever,

        dense_weight=0.7,

        sparse_weight=0.3,

        rrf_k=60
    )

    # =================================================
    # 6. Reranker
    # =================================================

    reranker = Reranker(
        model_name="BAAI/bge-reranker-base"
    )

    # =================================================
    # 7. Complete pipeline
    # =================================================

    pipeline = RetrievalPipeline(

        hybrid_retriever=hybrid_retriever,

        reranker=reranker
    )

    # =================================================
    # 8. Query
    # =================================================

    query = input(
        "\nEnter your question: "
    )

    # =================================================
    # 9. Retrieve
    # =================================================

    results = pipeline.retrieve(

        query=query,

        dense_top_k=10,

        sparse_top_k=10,

        fusion_top_k=20,

        final_top_k=5
    )

    # =================================================
    # 10. Display
    # =================================================

    print("\n" + "=" * 80)

    print(
        "FINAL RERANKED RESULTS"
    )

    print("=" * 80)

    for result in results:

        print(
            f"\nFinal Rank: "
            f"{result['rank']}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
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