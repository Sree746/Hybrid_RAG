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

from src.generation.generator import (
    LocalGenerator
)

from src.generation.rag_generator import (
    RAGGenerator
)


def main():

    # =================================================
    # Embedding model
    # =================================================

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # =================================================
    # Vector database
    # =================================================

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    # =================================================
    # Dense retrieval
    # =================================================

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # =================================================
    # BM25
    # =================================================

    sparse_retriever = BM25Retriever(
        vector_store=vector_store
    )

    # =================================================
    # Hybrid retrieval
    # =================================================

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60
    )

    # =================================================
    # Reranker
    # =================================================

    reranker = Reranker(
        model_name="BAAI/bge-reranker-base"
    )

    # =================================================
    # Retrieval pipeline
    # =================================================

    retrieval_pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker
    )

    # =================================================
    # Generator
    # =================================================

    generator = LocalGenerator(
        model_name="google/flan-t5-base"
    )

    rag_generator = RAGGenerator(
        generator=generator
    )

    # =================================================
    # User query
    # =================================================

    query = input(
        "\nEnter your question: "
    )

    # =================================================
    # Retrieve
    # =================================================

    retrieved_chunks = (
        retrieval_pipeline.retrieve(

            query=query,

            dense_top_k=10,

            sparse_top_k=10,

            fusion_top_k=20,

            final_top_k=5
        )
    )

    # =================================================
    # Generate
    # =================================================

    response = rag_generator.answer(

        query=query,

        retrieved_chunks=retrieved_chunks
    )

    # =================================================
    # Display
    # =================================================

    print("\n" + "=" * 80)

    print("ANSWER")

    print("=" * 80)

    print(
        response["answer"]
    )

    print("\n" + "=" * 80)

    print("SOURCES")

    print("=" * 80)

    for index, chunk in enumerate(
        response["citations"],
        start=1
    ):

        metadata = chunk["metadata"]

        print(
            f"[{index}] "
            f"{metadata.get('source')}"
        )

        print(
            chunk["text"][:300]
        )

        print("-" * 80)


if __name__ == "__main__":
    main()