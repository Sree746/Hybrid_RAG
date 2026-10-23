
from src.embeddings.embedder import EmbeddingModel

from src.vectorstore.chroma_store import ChromaVectorStore

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.pipeline import RetrievalPipeline

from src.generation.generator import get_generator
from src.generation.rag_generator import RAGGenerator
from src.generation.citation_judge import CitationJudge

from src.rag_pipeline import RAGPipeline


def main():

    # =====================================================
    # STEP 1: LOAD THE EMBEDDING MODEL
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZING EMBEDDING MODEL")
    print("=" * 80)

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # =====================================================
    # STEP 2: CONNECT TO CHROMADB
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 2: CONNECTING TO CHROMADB")
    print("=" * 80)

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    document_count = vector_store.count()

    print(f"Chunks in ChromaDB: {document_count}")

    if document_count == 0:
        raise RuntimeError(
            "ChromaDB is empty. Add documents and run "
            "the ingestion pipeline before testing RAG."
        )

    # =====================================================
    # STEP 3: INITIALIZE DENSE RETRIEVER
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 3: INITIALIZING DENSE RETRIEVER")
    print("=" * 80)

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # =====================================================
    # STEP 4: INITIALIZE BM25 RETRIEVER
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 4: INITIALIZING BM25 RETRIEVER")
    print("=" * 80)

    sparse_retriever = BM25Retriever(
        vector_store=vector_store
    )

    # =====================================================
    # STEP 5: INITIALIZE HYBRID RETRIEVER
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 5: INITIALIZING HYBRID RETRIEVER")
    print("=" * 80)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60
    )

    # =====================================================
    # STEP 6: INITIALIZE RERANKER
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 6: INITIALIZING RERANKER")
    print("=" * 80)

    reranker = Reranker(
        model_name="BAAI/bge-reranker-base"
    )

    # =====================================================
    # STEP 7: BUILD RETRIEVAL PIPELINE
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 7: BUILDING RETRIEVAL PIPELINE")
    print("=" * 80)

    retrieval_pipeline = RetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker
    )

    # =====================================================
    # STEP 8: LOAD LOCAL GENERATION MODEL
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 8: LOADING LOCAL GENERATION MODEL")
    print("=" * 80)

    generator = get_generator(prefer="groq")

    rag_generator = RAGGenerator(
        generator=generator
    )

    # =====================================================
    # STEP 9: INITIALIZE CITATION JUDGE
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 9: INITIALIZING CITATION JUDGE")
    print("=" * 80)

    citation_judge = CitationJudge(
        generator=generator
    )

    # =====================================================
    # STEP 10: BUILD COMPLETE RAG PIPELINE
    # =====================================================

    print("\n" + "=" * 80)
    print("STEP 10: BUILDING COMPLETE RAG PIPELINE")
    print("=" * 80)

    rag_pipeline = RAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        rag_generator=rag_generator,
        citation_judge=citation_judge
    )

    print("RAG pipeline initialized successfully.")

    # =====================================================
    # STEP 11: ACCEPT USER QUESTIONS
    # =====================================================

    print("\n" + "=" * 80)
    print("RAG QUESTION-ANSWERING SYSTEM")
    print("=" * 80)

    print("Ask questions about your indexed documents.")
    print("Type 'exit' to stop.")

    while True:

        query = input("\nYour question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting RAG system.")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:

            # =============================================
            # STEP 12: RUN THE COMPLETE RAG PIPELINE
            # =============================================

            print("\nRetrieving evidence and generating answer...")

            result = rag_pipeline.ask(query)

            # =============================================
            # STEP 13: DISPLAY THE ANSWER
            # =============================================

            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)

            print(result["answer"])

            # =============================================
            # STEP 14: DISPLAY STATUS
            # =============================================

            print("\n" + "=" * 80)
            print("RELIABILITY METRICS")
            print("=" * 80)

            print(
                f"Abstained: "
                f"{result.get('abstained', False)}"
            )

            print(
                f"Retrieval relevance: "
                f"{result.get('retrieval_relevance', 0.0):.2%}"
            )

            print(
                f"Citation coverage: "
                f"{result.get('citation_coverage', 0.0):.2%}"
            )

            print(
                f"Citation accuracy: "
                f"{result.get('citation_accuracy', 0.0):.2%}"
            )

            print(
                f"Completeness: "
                f"{result.get('completeness', 0.0):.2%}"
            )

            print(
                f"Overall confidence: "
                f"{result.get('confidence', 0.0):.2%}"
            )

            if result.get("abstention_reason"):
                print(
                    f"Abstention reason: "
                    f"{result['abstention_reason']}"
                )

            # =============================================
            # STEP 15: DISPLAY CITATIONS
            # =============================================

            print("\n" + "=" * 80)
            print("CITATIONS")
            print("=" * 80)

            citations = result.get("citations", [])

            if citations:
                print(citations)
            else:
                print("No citations.")

            invalid_citations = result.get(
                "invalid_citations", []
            )

            if invalid_citations:
                print(
                    "WARNING: Invalid citation numbers:",
                    invalid_citations
                )

            # =============================================
            # STEP 16: DISPLAY CLAIM VERIFICATION
            # =============================================

            print("\n" + "=" * 80)
            print("CLAIM-LEVEL CITATION VERIFICATION")
            print("=" * 80)

            # The current RAGPipeline returns the verified
            # citation metrics but does not return the full
            # claim_verifications list. We display it if
            # you later expose that field in rag_pipeline.py.

            claim_verifications = result.get(
                "claim_verifications", []
            )

            if claim_verifications:

                for item in claim_verifications:

                    print(f"\nClaim: {item['claim']}")

                    print(
                        f"Citations: {item['citations']}"
                    )

                    print(
                        f"Supported: {item['supported']}"
                    )

            else:

                print(
                    "Claim-level details are not exposed "
                    "by the current RAGPipeline return value."
                )

            # =============================================
            # STEP 17: DISPLAY RETRIEVED SOURCES
            # =============================================

            print("\n" + "=" * 80)
            print("RETRIEVED EVIDENCE")
            print("=" * 80)

            retrieved_chunks = result.get(
                "retrieved_chunks", []
            )

            if not retrieved_chunks:

                print("No chunks were retrieved.")

            for index, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                metadata = chunk.get(
                    "metadata", {}
                )

                print(
                    f"\n[{index}] "
                    f"Source: {metadata.get('source', 'unknown')}"
                )

                print(
                    f"Chunk index: "
                    f"{metadata.get('chunk_index', 'unknown')}"
                )

                if "reranker_score" in chunk:
                    print(
                        f"Reranker score: "
                        f"{chunk['reranker_score']:.4f}"
                    )

                if "rrf_score" in chunk:
                    print(
                        
                        f"RRF score: "
                        f"{chunk['rrf_score']:.6f}"
                    )

                print("\nText:")

                print(
                    chunk.get("text", "")[:500]
                )

                print("-" * 80)

        except Exception as error:

            print("\nAn error occurred while answering the question.")
            print(f"Error type: {type(error).__name__}")
            print(f"Details: {error}")


if __name__ == "__main__":
    main()

