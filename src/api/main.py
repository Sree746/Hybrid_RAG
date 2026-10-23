
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

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

app = FastAPI(
    title="Hybrid RAG API",
    description="Question answering over indexed documents with hybrid retrieval and citation verification.",
    version="1.0.0",
)

RAW_DIR = Path("data/raw")
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Initialize shared components once when the API starts.
embedding_model = EmbeddingModel(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = ChromaVectorStore(
    persist_directory="./data/chroma",
    collection_name="internal_docs",
)

dense_retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)

sparse_retriever = BM25Retriever(
    vector_store=vector_store,
)

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    dense_weight=0.7,
    sparse_weight=0.3,
    rrf_k=60,
)

reranker = Reranker(
    model_name="BAAI/bge-reranker-base"
)

retrieval_pipeline = RetrievalPipeline(
    hybrid_retriever=hybrid_retriever,
    reranker=reranker,
)

generator = get_generator(prefer="groq")
rag_generator = RAGGenerator(generator=generator)
citation_judge = CitationJudge(generator=generator)

rag_pipeline = RAGPipeline(
    retrieval_pipeline=retrieval_pipeline,
    rag_generator=rag_generator,
    citation_judge=citation_judge,
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        description="Question to ask about the indexed documents",
    )


@app.get("/health")
def health():
    """Basic API health and index status."""
    return {
        "status": "ok",
        "indexed_chunks": vector_store.count(),
    }


@app.post("/v1/ask")
def ask(request: AskRequest):
    """Answer a question using the existing RAG pipeline."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        
        result = rag_pipeline.ask(request.question)
        
        retrieved_chunks = result.get("retrieved_chunks", [])
        citation_numbers = result.get("citations", [])
        sources = map_citations_to_sources(
            citation_numbers,
            retrieved_chunks,
        )
        print("Citation numbers:", result.get("citations"))

        chunks = result.get("retrieved_chunks", [])
        print("Number of retrieved chunks:", len(chunks))
        
        if chunks:
            print("First retrieved chunk keys:", chunks[0].keys())
            print("First retrieved chunk metadata:", chunks[0].get("metadata"))
            print("First retrieved chunk score:", chunks[0].get("reranker_score"))
            
        
        return {
            "query": result["query"],
            "answer": result["answer"],
            "citations": citation_numbers,
            "sources": sources,
            "invalid_citations": result["invalid_citations"],
            "citation_coverage": result["citation_coverage"],
            "citation_accuracy": result["citation_accuracy"],
            "retrieval_relevance": result["retrieval_relevance"],
            "completeness": result["completeness"],
            "confidence": result["confidence"],
            "abstained": result["abstained"],
            "abstention_reason": result["abstention_reason"],
        }



    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/v1/documents")
def list_documents():
    """List unique source files represented in the ChromaDB index."""
    try:
        records = vector_store.get_all()
        metadatas = records.get("metadatas", []) or []

        documents = {}

        for metadata in metadatas:
            metadata = metadata or {}
            source = (
                metadata.get("source")
                or metadata.get("source_file")
                or metadata.get("file_name")
                or metadata.get("filename")
            )

            if source:
                documents[source] = documents.get(source, 0) + 1

        return {
            "document_count": len(documents),
            "indexed_chunks": vector_store.count(),
            "documents": [
                {"source": source, "chunk_count": count}
                for source, count in sorted(documents.items())
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not list documents: {exc}",
        ) from exc


@app.post("/v1/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """Save uploaded documents and index them using the existing pipeline."""
    from src.ingestion.loaders import load_documents
    from src.ingestion.chunkers import chunk_document

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one file.",
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type for {upload.filename!r}. "
                        f"Allowed extensions: {sorted(ALLOWED_EXTENSIONS)}"
                    ),
                )

            # Use only the basename to prevent path traversal.
            safe_name = Path(upload.filename or "upload").name
            destination = RAW_DIR / safe_name

            content = await upload.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{safe_name} is empty.",
                )

            destination.write_bytes(content)
            saved_files.append(destination)

        documents = load_documents(str(RAW_DIR))

        # Reuse the same chunking strategy as the current ingestion script.
        chunks = []
        for document in documents:
            document_source = str(
                (document.metadata or {}).get("source", "")
            )

            # Only process files uploaded by this request.
            if not any(
                Path(document_source).name == path.name
                for path in saved_files
            ):
                continue

            chunks.extend(
                chunk_document(
                    document,
                    strategy="recursive",
                    chunk_size=1000,
                    chunk_overlap=200,
                )
            )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No chunks were produced. Check the file format "
                    "and the document loader's source metadata."
                ),
            )

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_model.embed_documents(
            texts,
            batch_size=32,
        )

        vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "status": "success",
            "uploaded_files": [path.name for path in saved_files],
            "chunks_created": len(chunks),
            "total_indexed_chunks": vector_store.count(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {type(exc).__name__}: {exc}",
        ) from exc



def map_citations_to_sources(citation_numbers, retrieved_chunks):
    """Map answer citation numbers to their actual retrieved chunks."""
    sources = []
    seen = set()

    for number in citation_numbers:
        if not isinstance(number, int):
            continue

        if number < 1 or number > len(retrieved_chunks):
            continue

        chunk = retrieved_chunks[number - 1]
        metadata = chunk.get("metadata") or {}

        source_name = (
            metadata.get("source")
            or chunk.get("source")
            or "unknown"
        )

        # Keep each citation number only once.
        if number in seen:
            continue
        seen.add(number)

        sources.append({
            "citation": number,
            "source": source_name,
            "chunk_index": metadata.get("chunk_index"),
            "score": chunk.get("reranker_score"),
            "text_preview": chunk.get("text", "")[:300],
        })

    return sources

