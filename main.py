from src.ingestion.loaders import load_documents
from src.ingestion.chunkers import chunk_document

from src.embeddings.embedder import EmbeddingModel

from src.vectorstore.chroma_store import ChromaVectorStore


def main():

    # =========================================================
    # STEP 1: LOAD DOCUMENTS
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 60)

    documents = load_documents(
        "data/raw"
    )

    print(
        f"\nTotal documents loaded: {len(documents)}"
    )


    # =========================================================
    # STEP 2: CHUNK DOCUMENTS
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("=" * 60)

    all_chunks = []

    for document in documents:

        chunks = chunk_document(
            document,
            strategy="recursive",
            chunk_size=1000,
            chunk_overlap=200
        )

        all_chunks.extend(chunks)

        print(
            f"{document.metadata['source']}: "
            f"{len(chunks)} chunks"
        )

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )


    # =========================================================
    # STEP 3: CREATE EMBEDDINGS
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 3: CREATING EMBEDDINGS")
    print("=" * 60)

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-small-en-v1.5"
    )

    chunk_texts = [
        chunk.text
        for chunk in all_chunks
    ]

    embeddings = embedding_model.embed_documents(
        chunk_texts,
        batch_size=32
    )

    print(
        f"\nNumber of embeddings: "
        f"{len(embeddings)}"
    )

    if embeddings:

        print(
            f"Embedding dimension: "
            f"{len(embeddings[0])}"
        )


    # =========================================================
    # STEP 4: STORE IN CHROMADB
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 4: STORING IN CHROMADB")
    print("=" * 60)

    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma",
        collection_name="internal_docs"
    )

    vector_store.add_chunks(
        chunks=all_chunks,
        embeddings=embeddings
    )

    print(
        f"\nTotal chunks in ChromaDB: "
        f"{vector_store.count()}"
    )


if __name__ == "__main__":
    main()