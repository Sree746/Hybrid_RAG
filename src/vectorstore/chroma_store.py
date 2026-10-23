import chromadb

from typing import List, Dict, Any
from ..ingestion.models import Chunk


class ChromaVectorStore:

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "internal_docs"
    ):
        """
        Initialize persistent ChromaDB.
        """

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        print("Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

        print(
            f"ChromaDB collection '{collection_name}' "
            "initialized successfully."
        )

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ):
        """
        Add chunks and embeddings to ChromaDB.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be equal."
            )

        if not chunks:
            print("No chunks to add.")
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            source = chunk.metadata.get(
                "source",
                "unknown"
            )

            chunk_index = chunk.metadata.get(
                "chunk_index",
                0
            )

            strategy = chunk.metadata.get("chunking_strategy", "fixed")
            chunk_id = f"{source}::{strategy}::{chunk_index}"

            ids.append(chunk_id)

            documents.append(chunk.text)

            metadatas.append(chunk.metadata)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(
            f"Added/updated {len(chunks)} chunks "
            "in ChromaDB."
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Perform dense vector search.

        Parameters
        ----------
        query_embedding:
            Embedding of the user's query.

        top_k:
            Number of results to retrieve.

        Returns
        -------
        Dictionary containing retrieved documents,
        metadata, distances and IDs.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results

    def count(self) -> int:
        """
        Return number of stored chunks.
        """

        return self.collection.count()

    def get_all(self):
        """
        Retrieve all stored records.
        """

        return self.collection.get()

    def delete_collection(self):
        """
        Delete the current collection.
        """

        self.client.delete_collection(
            name=self.collection_name
        )

        print(
            f"Collection '{self.collection_name}' deleted."
        )