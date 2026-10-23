from typing import List, Dict, Any

from ..embeddings.embedder import EmbeddingModel
from ..vectorstore.chroma_store import ChromaVectorStore


class DenseRetriever:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore
    ):
        """
        Initialize dense retriever.

        Parameters
        ----------
        embedding_model:
            Model used to convert queries into embeddings.

        vector_store:
            ChromaDB vector store.
        """

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most semantically similar chunks.

        Parameters
        ----------
        query:
            User's question.

        top_k:
            Number of chunks to retrieve.

        Returns
        -------
        List of retrieved chunks.
        """

        # -------------------------------------------------
        # STEP 1: Convert query into embedding
        # -------------------------------------------------

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        # -------------------------------------------------
        # STEP 2: Search ChromaDB
        # -------------------------------------------------

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # -------------------------------------------------
        # STEP 3: Convert Chroma's response into a
        # convenient list of dictionaries
        # -------------------------------------------------

        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for rank, (
            document,
            metadata,
            distance,
            chunk_id
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
                ids
            ),
            start=1
        ):

            retrieved_chunks.append({

                "rank": rank,

                "id": chunk_id,

                "text": document,

                "metadata": metadata,

                "distance": distance
            })

        return retrieved_chunks