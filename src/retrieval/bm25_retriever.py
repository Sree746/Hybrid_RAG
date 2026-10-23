import re

from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from ..vectorstore.chroma_store import ChromaVectorStore


class BM25Retriever:

    def __init__(
        self,
        vector_store: ChromaVectorStore
    ):
        """
        Initialize BM25 retriever.

        BM25 works on text rather than embeddings.
        """

        self.vector_store = vector_store

        # Load all documents from ChromaDB
        data = vector_store.get_all()

        self.documents = data["documents"]

        self.metadatas = data["metadatas"]

        self.ids = data["ids"]

        # Tokenize every document
        self.tokenized_documents = [
            self._tokenize(document)
            for document in self.documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Convert text into tokens.
        """

        text = text.lower()

        tokens = re.findall(
            r"\b\w+\b",
            text
        )

        return tokens

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        top_indices = ranked_indices[:top_k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            results.append({

                "rank": rank,

                "id": self.ids[index],

                "text": self.documents[index],

                "metadata": self.metadatas[index],

                "score": float(scores[index])
            })

        return results