from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base"
    ):
        """
        Initialize the cross-encoder reranker.

        Parameters
        ----------
        model_name:
            Name of the pretrained reranking model.
        """

        self.model_name = model_name

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Reranker loaded successfully."
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents according to
        query-document relevance.

        Parameters
        ----------
        query:
            Original user query.

        documents:
            Candidate chunks returned by hybrid retrieval.

        top_k:
            Number of final chunks to keep.

        Returns
        -------
        List of reranked documents.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # -------------------------------------------------
        # Create query-document pairs
        # -------------------------------------------------

        pairs = []

        for document in documents:

            pairs.append(
                (
                    query,
                    document["text"]
                )
            )

        # -------------------------------------------------
        # Calculate relevance scores
        # -------------------------------------------------

        scores = self.model.predict(
            pairs,
            show_progress_bar=True
        )

        # -------------------------------------------------
        # Attach reranker scores
        # -------------------------------------------------

        reranked_documents = []

        for document, score in zip(
            documents,
            scores
        ):

            result = document.copy()

            result["reranker_score"] = float(
                score
            )

            reranked_documents.append(
                result
            )

        # -------------------------------------------------
        # Sort by reranker score
        # -------------------------------------------------

        reranked_documents.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        # -------------------------------------------------
        # Keep only top-k
        # -------------------------------------------------

        final_documents = (
            reranked_documents[:top_k]
        )

        # -------------------------------------------------
        # Reassign rank after reranking
        # -------------------------------------------------

        for rank, document in enumerate(
            final_documents,
            start=1
        ):

            document["rank"] = rank

        return final_documents