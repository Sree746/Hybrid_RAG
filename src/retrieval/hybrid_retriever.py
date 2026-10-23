from typing import List, Dict, Any


class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.

        Parameters
        ----------
        dense_retriever:
            Semantic vector retriever.

        sparse_retriever:
            BM25 retriever.

        dense_weight:
            Weight assigned to dense retrieval.

        sparse_weight:
            Weight assigned to BM25.

        rrf_k:
            RRF smoothing constant.
        """

        if dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative."
            )

        if sparse_weight < 0:
            raise ValueError(
                "sparse_weight cannot be negative."
            )

        if dense_weight + sparse_weight == 0:
            raise ValueError(
                "At least one retrieval weight "
                "must be greater than zero."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        self.dense_retriever = dense_retriever

        self.sparse_retriever = sparse_retriever

        self.dense_weight = dense_weight

        self.sparse_weight = sparse_weight

        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        final_top_k: int = 20
    ) -> List[Dict[str, Any]]:

        # =================================================
        # 1. Dense retrieval
        # =================================================

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=dense_top_k
            )
        )

        # =================================================
        # 2. Sparse retrieval
        # =================================================

        sparse_results = (
            self.sparse_retriever.retrieve(
                query=query,
                top_k=sparse_top_k
            )
        )

        # =================================================
        # 3. Store fused scores
        # =================================================

        fused_scores = {}

        chunk_data = {}

        # =================================================
        # 4. Add dense RRF scores
        # =================================================

        for result in dense_results:

            chunk_id = result["id"]

            rank = result["rank"]

            score = (
                self.dense_weight
                /
                (self.rrf_k + rank)
            )

            fused_scores[chunk_id] = (
                fused_scores.get(
                    chunk_id,
                    0.0
                ) + score
            )

            chunk_data[chunk_id] = result

        # =================================================
        # 5. Add sparse RRF scores
        # =================================================

        for result in sparse_results:

            chunk_id = result["id"]

            rank = result["rank"]

            score = (
                self.sparse_weight
                /
                (self.rrf_k + rank)
            )

            fused_scores[chunk_id] = (
                fused_scores.get(
                    chunk_id,
                    0.0
                ) + score
            )

            if chunk_id not in chunk_data:

                chunk_data[chunk_id] = result

        # =================================================
        # 6. Sort by fused score
        # =================================================

        ranked_chunks = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # =================================================
        # 7. Create final results
        # =================================================

        final_results = []

        for rank, (
            chunk_id,
            fused_score
        ) in enumerate(
            ranked_chunks[:final_top_k],
            start=1
        ):

            result = chunk_data[chunk_id].copy()

            result["rank"] = rank

            result["rrf_score"] = fused_score

            final_results.append(result)

        return final_results