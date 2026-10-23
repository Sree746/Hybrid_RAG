from typing import List, Dict, Any


class RetrievalPipeline:

    def __init__(
        self,
        hybrid_retriever,
        reranker
    ):
        """
        Combine hybrid retrieval and reranking.
        """

        self.hybrid_retriever = hybrid_retriever

        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        fusion_top_k: int = 20,
        final_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline:

        Dense + BM25
             ↓
           RRF
             ↓
          Top 20
             ↓
         Reranker
             ↓
           Top 5
        """

        # =================================================
        # STEP 1: Hybrid retrieval
        # =================================================

        hybrid_results = (
            self.hybrid_retriever.retrieve(

                query=query,

                dense_top_k=dense_top_k,

                sparse_top_k=sparse_top_k,

                final_top_k=fusion_top_k
            )
        )

        # =================================================
        # STEP 2: Reranking
        # =================================================

        reranked_results = (
            self.reranker.rerank(

                query=query,

                documents=hybrid_results,

                top_k=final_top_k
            )
        )

        return reranked_results