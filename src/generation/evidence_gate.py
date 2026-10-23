from typing import List, Dict, Any


class EvidenceGate:

    def __init__(
        self,
        minimum_results: int = 1,
        minimum_relevance: float = 0.20
    ):
        """
        Decide whether retrieved evidence is strong
        enough to attempt generation.

        Parameters
        ----------
        minimum_results:
            Minimum number of retrieved chunks required.

        minimum_relevance:
            Minimum normalized retrieval relevance.
        """

        self.minimum_results = minimum_results

        self.minimum_relevance = (
            minimum_relevance
        )

    def check(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        retrieval_relevance: float
    ) -> Dict[str, Any]:
        """
        Determine whether evidence is sufficient.
        """

        if (
            len(retrieved_chunks)
            <
            self.minimum_results
        ):

            return {
                "sufficient": False,
                "reason": "No retrieved evidence"
            }

        if (
            retrieval_relevance
            <
            self.minimum_relevance
        ):

            return {
                "sufficient": False,
                "reason": (
                    "Retrieved evidence has "
                    "low relevance"
                )
            }

        return {
            "sufficient": True,
            "reason": "Evidence sufficient"
        }