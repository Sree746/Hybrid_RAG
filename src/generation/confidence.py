from typing import List, Dict, Any


class ConfidenceScorer:

    def calculate_retrieval_relevance(
        self,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> float:

        if not retrieved_chunks:
            return 0.0

        import math

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

        # bge-reranker outputs raw logits; sigmoid gives an absolute 0-1 relevance
        # (min-max over the top-k always pins the best chunk at 1.0, so weak
        # retrievals were never detected).
        if not any("reranker_score" in c for c in retrieved_chunks):
            return 0.0
        probs = [sigmoid(c.get("reranker_score", -30.0)) for c in retrieved_chunks]
        weights = [1.0 / (i + 1) for i in range(len(probs))]
        return sum(p * w for p, w in zip(probs, weights)) / sum(weights)

    def calculate_confidence(
        self,
        retrieval_relevance: float,
        citation_coverage: float,
        citation_accuracy: float,
        completeness: float
    ) -> float:
        """
        Calculate final confidence.

        Weights:

        Retrieval relevance = 35%
        Citation coverage   = 25%
        Citation accuracy   = 20%
        Completeness        = 20%
        """

        confidence = (

            0.35 * retrieval_relevance

            +

            0.25 * citation_coverage

            +

            0.20 * citation_accuracy

            +

            0.20 * completeness
        )

        return max(
            0.0,
            min(
                1.0,
                confidence
            )
        )


def estimate_completeness(
    answer: str,
    query: str
) -> float:

    if not answer.strip():
        return 0.0

    insufficient_phrases = [
        "i don't have enough information",
        "not enough information",
        "cannot answer",
        "insufficient information"
    ]

    answer_lower = answer.lower()

    for phrase in insufficient_phrases:

        if phrase in answer_lower:

            return 0.0

    word_count = len(
        answer.split()
    )

    if word_count < 10:

        return 0.3

    elif word_count < 30:

        return 0.6

    else:

        return 1.0