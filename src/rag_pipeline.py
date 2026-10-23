from typing import Dict, Any

from src.generation.rag_generator import (
    RAGGenerator
)

from src.generation.citation_verifier import (
    CitationVerifier
)

from src.generation.confidence import (
    ConfidenceScorer,
    estimate_completeness
)

from src.generation.evidence_gate import (
    EvidenceGate
)


class RAGPipeline:

    def __init__(
        self,
        retrieval_pipeline,
        rag_generator,
        citation_judge=None
    ):

        self.retrieval_pipeline = (
            retrieval_pipeline
        )

        self.rag_generator = (
            rag_generator
        )

        self.citation_verifier = (
            CitationVerifier(
                citation_judge=citation_judge
            )
        )

        self.confidence_scorer = (
            ConfidenceScorer()
        )

        self.evidence_gate = (
            EvidenceGate(
                minimum_results=1,
                minimum_relevance=0.20
            )
        )

    def ask(
        self,
        query: str
    ) -> Dict[str, Any]:

        # =================================================
        # 1. Retrieve
        # =================================================

        retrieved_chunks = (
            self.retrieval_pipeline.retrieve(

                query=query,

                dense_top_k=10,

                sparse_top_k=10,

                fusion_top_k=20,

                final_top_k=5
            )
        )

        # =================================================
        # 2. Calculate retrieval relevance
        # =================================================

        retrieval_relevance = (
            self.confidence_scorer
            .calculate_retrieval_relevance(
                retrieved_chunks
            )
        )

        # =================================================
        # 3. Evidence gate
        # =================================================

        evidence_result = (
            self.evidence_gate.check(

                retrieved_chunks=
                    retrieved_chunks,

                retrieval_relevance=
                    retrieval_relevance
            )
        )

        # =================================================
        # 4. Refuse to answer if evidence is weak
        # =================================================

        if not evidence_result[
            "sufficient"
        ]:

            return {

                "query": query,

                "answer": (
                    "I don't have enough "
                    "information in the provided "
                    "documents to answer this question."
                ),

                "citations": [],

                "invalid_citations": [],

                "citation_coverage": 0.0,

                "citation_accuracy": 0.0,

                "retrieval_relevance":
                    retrieval_relevance,

                "completeness": 0.0,

                "confidence":
                    min(retrieval_relevance, 0.2),

                "abstained": True,

                "abstention_reason":
                    evidence_result["reason"],

                "retrieved_chunks":
                    retrieved_chunks
            }

        # =================================================
        # 5. Generate
        # =================================================

        generation_result = (
            self.rag_generator.answer(

                query=query,

                retrieved_chunks=
                    retrieved_chunks
            )
        )

        if generation_result.get("generation_failed"):
            return {
                "query": query,
                "answer": generation_result["answer"],
                "citations": [],
                "invalid_citations": [],
                "citation_coverage": 0.0,
                "citation_accuracy": 0.0,
                "retrieval_relevance": retrieval_relevance,
                "completeness": 0.0,
                "confidence": min(retrieval_relevance, 0.2),
                "abstained": True,
                "abstention_reason": (
                    "Generator returned no substantive answer "
                    "(raw output: "
                    f"{generation_result.get('raw_answer')!r}). "
                    "Use a stronger model (GroqGenerator)."
                ),
                "retrieved_chunks": retrieved_chunks
            }

        answer = generation_result[
            "answer"
        ]

        # =================================================
        # 6. Verify citations
        # =================================================

        citation_result = (
            self.citation_verifier.verify(

                answer=answer,

                retrieved_chunks=
                    retrieved_chunks
            )
        )

        # =================================================
        # 7. Completeness
        # =================================================

        completeness = (
            estimate_completeness(

                answer=answer,

                query=query
            )
        )

        # =================================================
        # 8. Final confidence
        # =================================================

        confidence = (
            self.confidence_scorer
            .calculate_confidence(

                retrieval_relevance=
                    retrieval_relevance,

                citation_coverage=
                    citation_result[
                        "citation_coverage"
                    ],

                citation_accuracy=
                    citation_result[
                        "citation_accuracy"
                    ],

                completeness=
                    completeness
            )
        )

        if completeness < 0.5:
            confidence = min(confidence, 0.3)

        # =================================================
        # 9. Return final response
        # =================================================

        return {

            "query": query,

            "answer": answer,

            "citations":
                citation_result[
                    "citations"
                ],

            "invalid_citations":
                citation_result[
                    "invalid_citations"
                ],

            "citation_coverage":
                citation_result[
                    "citation_coverage"
                ],

            "citation_accuracy":
                citation_result[
                    "citation_accuracy"
                ],

            "retrieval_relevance":
                retrieval_relevance,

            "completeness":
                completeness,

            "confidence":
                confidence,

            "citation_method":
                generation_result.get(
                    "citation_method"
                ),
            "generation_attempts":
                generation_result.get(
                    "attempts"
                ),
            "claim_verifications":
                citation_result[
                    "claim_verifications"
                ],
            "abstained": answer.strip().lower().startswith(
                "i don't have enough information"
            ),

            "abstention_reason": (
                "The generated answer indicates insufficient evidence."
                if answer.strip().lower().startswith(
                    "i don't have enough information"
                )
                else None
            ),

            "retrieved_chunks":
                retrieved_chunks
        }