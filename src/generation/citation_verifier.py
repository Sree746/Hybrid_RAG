import re

from typing import List, Dict, Any

from .claim_extractor import (
    split_into_claims
)

from .citations import (
    find_citations,
    is_substantive,
    strip_citations,
    extract_citation_numbers,
    validate_citations,
    calculate_citation_coverage
)


class CitationVerifier:

    def __init__(
        self,
        citation_judge=None
    ):
        """
        Initialize citation verifier.

        citation_judge:
            Optional LLM-based judge used to verify
            whether citations actually support claims.
        """

        self.citation_judge = citation_judge

    def extract_claim_citations(
        self,
        answer: str
    ) -> List[Dict[str, Any]]:
        """
        Extract claims and citations associated with
        each sentence.
        """

        claims = split_into_claims(
            answer
        )

        results = []

        for claim in claims:

            citations = list(dict.fromkeys(find_citations(claim)))

            results.append({
                "claim": claim,
                "citations": citations
            })

        return results

    def verify(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform citation verification.
        """

        number_of_sources = len(
            retrieved_chunks
        )

        # =================================================
        # 1. Check citation numbers
        # =================================================

        citation_validation = (
            validate_citations(
                answer=answer,
                number_of_sources=number_of_sources
            )
        )

        # =================================================
        # 2. Calculate citation coverage
        # =================================================

        coverage = (
            calculate_citation_coverage(
                answer
            )
        )

        # =================================================
        # 3. Extract claim-citation pairs
        # =================================================

        claim_citations = (
            self.extract_claim_citations(
                answer
            )
        )

        claim_verifications = []

        # =================================================
        # 4. Verify every cited claim
        # =================================================

        for item in claim_citations:

            claim = item["claim"]

            citations = item["citations"]

            # Citation marker with no actual claim text, e.g. just "[1]"
            if not is_substantive(claim, min_words=3):
                claim_verifications.append({
                    "claim": claim,
                    "citations": citations,
                    "supported": False,
                    "reason": "Empty claim (citation only)"
                })
                continue

            # No citation attached to claim
            if not citations:

                claim_verifications.append({
                    "claim": claim,
                    "citations": [],
                    "supported": False,
                    "reason": "No citation"
                })

                continue

            supported = False

            evidence_results = []

            for citation_number in citations:

                # Skip invalid citations
                if (
                    citation_number < 1
                    or
                    citation_number >
                    number_of_sources
                ):

                    evidence_results.append({
                        "citation": citation_number,
                        "verdict": "INVALID"
                    })

                    continue

                chunk = retrieved_chunks[
                    citation_number - 1
                ]

                evidence = chunk["text"]

                # -------------------------------------------------
                # If no LLM judge exists, we only know the citation
                # reference is valid.
                # -------------------------------------------------

                if self.citation_judge is None:

                    evidence_results.append({
                        "citation": citation_number,
                        "verdict": "NOT_VERIFIED"
                    })

                    continue

                # -------------------------------------------------
                # LLM judge
                # -------------------------------------------------

                verification = self.citation_judge.verify_claim(
                    claim=strip_citations(claim),
                    evidence=evidence
                )
                
                print("\n========== CITATION VERIFICATION ==========")
                print("Claim:", strip_citations(claim))
                print("Citation number:", citation_number)
                print("Chunk index:", citation_number - 1)
                print("Chunk metadata:", {
                    key: value
                    for key, value in chunk.items()
                    if key != "text"
                })
                print("Evidence sent to judge:", repr(evidence[:1000]))
                print("Judge verdict:", verification.get("verdict"))
                print("Raw judge output:", repr(
                    verification.get("raw_verdict", "NOT EXPOSED")
                ))
                print("===========================================\n")
                
                evidence_results.append({
                    "citation": citation_number,
                    "verdict": verification["verdict"]
                })
                
                if verification["verdict"] == "SUPPORTED":
                    supported = True

            claim_verifications.append({
                "claim": claim,
                "citations": citations,
                "supported": supported,
                "evidence_results": evidence_results
            })

        # =================================================
        # 5. Calculate citation accuracy
        # =================================================

        cited_claims = [
            item
            for item in claim_verifications
            if item["citations"]
        ]

        if not cited_claims:

            citation_accuracy = 0.0

        else:

            supported_claim_count = sum(
                1
                for item in cited_claims
                if item["supported"]
            )

            citation_accuracy = (
                supported_claim_count
                /
                len(cited_claims)
            )

        return {

            "citations":
                citation_validation[
                    "citations"
                ],

            "invalid_citations":
                citation_validation[
                    "invalid_citations"
                ],

            "all_citations_valid":
                citation_validation[
                    "all_valid"
                ],

            "citation_coverage":
                coverage,

            "claim_verifications":
                claim_verifications,

            "citation_accuracy":
                citation_accuracy
        }