import re
from typing import Dict, Any


class CitationJudge:

    def __init__(self, generator):
        self.generator = generator

    def build_prompt(self, claim: str, evidence: str) -> str:
        return f"""
You are a strict citation verification system.

Your task is to determine whether the supplied evidence supports
the supplied claim.

Rules:
1. Use only the supplied evidence.
2. Mark SUPPORTED if the evidence directly or logically supports
   the claim.
3. Mark NOT_SUPPORTED if the evidence does not support the claim.
4. Do not reject a claim merely because the evidence uses different
   wording from the claim.
5. Do not use outside knowledge.
6. Output exactly one label. Do not explain your decision.

CLAIM:
{claim}

EVIDENCE:
{evidence}

VERDICT:
"""

    
    def verify_claim(
        self,
        claim: str,
        evidence: str
    ) -> Dict[str, Any]:
    
        prompt = self.build_prompt(
            claim=claim,
            evidence=evidence
        )
    
        # First attempt: normal verification
        verdict = self.generator.generate(
            prompt=prompt,
            max_new_tokens=256
        )
    
        normalized = verdict.strip().upper().replace("-", "_")
    
        # Retry once if the model returned no usable verdict
        if not normalized.startswith(("SUPPORTED", "NOT_SUPPORTED")):
            retry_prompt = prompt + """
    IMPORTANT: Respond with exactly one token label:
    SUPPORTED or NOT_SUPPORTED.
    Do not explain your reasoning.
    VERDICT:
    """
    
            verdict = self.generator.generate(
                prompt=retry_prompt,
                max_new_tokens=512
            )
    
            normalized = verdict.strip().upper().replace("-", "_")
    
        if normalized.startswith("NOT_SUPPORTED"):
            label = "NOT_SUPPORTED"
        elif normalized.startswith("SUPPORTED"):
            label = "SUPPORTED"
        else:
            label = "UNKNOWN"
    
        return {
            "claim": claim,
            "evidence": evidence,
            "verdict": label,
            "raw_verdict": verdict
        }
