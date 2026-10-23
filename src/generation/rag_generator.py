from typing import List, Dict, Any

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .citations import (find_citations, normalize_citations, auto_attribute,
                        validate_citations, is_substantive)

NO_INFO = ("I don't have enough information in the provided documents "
           "to answer this question.")


class RAGGenerator:
    def __init__(self, generator, max_retries: int = 1, min_citation_ratio: float = 0.6):
        self.generator = generator
        self.max_retries = max_retries
        self.min_citation_ratio = min_citation_ratio

    def _call(self, query, chunks, strict):
        user = build_user_prompt(query, chunks, strict=strict)
        if getattr(self.generator, "supports_system", False):
            return user, self.generator.generate(prompt=user, system=SYSTEM_PROMPT)
        from .prompt import build_grounded_prompt
        prompt = build_grounded_prompt(query, chunks, strict=strict)
        return prompt, self.generator.generate(prompt=prompt)

    def _cited_ratio(self, answer: str) -> float:
        from .citations import calculate_citation_coverage
        return calculate_citation_coverage(answer)

    def answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not retrieved_chunks:
            return {"answer": NO_INFO, "citations": [], "prompt": None, "attempts": 0,
                    "citation_method": "none"}

        n = len(retrieved_chunks)
        prompt, answer, method, attempts = None, "", "model", 0
        for attempt in range(self.max_retries + 1):
            attempts += 1
            prompt, answer = self._call(query, retrieved_chunks, strict=attempt > 0)
            answer = normalize_citations(answer)
            v = validate_citations(answer, n)
            ok = (is_substantive(answer) and find_citations(answer) and v["all_valid"]
                  and self._cited_ratio(answer) >= self.min_citation_ratio)
            if ok or answer.strip().startswith("I don't have enough information"):
                break
        else:
            if not is_substantive(answer):
                return {"answer": NO_INFO, "citations": [], "prompt": prompt, "attempts": attempts,
                        "citation_method": "none", "generation_failed": True,
                        "raw_answer": answer}
            # model still ignored the rule -> attach citations by lexical match
            answer = auto_attribute(answer, retrieved_chunks)
            method = "auto_attributed"

        return {"answer": answer, "citations": retrieved_chunks, "prompt": prompt,
                "attempts": attempts, "citation_method": method}
