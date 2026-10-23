"""Reusable metric helpers; adapt metadata fields to the current retriever schema."""
from collections import Counter
from typing import Any, Dict, Iterable, List


def token_f1(prediction: str, reference: str) -> float:
    """Lexical baseline only; use an LLM judge for semantic correctness."""
    p, r = prediction.lower().split(), reference.lower().split()
    if not p and not r: return 1.0
    if not p or not r: return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0: return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def retrieval_relevance_at_k(chunks: List[Dict[str, Any]], relevant_files: Iterable[str], k: int = 5) -> float:
    """Fraction of top-k chunks from a gold relevant source file."""
    relevant, top = set(relevant_files), chunks[:k]
    if not relevant or not top: return 0.0
    hits = 0
    for chunk in top:
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source_file") or meta.get("source") or meta.get("file_name")
        if source in relevant: hits += 1
    return hits / len(top)


def citation_accuracy(verifications: List[Dict[str, Any]]) -> float:
    """Unknown/empty verdicts are not supported; no-citation claims excluded."""
    cited = [v for v in verifications if v.get("citations")]
    if not cited: return 0.0
    return sum(v.get("supported") is True for v in cited) / len(cited)


def abstention_correct(answerable: bool, abstained: bool) -> bool:
    return (not abstained) if answerable else bool(abstained)
