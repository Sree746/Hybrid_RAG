"""Offline tests (no models, no network) for the citation fixes."""
from src.generation.citations import (
    find_citations, normalize_citations, auto_attribute, validate_citations,
    calculate_citation_coverage,
)
from src.generation.rag_generator import RAGGenerator
from src.generation.confidence import ConfidenceScorer

CHUNKS = [
    {"text": "BM25 ranks documents using term frequency and inverse document frequency."},
    {"text": "Reciprocal rank fusion merges ranked lists from dense and sparse retrieval."},
]


def test_parser_variants():
    assert find_citations("A [1]. B [1, 2]. C [Source 2]. D [1][2].") == [1, 1, 2, 2, 1, 2]
    assert normalize_citations("X [1, 2].") == "X [1][2]."


def test_validate():
    assert validate_citations("A [3].", 2)["invalid_citations"] == [3]


def test_auto_attribute():
    out = auto_attribute("BM25 ranks documents by term frequency.", CHUNKS)
    assert "[1]" in out


class Stubborn:  # never cites
    supports_system = True
    def generate(self, prompt, max_new_tokens=0, system=None):
        return "BM25 ranks documents using term frequency. Fusion merges ranked lists from dense and sparse retrieval."


class Compliant:
    supports_system = True
    def __init__(self): self.calls = 0
    def generate(self, prompt, max_new_tokens=0, system=None):
        self.calls += 1
        return "BM25 ranks documents using term frequency [1]. Fusion merges ranked lists [2]."


def test_fallback_when_model_ignores_rule():
    r = RAGGenerator(Stubborn()).answer("q", CHUNKS)
    assert r["attempts"] == 2 and r["citation_method"] == "auto_attributed"
    assert calculate_citation_coverage(r["answer"]) == 1.0


def test_no_retry_when_compliant():
    g = Compliant()
    r = RAGGenerator(g).answer("q", CHUNKS)
    assert g.calls == 1 and r["citation_method"] == "model"


def test_relevance_not_pinned_to_one():
    s = ConfidenceScorer()
    weak = [{"reranker_score": -6.0}, {"reranker_score": -7.0}]
    strong = [{"reranker_score": 5.0}, {"reranker_score": 4.0}]
    assert s.calculate_retrieval_relevance(weak) < 0.2 < s.calculate_retrieval_relevance(strong)


class BareCitation:
    supports_system = False
    def generate(self, prompt, max_new_tokens=0, system=None):
        return "[1]"


def test_bare_citation_is_not_an_answer():
    from src.generation.citations import is_substantive
    assert not is_substantive("[1]")
    r = RAGGenerator(BareCitation()).answer("q", CHUNKS)
    assert r.get("generation_failed") and r["citations"] == []
    assert calculate_citation_coverage("[1]") == 0.0
