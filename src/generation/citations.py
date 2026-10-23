import re
from typing import List, Dict, Any

# Matches [1], [1, 2], [1,2], [1][2], [Source 1], [Evidence 2], [1-3]
_BRACKET = re.compile(r"\[\s*(?:source|evidence|doc|document|ref)?\s*([\d,\s\-–and]+?)\s*\]", re.I)


def _numbers_in(group: str) -> List[int]:
    nums: List[int] = []
    for part in re.split(r"[,\s]+|and", group):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < hi - lo <= 10:
                nums.extend(range(lo, hi + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


def find_citations(text: str) -> List[int]:
    """All citation numbers in text, in order (duplicates kept)."""
    out: List[int] = []
    for m in _BRACKET.finditer(text):
        out.extend(_numbers_in(m.group(1)))
    return out


def extract_citation_numbers(answer: str) -> List[int]:
    """Unique citation numbers, in order of first appearance."""
    seen: List[int] = []
    for n in find_citations(answer):
        if n not in seen:
            seen.append(n)
    return seen


def normalize_citations(answer: str) -> str:
    """Rewrite [Source 1], [1, 2], [1-3] into canonical [1][2][3] form."""
    def repl(m):
        nums = _numbers_in(m.group(1))
        return "".join(f"[{n}]" for n in nums) if nums else m.group(0)
    return _BRACKET.sub(repl, answer)


def validate_citations(answer: str, number_of_sources: int) -> Dict[str, Any]:
    citations = extract_citation_numbers(answer)
    invalid = [c for c in citations if c < 1 or c > number_of_sources]
    return {"citations": citations, "invalid_citations": invalid, "all_valid": not invalid}


def split_sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(\d])", text.strip()) if s.strip()]


def calculate_citation_coverage(answer: str) -> float:
    """Fraction of answer sentences carrying a citation (presence, not correctness)."""
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if find_citations(s)) / len(sentences)


# --------------------------------------------------------------------------
# Fallback: attach citations ourselves when the model ignored the instruction
# --------------------------------------------------------------------------
_STOP = set("the a an of to in and or is are was were for on with as by that this it its be from at which".split())


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2}


def auto_attribute(answer: str, chunks: List[Dict[str, Any]], min_overlap: float = 0.25) -> str:
    """
    Append [n] to each uncited sentence, choosing the chunk with the highest
    lexical overlap. Sentences with no good match are left uncited (and will be
    flagged by the verifier) rather than given a made-up source.
    """
    chunk_tokens = [_tokens(c["text"]) for c in chunks]
    out = []
    for sentence in split_sentences(answer):
        if find_citations(sentence):
            out.append(sentence)
            continue
        st = _tokens(sentence)
        if not st:
            out.append(sentence)
            continue
        scores = [len(st & ct) / len(st) for ct in chunk_tokens]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        if best is not None and scores[best] >= min_overlap:
            sentence = re.sub(r"([.!?])?$", rf" [{best + 1}]\1", sentence, count=1)
        out.append(sentence)
    return " ".join(out)


def strip_citations(text: str) -> str:
    """Remove all [n]-style markers, leaving only the prose."""
    return re.sub(r"\s+", " ", _BRACKET.sub("", text)).strip()


def is_substantive(text: str, min_words: int = 8) -> bool:
    """True if the answer has real prose beyond citation markers (a bare "[1]" is not an answer)."""
    return len(re.findall(r"[A-Za-z0-9]+", strip_citations(text))) >= min_words


def calculate_citation_coverage(answer: str) -> float:  # redefined: ignore empty "sentences" like "[1]"
    sentences = [s for s in split_sentences(answer) if is_substantive(s, min_words=3)]
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if find_citations(s)) / len(sentences)
