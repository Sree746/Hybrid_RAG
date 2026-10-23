from typing import List
from .citations import split_sentences


def split_into_claims(answer: str) -> List[str]:
    """Baseline claim extractor: one claim per sentence (citations stay attached)."""
    return split_sentences(answer) if answer.strip() else []
