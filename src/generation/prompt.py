from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a technical assistant that answers strictly from numbered evidence blocks.

CITATION RULES (mandatory):
- Every sentence that states a fact MUST end with one or more evidence IDs in square brackets, e.g. "BM25 ranks documents by term frequency [2]." or "... [1][3]."
- Use only the numbers of the evidence blocks provided. Never invent a number.
- Do not put citations in a separate list at the end; put them inline, after each claim.
- Use only information found in the evidence. If the evidence does not answer the question, reply exactly: "I don't have enough information in the provided documents to answer this question."
- If only part of the question is answered, answer that part (with citations) and say what is missing.
- For questions asking "how many", state the number explicitly.
- Enumerate every item when the evidence provides a list.
- Explain each item in a separate sentence or bullet.
- Never end the answer with an incomplete phrase.
- If the retrieved evidence contains only the beginning of a list,
  use additional evidence before answering."""


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        md = chunk.get("metadata", {}) or {}
        header = f"Source: {md.get('source', 'unknown')}"
        if md.get("page") is not None:
            header += f", Page: {md['page']}"
        if md.get("section") is not None:
            header += f", Section: {md['section']}"
        parts.append(f"[{index}] ({header})\n{chunk['text']}")
    return "\n\n".join(parts)


def build_user_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], strict: bool = False) -> str:
    """Evidence first, then question, then the citation reminder LAST (closest to generation)."""
    n = len(retrieved_chunks)
    reminder = (
        f"Answer using only the evidence above. End EVERY factual sentence with an evidence ID "
        f"from [1] to [{n}]."
    )
    if strict:
        reminder += (
            " Your previous answer had missing citations. Rewrite it so that every sentence "
            "ends with at least one [number], for example: \"The method splits text into chunks [1].\""
        )
    return f"EVIDENCE:\n{format_context(retrieved_chunks)}\n\nQUESTION: {query}\n\n{reminder}\n\nANSWER:"


def build_grounded_prompt(query: str, retrieved_chunks: List[Dict[str, Any]], strict: bool = False) -> str:
    """Single-string prompt (for models without a system role, e.g. flan-t5)."""
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(query, retrieved_chunks, strict)}"
