from typing import List

from .models import Document, Chunk

# ......................................Fixed Chunk...........................................
def fixed_size_chunk(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = document.text

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:

            metadata = document.metadata.copy()

            metadata.update({
                "chunk_index": chunk_index,
                "chunking_strategy": "fixed",
                "char_count": len(chunk_text),
                "start_char": start,
                "end_char": min(end, len(text))
            })

            chunk = Chunk(
                text=chunk_text,
                metadata=metadata
            )

            chunks.append(chunk)

            chunk_index += 1

        start += chunk_size - chunk_overlap

    return chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ................................Recursive Chunk.....................................
def recursive_chunk(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    split_texts = splitter.split_text(document.text)

    chunks = []

    for chunk_index, chunk_text in enumerate(split_texts):

        metadata = document.metadata.copy()

        metadata.update({
            "chunk_index": chunk_index,
            "chunking_strategy": "recursive",
            "char_count": len(chunk_text)
        })

        chunks.append(
            Chunk(
                text=chunk_text,
                metadata=metadata
            )
        )

    return chunks
# .........................................unified chunking...............................................

def chunk_document(
    document: Document,
    strategy: str = "fixed",
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Chunk]:

    if strategy == "fixed":

        return fixed_size_chunk(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    elif strategy == "recursive":

        return recursive_chunk(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )
    