from pathlib import Path
from typing import List
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import Document


def load_txt(file_path: str) -> Document:

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8"
    )

    metadata = {
        "source": path.name,
        "file_type": "txt",
        "path": str(path)
    }

    return Document(
        text=text,
        metadata=metadata
    )


def load_markdown(file_path: str) -> Document:

    path = Path(file_path)

    text = path.read_text(
        encoding="utf-8"
    )

    metadata = {
        "source": path.name,
        "file_type": "markdown",
        "path": str(path)
    }

    return Document(
        text=text,
        metadata=metadata
    )


def load_html(file_path: str) -> Document:

    path = Path(file_path)

    html = path.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = soup.get_text(
        separator="\n",
        strip=True
    )

    metadata = {
        "source": path.name,
        "file_type": "html",
        "path": str(path)
    }

    return Document(
        text=text,
        metadata=metadata
    )


def load_pdf(file_path: str) -> Document:

    path = Path(file_path)

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        pages.append(text)

    full_text = "\n\n".join(pages)

    metadata = {
        "source": path.name,
        "file_type": "pdf",
        "path": str(path),
        "page_count": len(reader.pages)
    }

    return Document(
        text=full_text,
        metadata=metadata
    )


def load_document(file_path: str) -> Document:

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    elif extension in [".md", ".markdown"]:
        return load_markdown(file_path)

    elif extension in [".html", ".htm"]:
        return load_html(file_path)

    elif extension == ".pdf":
        return load_pdf(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )


def load_documents(
    directory: str
) -> List[Document]:

    directory_path = Path(directory)

    documents = []

    supported_extensions = {
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".pdf"
    }

    for file_path in directory_path.iterdir():

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:

            document = load_document(
                str(file_path)
            )

            documents.append(document)

            print(
                f"Loaded: {file_path.name}"
            )

        except Exception as e:

            print(
                f"Failed to load {file_path.name}: {e}"
            )

    return documents