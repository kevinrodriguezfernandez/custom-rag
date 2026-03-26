# ingestion/loader.py
"""Document loading — reads raw files and returns plain text."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path: Path) -> str:
    """Load a document from *file_path* and return its text content.

    Supported formats: .pdf, .txt, .md. All pages/sections are concatenated
    with newlines into a single string.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the source document.

    Returns
    -------
    str
        The raw text extracted from the document.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. Supported: .pdf, .txt, .md"
        )

    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)
