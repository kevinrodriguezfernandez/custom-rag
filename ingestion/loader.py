# ingestion/loader.py
"""Document loading — reads raw files and returns plain text."""

from pathlib import Path

from pypdf import PdfReader


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
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .txt, .pdf")
