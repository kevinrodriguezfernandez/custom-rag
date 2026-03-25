# ingestion/loader.py
"""Document loading — reads raw files and returns plain text."""

from pathlib import Path


def load_document(file_path: Path) -> str:
    """Load a document from *file_path* and return its text content.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the source document.

    Returns
    -------
    str
        The raw text extracted from the document.
    """
    raise NotImplementedError("Document loading is not yet implemented.")
