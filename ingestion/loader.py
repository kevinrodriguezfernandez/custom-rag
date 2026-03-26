# ingestion/loader.py
"""Document loading — reads raw files and returns plain text."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)


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
    logger.info("Loading document — path=%s extension=%s", file_path, suffix)

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
    elif suffix == ".docx":
        import docx2txt
        text = docx2txt.process(str(file_path))
    elif suffix in {".xlsx", ".xls"}:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        parts = []
        for sheet in wb.worksheets:
            parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                parts.append("\t".join("" if c is None else str(c) for c in row))
        wb.close()
        text = "\n".join(parts)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. Supported: .pdf, .txt, .md, .docx, .xlsx, .xls"
        )

    logger.info("Document loaded — text_length=%d", len(text))
    return text
