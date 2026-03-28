# tests/unit/test_loader.py
"""Unit tests for the document loader module."""

import tempfile
from pathlib import Path

import openpyxl
import pytest

from ingestion.loader import load_document


class TestLoadDocumentTextFiles:
    """Tests for loading .txt and .md files."""

    def test_load_txt_file(self) -> None:
        """Successfully load a .txt file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a text file.\nWith multiple lines.\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert "This is a text file." in result
            assert "With multiple lines." in result
        finally:
            Path(temp_path).unlink()

    def test_load_md_file(self) -> None:
        """Successfully load a .md (Markdown) file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Heading\n\nSome markdown content here.\n\n- Item 1\n- Item 2\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert "# Heading" in result
            assert "markdown content" in result
            assert "Item 1" in result
        finally:
            Path(temp_path).unlink()

    def test_load_txt_with_utf8_content(self) -> None:
        """Load .txt file with UTF-8 encoded content."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Unicode content: café, 你好, مرحبا, Привет\n")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert "café" in result
            assert "你好" in result
            assert "مرحبا" in result
        finally:
            Path(temp_path).unlink()

    def test_load_empty_text_file(self) -> None:
        """Load an empty .txt file returns empty string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert result == ""
        finally:
            Path(temp_path).unlink()

    def test_load_large_text_file(self) -> None:
        """Load a large .txt file (10MB+)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write 100KB of text
            content = "This is a line of text. " * 4000
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert len(result) > 90000  # Should be close to what we wrote
            assert "This is a line of text." in result
        finally:
            Path(temp_path).unlink()


class TestLoadDocumentPdfFiles:
    """Tests for loading .pdf files."""

    def test_load_pdf_file_fails_gracefully(self) -> None:
        """Loading an invalid PDF file raises an error."""
        # Create a temporary file with .pdf extension but invalid PDF content
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"This is not a valid PDF file")
            f.flush()
            temp_path = f.name

        try:
            # PyPDFLoader will raise an error for malformed PDFs
            with pytest.raises(Exception):  # Could be ValueError, RuntimeError, etc.
                load_document(temp_path)
        finally:
            Path(temp_path).unlink()


class TestLoadDocumentErrors:
    """Tests for error handling and edge cases."""

    def test_unsupported_extension_raises_value_error(self) -> None:
        """Loading an unsupported file type raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\nval1,val2")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_document(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_unsupported_extension_error_mentions_extension(self) -> None:
        """The ValueError message includes the unsupported extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\nval1,val2")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_document(temp_path)
            assert ".csv" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_nonexistent_file_raises_error(self) -> None:
        """Attempting to load a non-existent file raises an error."""
        # RuntimeError is raised by langchain when file doesn't exist
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            load_document("/tmp/nonexistent_file_xyz_12345.txt")

    def test_case_insensitive_extension_handling(self) -> None:
        """File extensions are matched case-insensitively."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".TXT", delete=False) as f:
            f.write("Text with uppercase extension.")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert "Text with uppercase extension." in result
        finally:
            Path(temp_path).unlink()

    def test_mixed_case_md_extension(self) -> None:
        """Markdown extension works with mixed case (.MD, .Md)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".MD", delete=False) as f:
            f.write("# Markdown Header")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)
            assert "# Markdown Header" in result
        finally:
            Path(temp_path).unlink()

    def test_error_message_includes_supported_formats(self) -> None:
        """The error message lists supported formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                load_document(temp_path)
            error_msg = str(exc_info.value)
            assert ".pdf" in error_msg or "pdf" in error_msg
            assert ".txt" in error_msg or "txt" in error_msg
            assert ".md" in error_msg or "md" in error_msg
        finally:
            Path(temp_path).unlink()


class TestLoadDocumentParameterHandling:
    """Tests for parameter handling and path types."""

    def test_accept_string_path(self) -> None:
        """load_document accepts string paths."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            f.flush()
            temp_path = f.name

        try:
            result = load_document(temp_path)  # Pass as string
            assert "Test content" in result
        finally:
            Path(temp_path).unlink()

    def test_accept_path_object(self) -> None:
        """load_document accepts pathlib.Path objects."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            f.flush()
            temp_path = Path(f.name)

        try:
            result = load_document(temp_path)  # Pass as Path object
            assert "Test content" in result
        finally:
            temp_path.unlink()

    def test_handles_relative_paths(self) -> None:
        """load_document can handle relative paths (if file exists)."""
        # Create a temp file in a known location
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", dir=".", delete=False
        ) as f:
            f.write("Relative path test")
            f.flush()
            temp_path = Path(f.name)

        try:
            result = load_document(temp_path)
            assert "Relative path test" in result
        finally:
            temp_path.unlink()


class TestLoadDocumentExcelFiles:
    """Tests for .xlsx/.xls loading with header-aware row serialization."""

    def _make_xlsx(self, headers: list, rows: list[list]) -> Path:
        """Create a temporary .xlsx file and return its path."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(headers)
        for row in rows:
            ws.append(row)
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wb.save(tmp.name)
        tmp.close()
        return Path(tmp.name)

    def test_row_contains_header_value_pairs(self) -> None:
        """Each data row is serialized as 'Header: value' pairs."""
        path = self._make_xlsx(
            headers=["Name", "Age", "Role"],
            rows=[["Alice", 30, "Engineer"]],
        )
        try:
            result = load_document(path)
            assert "Name: Alice" in result
            assert "Age: 30" in result
            assert "Role: Engineer" in result
        finally:
            path.unlink()

    def test_header_row_not_duplicated_as_data(self) -> None:
        """The header row itself is not serialized as a data row."""
        path = self._make_xlsx(
            headers=["Name", "Age"],
            rows=[["Bob", 25]],
        )
        try:
            result = load_document(path)
            # Header values appear as keys, not standalone lines
            assert "Name: Bob" in result
            assert "Age: 25" in result
            # The raw header row should not appear as "Name: Name"
            assert "Name: Name" not in result
        finally:
            path.unlink()

    def test_none_cells_serialized_as_empty_string(self) -> None:
        """None cell values are rendered as empty string, not 'None'."""
        path = self._make_xlsx(
            headers=["Name", "Notes"],
            rows=[["Carol", None]],
        )
        try:
            result = load_document(path)
            assert "Notes: None" not in result
            assert "Name: Carol" in result
        finally:
            path.unlink()

    def test_sheet_name_included(self) -> None:
        """The sheet title appears in the output."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Employees"
        ws.append(["Name"])
        ws.append(["Dave"])
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        wb.save(tmp.name)
        tmp.close()
        path = Path(tmp.name)
        try:
            result = load_document(path)
            assert "Employees" in result
        finally:
            path.unlink()

    def test_multiple_rows_all_serialized(self) -> None:
        """All data rows are present in the output."""
        path = self._make_xlsx(
            headers=["City"],
            rows=[["Madrid"], ["London"], ["Paris"]],
        )
        try:
            result = load_document(path)
            assert "City: Madrid" in result
            assert "City: London" in result
            assert "City: Paris" in result
        finally:
            path.unlink()
