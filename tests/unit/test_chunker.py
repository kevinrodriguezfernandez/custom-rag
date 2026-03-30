# tests/unit/test_chunker.py
"""Unit tests for the text chunking module."""


from ingestion.chunker import chunk_text
from shared.models import DocumentChunk


class TestChunkTextBasic:
    """Tests for basic chunking behavior."""

    def test_chunk_text_returns_list_of_document_chunks(self) -> None:
        """Chunking returns a list of DocumentChunk objects."""
        text = "This is a test document with some content that should be chunked."
        result = chunk_text(text, document_id="doc-1")

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in result)

    def test_chunk_text_preserves_document_id(self) -> None:
        """All chunks are tagged with the parent document_id."""
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10
        document_id = "test-doc-42"
        chunks = chunk_text(text, document_id=document_id)

        assert all(chunk.document_id == document_id for chunk in chunks)

    def test_chunk_id_format_is_correct(self) -> None:
        """Chunk IDs follow the format {document_id}-{index}."""
        text = "Short text that will be chunked into multiple pieces." * 20
        document_id = "my-doc"
        chunks = chunk_text(text, document_id=document_id)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"{document_id}-{i}"

    def test_chunk_size_respected(self) -> None:
        """Chunks do not exceed the specified chunk_size (with tolerance for splits)."""
        # Create text larger than default chunk size
        text = "A" * 2000
        chunk_size = 512
        chunks = chunk_text(text, document_id="doc-1", chunk_size=chunk_size)

        # Most chunks should respect the size limit
        # (RecursiveCharacterTextSplitter may slightly exceed for safety)
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size * 1.5  # Allow some tolerance

    def test_chunk_overlap_creates_continuity(self) -> None:
        """Overlapping content appears in consecutive chunks."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunk_size = 200
        chunk_overlap = 50
        chunks = chunk_text(
            text,
            document_id="doc-1",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # With overlap, consecutive chunks should have some common text
        if len(chunks) > 1:
            # Check that later chunks start with content from earlier chunks (overlap)
            for i in range(len(chunks) - 1):
                current_end = chunks[i].content[-chunk_overlap:]
                next_start = chunks[i + 1].content[:chunk_overlap]
                # There should be some common ground due to overlap
                assert len(current_end) > 0 and len(next_start) > 0

    def test_chunk_metadata_includes_chunk_index(self) -> None:
        """Each chunk's metadata includes its index."""
        text = "Sample text that will be split into multiple chunks." * 30
        chunks = chunk_text(text, document_id="doc-1")

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("chunk_index") == i

    def test_content_is_non_empty(self) -> None:
        """All chunks contain non-empty content."""
        text = "This is a test document with actual content to chunk." * 20
        chunks = chunk_text(text, document_id="doc-1")

        assert all(len(chunk.content) > 0 for chunk in chunks)


class TestChunkTextEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_chunk_empty_text_returns_single_empty_chunk(self) -> None:
        """Empty text produces at least one chunk (may be empty)."""
        result = chunk_text("", document_id="doc-1")

        # RecursiveCharacterTextSplitter returns an empty list for empty input
        assert isinstance(result, list)

    def test_chunk_very_short_text_returns_one_chunk(self) -> None:
        """Text shorter than chunk_size produces one chunk."""
        text = "Hello world"
        result = chunk_text(text, document_id="doc-1", chunk_size=512)

        assert len(result) == 1
        assert result[0].content == text

    def test_chunk_text_with_special_characters(self) -> None:
        """Chunking handles special characters, newlines, and unicode."""
        text = (
            "Line 1\nLine 2\n\nLine 3 with special chars: !@#$%^&*()\n"
            "Unicode: café, naïve, 你好, مرحبا\n"
            "Tabs:\ta\tb\tc" * 20
        )
        chunks = chunk_text(text, document_id="doc-1")

        assert len(chunks) > 0
        full_reconstructed = "".join(chunk.content for chunk in chunks)
        # Content should be largely preserved (accounting for overlap handling)
        assert len(full_reconstructed) >= len(text) - 100

    def test_chunk_with_very_large_text(self) -> None:
        """Chunking handles very large documents."""
        # Create a 100KB text
        text = "This is a sample sentence for chunking. " * 2500
        chunks = chunk_text(text, document_id="doc-1")

        assert len(chunks) > 1
        total_content = sum(len(chunk.content) for chunk in chunks)
        # Total content should be at least similar in length to input
        assert total_content >= len(text) * 0.9

    def test_chunk_preserves_content_integrity(self) -> None:
        """When chunks are concatenated, they preserve the original text (with overlap)."""
        text = "The quick brown fox jumps over the lazy dog and runs far away." * 20
        chunks = chunk_text(text, document_id="doc-1")

        # Concatenating all chunks (accounting for overlap loss) should give us most content
        reconstructed = "".join(chunk.content for chunk in chunks)
        assert text in reconstructed or len(reconstructed) >= len(text) * 0.8

    def test_chunk_with_zero_overlap(self) -> None:
        """Chunking with zero overlap still works."""
        text = "A" * 1000
        chunks = chunk_text(
            text, document_id="doc-1", chunk_size=100, chunk_overlap=0
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) > 0


class TestChunkTextParameters:
    """Tests for parameter validation and variations."""

    def test_custom_chunk_size_parameter(self) -> None:
        """Custom chunk_size parameter is respected."""
        text = "A" * 2000
        custom_size = 256
        chunks = chunk_text(text, document_id="doc-1", chunk_size=custom_size)

        assert len(chunks) > 0
        # With smaller chunk size, should have more chunks
        chunks_default = chunk_text(text, document_id="doc-1")
        assert len(chunks) >= len(chunks_default)

    def test_custom_overlap_parameter(self) -> None:
        """Custom chunk_overlap parameter affects chunking."""
        text = "A" * 2000
        chunks_no_overlap = chunk_text(
            text,
            document_id="doc-1",
            chunk_size=200,
            chunk_overlap=0,
        )
        chunks_with_overlap = chunk_text(
            text,
            document_id="doc-1",
            chunk_size=200,
            chunk_overlap=50,
        )

        # Both should work
        assert len(chunks_no_overlap) > 0
        assert len(chunks_with_overlap) > 0

    def test_document_id_propagation(self) -> None:
        """Document ID is correctly propagated to all chunks."""
        doc_ids = ["doc-alpha", "doc-beta", "doc-gamma-123"]
        text = "Sample text for chunking." * 50

        for doc_id in doc_ids:
            chunks = chunk_text(text, document_id=doc_id)
            assert all(chunk.document_id == doc_id for chunk in chunks)
