from graphrag_assistant.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", 100, 10) == []


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)


def test_chunks_cover_text_with_overlap():
    text = "x" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert chunks[0] == "x" * 100
    # step = size - overlap = 80, so starts at 0, 80, 160 (160->250 ends it)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)
