"""Text chunking.

A simple fixed-size character splitter with overlap is plenty for Phase 1.
Swap for a token-aware splitter later (e.g. neo4j_graphrag's FixedSizeSplitter)
without changing any callers.
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks
