"""Phase 1 ingestion pipeline.

load PDFs -> chunk -> embed -> write (:Paper)-[:HAS_CHUNK]->(:Chunk) with
a vector index on chunk embeddings. Run with:

    uv run python -m graphrag_assistant.ingest
"""

from tqdm import tqdm

from .chunking import chunk_text
from .config import settings
from .db import ensure_schema, get_driver
from .llm import Embedder
from .loaders import load_papers

WRITE_QUERY = """
MERGE (p:Paper {id: $paper_id})
SET p.title = $title, p.path = $path
WITH p
UNWIND $chunks AS chunk
MERGE (c:Chunk {id: chunk.id})
SET c.text = chunk.text, c.index = chunk.index, c.embedding = chunk.embedding
MERGE (p)-[:HAS_CHUNK]->(c)
"""

EMBED_BATCH = 64  # keep request payloads reasonable


def _embed_all(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embedder.embed(texts[i : i + EMBED_BATCH]))
    return vectors


def ingest() -> None:
    driver = get_driver()
    ensure_schema(driver)
    embedder = Embedder()

    papers = load_papers(settings.papers_dir)
    if not papers:
        print(f"No PDFs found in '{settings.papers_dir}'. Drop some papers there and re-run.")
        driver.close()
        return

    print(f"Loaded {len(papers)} paper(s) from '{settings.papers_dir}'.")
    total_chunks = 0
    with driver.session(database=settings.neo4j_database) as session:
        for paper in tqdm(papers, desc="Ingesting"):
            texts = chunk_text(paper.text, settings.chunk_size, settings.chunk_overlap)
            if not texts:
                print(f"  skipped '{paper.id}' (no extractable text)")
                continue
            embeddings = _embed_all(embedder, texts)
            chunks = [
                {"id": f"{paper.id}::{i}", "text": t, "index": i, "embedding": e}
                for i, (t, e) in enumerate(zip(texts, embeddings))
            ]
            session.run(
                WRITE_QUERY,
                paper_id=paper.id,
                title=paper.title,
                path=paper.path,
                chunks=chunks,
            )
            total_chunks += len(chunks)

    driver.close()
    print(f"Done. Wrote {total_chunks} chunks across {len(papers)} papers.")
    print("Inspect at http://localhost:7474  (try:  MATCH (p:Paper)-[:HAS_CHUNK]->(c) RETURN p,c LIMIT 25)")


if __name__ == "__main__":
    ingest()
