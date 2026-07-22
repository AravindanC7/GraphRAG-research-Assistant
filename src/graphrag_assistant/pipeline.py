"""On-demand processing of a single uploaded document.

Runs the full pipeline for one PDF — chunk -> embed -> write lexical graph ->
extract entities/relationships -> resolve — reusing the same building blocks as
the batch CLI, but scoped to one file and driven by a user-supplied API key/model.
Progress is reported by mutating the shared `job` dict so the /status endpoint
can show it.
"""

from pathlib import Path

from .build_graph import (ENTITY_CONSTRAINT, extract_from_chunk, write_chunk)
from .chunking import chunk_text
from .config import settings
from .db import ensure_schema
from .ingest import WRITE_QUERY
from .llm import ChatLLM, Embedder
from .loaders import load_pdf
from .resolve_entities import resolve

EMBED_BATCH = 64


def _embed_all(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(embedder.embed(texts[i:i + EMBED_BATCH]))
    return out


def process_document(driver, path: str, api_key: str, chat_model: str, job: dict) -> None:
    """Full pipeline for one uploaded PDF. Updates `job` in place."""
    try:
        job.update(status="processing", stage="loading", progress=0, total=0)

        paper = load_pdf(Path(path))
        paper.__dict__["id"] = f"upload__{paper.id}"  # namespace uploads
        texts = chunk_text(paper.text, settings.chunk_size, settings.chunk_overlap)
        if not texts:
            job.update(status="error", message="No extractable text (scanned PDF?).")
            return

        ensure_schema(driver)
        embedder = Embedder(api_key=api_key)                 # fixed embedding model
        llm = ChatLLM(api_key=api_key, model=chat_model)

        # 1) embed + write the lexical layer
        job.update(stage="embedding", total=len(texts))
        embeddings = _embed_all(embedder, texts)
        chunk_rows = [
            {"id": f"{paper.id}::{i}", "text": t, "index": i, "embedding": e}
            for i, (t, e) in enumerate(zip(texts, embeddings))
        ]
        with driver.session(database=settings.neo4j_database) as session:
            session.run(WRITE_QUERY, paper_id=paper.id, title=paper.title,
                        path=paper.path, chunks=chunk_rows)
            session.run(ENTITY_CONSTRAINT)

            # 2) extract entities/relationships per chunk (the slow part)
            job.update(stage="extracting", progress=0)
            for i, row in enumerate(chunk_rows):
                entities, rels = extract_from_chunk(llm, row["text"])
                write_chunk(session, row["id"], entities, rels)
                job["progress"] = i + 1

        # 3) resolve duplicate entities across the whole graph
        job.update(stage="resolving")
        resolve(report=False)

        job.update(status="done", stage="done", title=paper.title)
    except Exception as exc:  # surface any failure to the UI
        job.update(status="error", message=str(exc)[:300])
