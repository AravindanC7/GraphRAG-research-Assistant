"""Phase 5: FastAPI service exposing the GraphRAG assistant.

Run locally:
    uv run uvicorn graphrag_assistant.api:app --reload

Then open http://localhost:8000/docs for interactive, browser-testable API docs.

The retrievers and generator hold a Neo4j connection and an OpenAI client, so
they are built ONCE at startup (via the lifespan handler) and reused for every
request — not rebuilt per call.
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .generate import Generator
from .retrieve import GraphRetriever, VectorRetriever

# --- request / response schemas (FastAPI validates against these) ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="the question to answer")
    mode: Literal["vector", "graph"] = Field("graph", description="retrieval strategy")
    k: int = Field(5, ge=1, le=20, description="number of chunks to retrieve")


class ChunkRef(BaseModel):
    chunk_id: str
    paper_id: str
    score: float


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    chunks: list[ChunkRef]
    graph_facts: list[str]


# --- shared state, built once at startup -----------------------------------

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: build the heavy, reusable objects once
    state["vector"] = VectorRetriever()
    state["graph"] = GraphRetriever()
    state["generator"] = Generator()
    yield
    # shutdown: close the database connections cleanly
    state["vector"].close()
    state["graph"].close()


app = FastAPI(
    title="GraphRAG Research Assistant",
    version="0.1.0",
    description="Ask questions over a research-paper corpus via vector or graph retrieval.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness + Neo4j connectivity check."""
    try:
        with state["graph"].driver.session(database=settings.neo4j_database) as s:
            s.run("RETURN 1").consume()
        return {"status": "ok", "neo4j": "connected"}
    except Exception:
        return {"status": "degraded", "neo4j": "unreachable"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a question, grounded in the corpus, via the chosen retriever."""
    retriever = state[req.mode]  # "vector" or "graph" — validated by the schema
    result = retriever.retrieve(req.question, k=req.k)
    answer = state["generator"].answer(req.question, result)
    return AskResponse(
        question=req.question,
        mode=req.mode,
        answer=answer,
        chunks=[ChunkRef(chunk_id=c.chunk_id, paper_id=c.paper_id, score=c.score)
                for c in result.chunks],
        graph_facts=result.graph_facts,
    )
