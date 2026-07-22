"""FastAPI service: frontend + upload + ask, with bring-your-own-key.

Run:   uv run uvicorn graphrag_assistant.api:app --reload
Open:  http://localhost:8000/
"""

import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings
from .db import get_driver
from .generate import Generator
from .pipeline import process_document
from .retrieve import GraphRetriever, VectorRetriever

STATIC = Path(__file__).parent / "static"
UPLOADS = Path("/tmp/graphrag_uploads")
UPLOADS.mkdir(parents=True, exist_ok=True)

state: dict = {}
jobs: dict[str, dict] = {}  # job_id -> progress dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["driver"] = get_driver()   # one shared driver for all requests
    yield
    state["driver"].close()


app = FastAPI(title="GraphRAG Research Assistant", version="0.2.0", lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    try:
        with state["driver"].session(database=settings.neo4j_database) as s:
            s.run("RETURN 1").consume()
        return {"status": "ok", "neo4j": "connected"}
    except Exception:
        return {"status": "degraded", "neo4j": "unreachable"}


@app.post("/upload")
async def upload(file: UploadFile = File(...),
                 api_key: str = Form(...),
                 model: str = Form("gpt-4.1-mini")):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    job_id = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{job_id}_{file.filename}"
    dest.write_bytes(await file.read())

    jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0,
                    "total": 0, "filename": file.filename}
    threading.Thread(
        target=process_document,
        args=(state["driver"], str(dest), api_key, model, jobs[job_id]),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    return job


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["vector", "graph"] = "graph"
    k: int = Field(5, ge=1, le=20)
    api_key: str = Field(..., min_length=1)
    model: str = "gpt-4.1-mini"


class ChunkRef(BaseModel):
    chunk_id: str
    paper_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    mode: str
    chunks: list[ChunkRef]
    graph_facts: list[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    driver = state["driver"]
    retriever = (GraphRetriever(driver=driver, api_key=req.api_key)
                 if req.mode == "graph"
                 else VectorRetriever(driver=driver, api_key=req.api_key))
    try:
        result = retriever.retrieve(req.question, k=req.k)
        answer = Generator(api_key=req.api_key, model=req.model).answer(req.question, result)
    except Exception as exc:
        raise HTTPException(502, f"Model/DB error: {str(exc)[:200]}")
    finally:
        retriever.close()

    return AskResponse(
        answer=answer, mode=req.mode,
        chunks=[ChunkRef(chunk_id=c.chunk_id, paper_id=c.paper_id, score=c.score)
                for c in result.chunks],
        graph_facts=result.graph_facts,
    )
