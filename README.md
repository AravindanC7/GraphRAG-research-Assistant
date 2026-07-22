# GraphRAG Research Assistant

> Ask multi-hop questions over a corpus of ML research papers, answered by
> **knowledge-graph traversal + vector search** — with a measured improvement
> over a vector-only baseline.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.26-018bff?logo=neo4j&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A full, end-to-end system: PDF ingestion → LLM entity/relationship extraction →
entity resolution → hybrid retrieval → **evaluation against a baseline** → a
containerized FastAPI service with a bring-your-own-key web UI.

<!-- TODO: add a screenshot or demo GIF here — it makes the repo pop.
     Record the UI at http://localhost:8000/, save as docs/demo.gif, then:
     ![demo](docs/demo.gif)
-->

---

## Highlights

- **GraphRAG that's actually evaluated.** On a balanced, paper-verified 14-question
  set, graph retrieval improved fact recall to **0.744 vs 0.625** for the vector
  baseline — with gains concentrated on multi-hop and cross-paper questions.
- **Diagnosis-driven engineering.** Evaluation surfaced three concrete failure
  modes; two were fixed and re-measured (see [Findings](#findings)).
- **Bring-your-own-key web app.** Upload a PDF, watch the graph build in real time,
  then chat over it — vector or graph mode, with the traversed facts shown.
- **One-command deploy.** `docker compose up` starts the API and Neo4j together.

---

## Result

| Retriever | Recall | Precision |
|-----------|:------:|:---------:|
| Vector (baseline) | 0.625 | 0.46 |
| **Graph (GraphRAG)** | **0.744** | 0.44 |

`hub_max` (the traversal degree filter) was tuned via a parameter sweep;
0.744 is the peak. Single-hop questions tie; graph wins where traversal matters.

---

## Quickstart

Requires Docker and an OpenAI API key.

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up --build     # starts Neo4j + the API together
```

Then open:

| URL | What |
|-----|------|
| http://localhost:8000/       | **Web app** — upload PDFs + chat |
| http://localhost:8000/docs   | Interactive API docs (OpenAPI)  |
| http://localhost:8000/health | Health + Neo4j connectivity     |
| http://localhost:7474        | Neo4j Browser (neo4j / password123) |

Ask over HTTP:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which methods are evaluated on GSM8K?","mode":"graph","k":5,"api_key":"sk-..."}'
```

---

## How it works

Two graph layers share one Neo4j database:

- **Lexical layer** — `(:Paper)-[:HAS_CHUNK]->(:Chunk)` with embeddings, for vector search.
- **Domain layer** — typed entities (`Method`, `Dataset`, `Model`, …) joined by typed
  relationships (`EVALUATED_ON`, `IMPLEMENTS`, `EXTENDS`, …), linked back to source
  chunks via `(:Chunk)-[:MENTIONS]->(:Entity)`.

Graph retrieval pipeline:

```
question ─▶ vector entry ─▶ entity seeding ─▶ one-hop traversal
             (top-k chunks)   (chunk + name)     (hub-filtered)
                                                       │
        cited answer ◀─ grounded generation ◀─ relevance-ranked facts
```

The web app adds an upload path that runs the full pipeline per document in a
background thread, with live status polling.

---

## Findings

Evaluation showed GraphRAG's benefit is **conditional**. Three failure modes were
diagnosed:

1. **Hub-node noise** — traversal through high-degree nodes floods context with
   irrelevant facts. *Fixed:* node-degree filter, tuned via sweep to `hub_max=150`.
2. **Table-structured extraction gaps** — relationships encoded in document tables
   (not prose) are missed by LLM extraction. *Fixed:* added an `IMPLEMENTS`
   relationship type + targeted re-extraction (recovered 7/10 table entries).
3. **Fact-ranking bias** — relevance-ranking can suppress terse-but-correct
   relationship facts in favor of verbose facts sharing question vocabulary.
   *Diagnosed; future work: relationship-type-aware ranking.*

---

## Tech stack

Python 3.12 · Neo4j 5.26 (graph + native vector index) · OpenAI
(`text-embedding-3-small`, GPT-4.1 family) · FastAPI + Uvicorn · vanilla-JS
frontend · Docker Compose · `uv`.

---

## Rebuild the graph from scratch (optional)

```bash
# drop PDFs into data/papers/, then:
uv run python -m graphrag_assistant.ingest              # chunks + embeddings
uv run python -m graphrag_assistant.build_graph         # entities + relationships
uv run python -m graphrag_assistant.resolve_entities    # merge duplicates
uv run python -m graphrag_assistant.evaluate            # vector vs graph
uv run python -m graphrag_assistant.evaluate --sweep-hub 40,80,150,10000  # tune
```

---

## Limitations

Small corpus (6 papers) + 14-question set: a **proof of concept**, not a benchmark —
numbers are directional. LLM extraction from dense tables is imperfect (finding #2).
Per-upload processing is synchronous per document; a production system would use a
job queue (e.g. Celery). Retrieval and extraction are sequential.

---

## Project layout

```
src/graphrag_assistant/
  config.py db.py llm.py           settings · Neo4j · OpenAI wrappers
  loaders.py chunking.py schema.py loading · chunking · type system
  ingest.py                        lexical graph (chunks + embeddings)
  build_graph.py resolve_entities.py  extraction · entity resolution
  retrieve.py generate.py          vector + graph retrieval · generation
  evaluate.py testset.py           eval harness · paper-verified gold set
  pipeline.py api.py               per-upload pipeline · FastAPI service
  static/index.html                web frontend
Dockerfile · docker-compose.yml    one-command stack
```

---

## Author

**ARAVINDAN CHIDAMBARAM** · aravindan.2699@gmail.com

Built as an end-to-end applied-ML project: retrieval, knowledge graphs,
evaluation, and deployment.
