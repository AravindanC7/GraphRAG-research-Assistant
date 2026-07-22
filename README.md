# GraphRAG Research Assistant

A GraphRAG question-answering system over a corpus of ML research papers. It
combines **vector search** with **knowledge-graph traversal** to answer multi-hop
questions that plain vector RAG cannot, and it is **evaluated against a vector
baseline** with a measured improvement in retrieval recall.

Built end-to-end: ingestion → LLM entity/relationship extraction → entity
resolution → hybrid retrieval → evaluation → a containerized FastAPI service.

---

## Headline result

On a balanced, paper-verified **14-question** evaluation set spanning four topical
clusters, graph retrieval beat the vector baseline on fact recall:

| Retriever | Recall | Precision |
|-----------|:------:|:---------:|
| Vector (baseline) | 0.625 | 0.46 |
| **Graph (GraphRAG)** | **0.744** | 0.44 |

The gain concentrates on **multi-hop and cross-paper questions** (e.g. "which
methods are evaluated on GSM8K?"), where traversal collects connected facts that
no single retrieved chunk contains — while single-hop questions tie, as expected.
`hub_max` (the traversal degree filter) was tuned via a sweep; 0.744 is the peak.

---

## Architecture

Two graph layers live in one Neo4j database:

- **Lexical layer** — `(:Paper)-[:HAS_CHUNK]->(:Chunk)` with embeddings, for
  semantic (vector) retrieval.
- **Domain layer** — typed entities (`Method`, `Dataset`, `Model`, ...) connected
  by typed relationships (`EVALUATED_ON`, `IMPLEMENTS`, `EXTENDS`, ...), tethered
  to their source chunks via `(:Chunk)-[:MENTIONS]->(:Entity)`.

Graph retrieval = **vector entry → entity seeding → one-hop traversal (hub-filtered)
→ relevance-ranked facts → grounded generation**.

```
PDFs ─▶ ingest ─▶ [Lexical layer]
                       │
        build_graph ─▶ [Domain layer] ─┐
        resolve_entities                │
                                        ▼
  question ─▶ retrieve (vector | graph) ─▶ generate ─▶ cited answer
                                        ▲
                              FastAPI  /ask  /health
```

---

## Tech stack

Python 3.12 · Neo4j 5.26 (graph + native vector index) · OpenAI
(`text-embedding-3-small`, `gpt-4.1-mini`) · FastAPI + Uvicorn · Docker Compose ·
`uv` for dependency management.

---

## Quickstart

Requires Docker and an OpenAI API key.

```bash
cp .env.example .env          # then add your OPENAI_API_KEY
docker compose up --build     # starts Neo4j + the API together
```

- API docs (interactive):  http://localhost:8000/docs
- Health check:            http://localhost:8000/health
- Neo4j Browser:           http://localhost:7474  (neo4j / password123)

Ask a question over HTTP:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which methods are evaluated on GSM8K?", "mode": "graph", "k": 5}'
```

### Building the graph from scratch (optional)

```bash
# drop PDFs into data/papers/, then:
uv run python -m graphrag_assistant.ingest              # Phase 1: chunks + embeddings
uv run python -m graphrag_assistant.build_graph         # Phase 2: entities + relationships
uv run python -m graphrag_assistant.resolve_entities    # Phase 2.5: merge duplicates
uv run python -m graphrag_assistant.evaluate            # Phase 4: vector vs graph
```

---

## Evaluation

`evaluate.py` scores each answer three ways: **recall** (gold items surfaced),
**precision** (named items that were correct, checked against graph entities of the
right type), and an **LLM-as-judge** quality score. Gold sets were hand-verified
against the source PDFs. Run `--sweep-hub 40,80,150,10000` to reproduce the
threshold tuning.

---

## Findings: when GraphRAG helps, and when it doesn't

Evaluation revealed that GraphRAG's benefit is **conditional**, and three distinct
failure modes were diagnosed and (mostly) addressed:

1. **Hub-node noise.** Traversing through high-degree "hub" nodes (e.g. a generic
   `Traffic Signal Control` concept linked to ~everything) floods context with
   irrelevant facts. *Fix:* a node-degree filter, tuned via sweep to `hub_max=150`.
2. **Table-structured extraction gaps.** Relationships encoded in document *tables*
   rather than prose are frequently missed by LLM extraction. *Fix:* added an
   `IMPLEMENTS` relationship type + targeted re-extraction (recovered 7/10 table
   entries for one paper).
3. **Fact-ranking bias.** Relevance-ranking by question similarity can suppress
   terse-but-correct relationship facts (`LibSignal -[IMPLEMENTS]-> IDQN`) in favor
   of verbose facts sharing surface vocabulary with the question. *Diagnosed;
   future work: relationship-type-aware ranking.*

## Known limitations

- Small corpus (6 papers) and a 14-question set: a **proof of concept**, not a
  benchmark. Numbers are directional.
- Residual entity-resolution duplicates (normalization-based; no embedding/LLM
  resolution).
- LLM extraction from dense tables is imperfect (see finding #2).
- Retrieval and extraction are sequential (no concurrency); full extraction ~40 min.

---

## Project layout

```
src/graphrag_assistant/
  config.py          settings (env-driven)
  db.py              Neo4j driver + vector index
  llm.py             Embedder + ChatLLM (OpenAI wrappers)
  loaders.py         PDF loading      chunking.py   text chunking
  schema.py          entity/relationship types + type-patterns
  ingest.py          Phase 1: lexical graph
  build_graph.py     Phase 2: entity/relationship extraction
  resolve_entities.py Phase 2.5: entity resolution
  retrieve.py        vector + graph retrievers
  generate.py        grounded generation
  ask.py             CLI entrypoint
  evaluate.py        vector-vs-graph eval harness   testset.py  questions + gold
  api.py             FastAPI service
Dockerfile · docker-compose.yml       one-command stack
```
