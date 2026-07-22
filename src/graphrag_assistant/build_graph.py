"""Phase 2: knowledge-graph construction.

Reads the (:Chunk) nodes built in Phase 1, asks an LLM to extract entities and
relationships from each chunk (constrained to the schema in schema.py), and
writes them into Neo4j as typed entity nodes connected by typed relationships,
each entity tethered back to its source chunk via (:Chunk)-[:MENTIONS]->(entity).

Usage:
    # dry run: extract from a few chunks and PRINT the triples, write nothing
    uv run python -m graphrag_assistant.build_graph --limit 3 --dry-run

    # real run over the whole corpus
    uv run python -m graphrag_assistant.build_graph
"""

import argparse
import json
import re

from tqdm import tqdm

from .config import settings
from .db import get_driver
from .llm import ChatLLM
from .schema import ENTITY_TYPES, RELATIONSHIP_PATTERNS, RELATIONSHIP_TYPES

# Sets for fast validation of whatever the LLM returns.
_ALLOWED_ENTITIES = set(ENTITY_TYPES)
_ALLOWED_RELATIONSHIPS = set(RELATIONSHIP_TYPES)


def normalize_name(name: str) -> str:
    """Light name hygiene: collapse runs of whitespace and trim.

    Catches simple PDF artifacts (e.g. 'TokUR  ' -> 'TokUR') but deliberately
    does NOT lowercase or strip punctuation, which would risk merging genuinely
    distinct entities. Full entity resolution is a later refinement.
    """
    return re.sub(r"\s+", " ", name).strip()

SYSTEM_PROMPT = f"""You are an information-extraction engine for scientific \
machine-learning papers. From the given text, extract entities and the \
relationships between them.

Allowed entity types (use ONLY these): {", ".join(ENTITY_TYPES)}

Entity guidance:
- Model = a base/foundation model (e.g. Llama-3.2-1B-Instruct, Qwen-2.5-7B). \
A Method = a proposed technique/algorithm (e.g. TokUR). Keep these distinct.
- Dataset = a SPECIFIC named benchmark (e.g. GSM8K, MATH500), NOT a generic \
category like "mathematical reasoning datasets".
- Use short canonical names. Fix title-spacing artifacts: write "TokUR", not "TO KUR".
- Do NOT extract venues/conferences (ICLR), lab names, GitHub handles, or URLs.
- Do NOT extract vague concepts (e.g. "epistemic uncertainty") as Methods unless \
they name a specific proposed technique.
- Only extract entities clearly present in THIS text.

Allowed relationships, each with its required direction (source -> target):
- AUTHORED_BY: Paper -> Author
- PROPOSES: Paper -> Method
- EXTENDS: Method -> Method (source builds on / improves target)
- USES_METHOD: Paper or Method -> Method
- USES_MODEL: Paper or Method -> Model
- IMPLEMENTS: a library/toolkit/benchmark (Method) or Paper -> a Method it implements, includes, or benchmarks (e.g. a library 'implements IDQN, CoLight')
- EVALUATED_ON: Paper or Method -> Dataset (target MUST be a Dataset)
- ADDRESSES: Paper or Method -> Task
- REPORTS_METRIC: Paper or Method -> Metric
- CITES: Paper -> Paper

Relationship guidance:
- Respect the direction exactly. A paper PROPOSES a method, never the reverse.
- Every relationship's source and target name MUST also appear in your entities list.
- If the text has no relevant entities, return empty lists.

Respond with ONLY a JSON object of exactly this shape (no markdown, no prose):
{{
  "entities": [{{"name": "string", "type": "string"}}],
  "relationships": [{{"source": "string", "type": "string", "target": "string"}}]
}}"""

# --- Cypher --------------------------------------------------------------

ENTITY_CONSTRAINT = (
    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
)

# Merge each entity by name (exact-match resolution), tag it with its specific
# type as both a label (via APOC) and a property, then link it to its chunk.
WRITE_ENTITIES = """
MATCH (c:Chunk {id: $chunk_id})
UNWIND $entities AS row
MERGE (e:Entity {name: row.name})
SET e.type = row.type
WITH c, e, row
CALL apoc.create.addLabels(e, [row.type]) YIELD node
MERGE (c)-[:MENTIONS]->(node)
"""

# Connect already-existing entities. If the LLM names an endpoint it didn't
# also list as an entity, the MATCH simply finds nothing and the row is skipped.
WRITE_RELATIONSHIPS = """
UNWIND $rels AS row
MATCH (s:Entity {name: row.source})
MATCH (t:Entity {name: row.target})
CALL apoc.merge.relationship(s, row.type, {}, {}, t) YIELD rel
RETURN count(rel) AS created
"""


def fetch_chunks(driver, limit: int | None, prefix: str | None = None) -> list[dict]:
    where = "WHERE c.id STARTS WITH $prefix " if prefix else ""
    q = f"MATCH (c:Chunk) {where}RETURN c.id AS id, c.text AS text ORDER BY c.id"
    if limit:
        q += f" LIMIT {int(limit)}"
    with driver.session(database=settings.neo4j_database) as session:
        return [dict(r) for r in session.run(q, prefix=prefix)]


def extract_from_chunk(llm: ChatLLM, text: str) -> tuple[list[dict], list[dict]]:
    """Call the LLM, parse JSON, and keep only schema-valid entities/relationships.

    Validation has three layers:
      1. entity type must be in the allowed set
      2. relationship type must be in the allowed set, and both endpoints must
         refer to entities we actually extracted (no dangling edges)
      3. the endpoints' TYPES must match an allowed pattern for that relationship
         (e.g. EVALUATED_ON only Method/Paper -> Dataset) — this rejects
         semantically wrong edges even when the type itself is legal
    """
    raw = llm.complete_json(SYSTEM_PROMPT, text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], []  # model returned something unparseable; skip this chunk

    entities = [
        {"name": normalize_name(e["name"]), "type": e["type"].strip()}
        for e in data.get("entities", [])
        if isinstance(e, dict)
        and e.get("name")
        and e.get("type") in _ALLOWED_ENTITIES
    ]
    # name -> type lookup, used to type-check relationship endpoints
    type_of = {e["name"]: e["type"] for e in entities}

    rels = []
    for r in data.get("relationships", []):
        if not isinstance(r, dict) or r.get("type") not in _ALLOWED_RELATIONSHIPS:
            continue
        src = normalize_name(r.get("source", ""))
        tgt = normalize_name(r.get("target", ""))
        if src not in type_of or tgt not in type_of:
            continue  # dangling endpoint
        pattern = (type_of[src], type_of[tgt])
        if pattern not in RELATIONSHIP_PATTERNS[r["type"]]:
            continue  # endpoint TYPES violate this relationship's direction/range
        rels.append({"source": src, "type": r["type"], "target": tgt})

    return entities, rels


def write_chunk(session, chunk_id: str, entities: list[dict], rels: list[dict]) -> None:
    if entities:
        session.run(WRITE_ENTITIES, chunk_id=chunk_id, entities=entities)
    if rels:
        session.run(WRITE_RELATIONSHIPS, rels=rels)


def build_graph(limit: int | None = None, dry_run: bool = False, prefix: str | None = None) -> None:
    driver = get_driver()
    llm = ChatLLM()

    if not dry_run:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(ENTITY_CONSTRAINT)

    chunks = fetch_chunks(driver, limit, prefix)
    print(f"Processing {len(chunks)} chunk(s) with model '{settings.llm_model}'"
          f"{' (DRY RUN — nothing will be written)' if dry_run else ''}.")

    n_entities = n_rels = 0
    with driver.session(database=settings.neo4j_database) as session:
        for chunk in tqdm(chunks, desc="Extracting"):
            entities, rels = extract_from_chunk(llm, chunk["text"])
            n_entities += len(entities)
            n_rels += len(rels)

            if dry_run:
                if entities or rels:
                    print(f"\n--- chunk {chunk['id']} ---")
                    for e in entities:
                        print(f"  ENTITY  {e['type']:<12} {e['name']}")
                    for r in rels:
                        print(f"  REL     {r['source']} -[{r['type']}]-> {r['target']}")
            else:
                write_chunk(session, chunk["id"], entities, rels)

    driver.close()
    verb = "Found" if dry_run else "Wrote"
    print(f"\nDone. {verb} {n_entities} entity mentions and {n_rels} relationships "
          f"across {len(chunks)} chunks.")
    if not dry_run:
        print("Inspect at http://localhost:7474  (try:  MATCH (e:Entity)-[r]->(t) RETURN e,r,t LIMIT 50)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge graph from chunks.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N chunks (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print extracted triples without writing to the graph")
    parser.add_argument("--prefix", type=str, default=None,
                        help="only process chunks whose id starts with this (e.g. libsignal)")
    args = parser.parse_args()
    build_graph(limit=args.limit, dry_run=args.dry_run, prefix=args.prefix)


if __name__ == "__main__":
    main()