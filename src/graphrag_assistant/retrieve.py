"""Phase 3: retrieval.

Two retrievers behind ONE interface (.retrieve(question, k) -> Retrieval):

  VectorRetriever  — baseline. Embed the question, return the nearest chunks.
  GraphRetriever   — GraphRAG. Same vector entry, THEN traverse one hop into the
                     domain graph from the entities those chunks mention, and
                     return the connected facts alongside the chunks.

Sharing the interface is what lets Phase 4 compare them on equal footing.
"""

import math

from dataclasses import dataclass, field

from .config import settings
from .db import CHUNK_VECTOR_INDEX, get_driver
from .llm import Embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    text: str
    score: float


@dataclass
class Retrieval:
    """What a retriever returns: text chunks, plus (for GraphRAG) graph facts."""
    chunks: list[RetrievedChunk]
    graph_facts: list[str] = field(default_factory=list)


# --- Vector (baseline) ---------------------------------------------------

VECTOR_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score
MATCH (p:Paper)-[:HAS_CHUNK]->(node)
RETURN node.id AS chunk_id, p.id AS paper_id, node.text AS text, score
ORDER BY score DESC
"""


class VectorRetriever:
    name = "vector"

    def __init__(self) -> None:
        self.driver = get_driver()
        self.embedder = Embedder()

    def _embed(self, question: str) -> list[float]:
        return self.embedder.embed([question])[0]

    def retrieve(self, question: str, k: int = 5) -> Retrieval:
        vec = self._embed(question)
        with self.driver.session(database=settings.neo4j_database) as session:
            rows = session.run(VECTOR_QUERY, index=CHUNK_VECTOR_INDEX, k=k, embedding=vec)
            chunks = [
                RetrievedChunk(r["chunk_id"], r["paper_id"], r["text"], r["score"])
                for r in rows
            ]
        return Retrieval(chunks=chunks)

    def close(self) -> None:
        self.driver.close()


# --- Graph (GraphRAG) ----------------------------------------------------

# Vector-search for entry chunks, then from the entities those chunks MENTION,
# collect every one-hop domain relationship (both directions), rendered as a
# directed triple string. collect() drops nulls, so chunks with no entities or
# no neighbours simply contribute nothing.
# Part A: fetch the entry chunks (for prose grounding + citation).
CHUNKS_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node AS chunk, score
MATCH (p:Paper)-[:HAS_CHUNK]->(chunk)
RETURN chunk.id AS chunk_id, p.id AS paper_id, chunk.text AS text, score
ORDER BY score DESC
"""

# Part B: collect one-hop facts from TWO seed sources, with a hub-node filter.
#   seed 1 (semantic): entities mentioned in the retrieved chunks
#   seed 2 (lexical):  entities whose name appears in the question terms
# The WHERE clause drops any relationship touching a node whose total degree
# exceeds $hub_max — that skips garbage-magnet hubs like 'Traffic Signal Control'.
FACTS_QUERY = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.id IN $chunk_ids
RETURN e AS seed
UNION
MATCH (e:Entity)
WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS term)
RETURN e AS seed
"""

FACTS_FROM_SEEDS = """
UNWIND $seed_names AS sname
MATCH (e:Entity {name: sname})
MATCH (e)-[r]-(nbr:Entity)
WHERE count{ (nbr)--() } <= $hub_max AND count{ (e)--() } <= $hub_max
RETURN DISTINCT startNode(r).name + ' -[' + type(r) + ']-> ' + endNode(r).name AS fact
LIMIT $limit
"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class GraphRetriever:
    """GraphRAG with relevance-ranked facts.

    Traversal collects the one-hop neighborhood (a candidate pool). We then
    embed each candidate fact and the question, and keep only the top_facts
    most similar — so a flood of off-topic neighbourhood facts (e.g. everything
    connected to a high-degree hub node) gets filtered out before generation.
    """

    name = "graph"

    def __init__(self, candidate_facts: int = 120, top_facts: int = 15,
                 hub_max: int = 150) -> None:
        self.driver = get_driver()
        self.embedder = Embedder()
        self.candidate_facts = candidate_facts
        self.top_facts = top_facts
        self.hub_max = hub_max  # skip traversal through nodes with degree > hub_max

    def _rank_facts(self, question: str, facts: list[str]) -> list[str]:
        if len(facts) <= self.top_facts:
            return facts
        # one batched embedding call: the question + all candidate facts
        vectors = self.embedder.embed([question] + facts)
        qv, fvs = vectors[0], vectors[1:]
        scored = sorted(zip(facts, fvs), key=lambda p: _cosine(qv, p[1]), reverse=True)
        return [f for f, _ in scored[: self.top_facts]]

    _STOP = {"the", "which", "what", "does", "for", "and", "are", "was", "use",
             "used", "uses", "with", "that", "this", "from", "into", "how", "its",
             "their", "compare", "against", "evaluate", "evaluated", "than", "more",
             "paper", "papers", "method", "methods", "model", "models", "dataset",
             "datasets", "benchmark", "benchmarks", "algorithm", "algorithms"}

    @classmethod
    def _question_terms(cls, question: str) -> list[str]:
        # content word tokens >= 3 chars, minus common/generic words, so lexical
        # anchoring seeds from distinctive names (e.g. "libsignal") not "does".
        import re
        toks = re.findall(r"[a-z0-9]+", question.lower())
        return [t for t in toks if len(t) >= 3 and t not in cls._STOP]

    def retrieve(self, question: str, k: int = 5) -> Retrieval:
        vec = self.embedder.embed([question])[0]
        terms = self._question_terms(question)
        with self.driver.session(database=settings.neo4j_database) as session:
            chunk_rows = list(session.run(
                CHUNKS_QUERY, index=CHUNK_VECTOR_INDEX, k=k, embedding=vec))
            chunk_ids = [r["chunk_id"] for r in chunk_rows]

            # seed 1 (chunk mentions) UNION seed 2 (question-matched entities)
            seed_rows = session.run(FACTS_QUERY, chunk_ids=chunk_ids, terms=terms)
            seed_names = [r["seed"]["name"] for r in seed_rows]

            facts = []
            if seed_names:
                facts = [r["fact"] for r in session.run(
                    FACTS_FROM_SEEDS, seed_names=seed_names,
                    hub_max=self.hub_max, limit=self.candidate_facts)]

        chunks = [RetrievedChunk(r["chunk_id"], r["paper_id"], r["text"], r["score"])
                  for r in chunk_rows]
        facts = self._rank_facts(question, facts)
        return Retrieval(chunks=chunks, graph_facts=facts)

    def close(self) -> None:
        self.driver.close()