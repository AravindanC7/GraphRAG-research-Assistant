"""Phase 3: generation.

Takes a question plus a Retrieval (chunks, and for GraphRAG also graph facts)
and asks the LLM to answer STRICTLY from that context, with citations. This is
the 'Augment' + 'Generate' half of RAG; it serves both retrievers unchanged.
"""

from .llm import ChatLLM
from .retrieve import Retrieval

SYSTEM_PROMPT = """You are a research assistant answering questions about a \
collection of machine-learning papers.

You are given (1) context passages from the papers and (2) optional \
knowledge-graph facts (directed relationships extracted from the papers).

Rules:
- Answer using ONLY the provided passages and facts. Do NOT use outside knowledge.
- The knowledge-graph facts are reliable structured relationships — use them, \
especially for questions asking to list or connect things.
- If the context does not contain the answer, say exactly: "I don't know based \
on the provided documents."
- Cite passages with their [chunk_id] markers. Be concise and precise."""


def build_context(retrieval: Retrieval) -> str:
    parts = []
    if retrieval.chunks:
        passages = "\n\n---\n\n".join(
            f"[{c.chunk_id}] (from paper {c.paper_id})\n{c.text}"
            for c in retrieval.chunks
        )
        parts.append("Context passages:\n\n" + passages)
    if retrieval.graph_facts:
        facts = "\n".join(f"- {f}" for f in retrieval.graph_facts)
        parts.append("Knowledge-graph facts:\n\n" + facts)
    return "\n\n========\n\n".join(parts)


class Generator:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.llm = ChatLLM(api_key=api_key, model=model)

    def answer(self, question: str, retrieval: Retrieval) -> str:
        if not retrieval.chunks and not retrieval.graph_facts:
            return "No relevant context was retrieved."
        user = f"{build_context(retrieval)}\n\n========\n\nQuestion: {question}"
        return self.llm.complete(SYSTEM_PROMPT, user)
