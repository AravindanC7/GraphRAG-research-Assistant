"""Phase 3 entrypoint: ask a question over your papers, via either retriever.

    uv run python -m graphrag_assistant.ask "What datasets is TokUR evaluated on?"
    uv run python -m graphrag_assistant.ask --mode graph "What datasets is TokUR evaluated on?"
    uv run python -m graphrag_assistant.ask --mode graph --k 8 "..."
"""

import argparse

from .generate import Generator
from .retrieve import GraphRetriever, VectorRetriever

RETRIEVERS = {"vector": VectorRetriever, "graph": GraphRetriever}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over your paper corpus.")
    parser.add_argument("question", help="your question (wrap in quotes)")
    parser.add_argument("--mode", choices=RETRIEVERS, default="vector",
                        help="retrieval strategy (default: vector)")
    parser.add_argument("--k", type=int, default=5, help="number of chunks to retrieve")
    parser.add_argument("--hub-max", type=int, default=150, help="graph mode: max node degree to traverse through")
    args = parser.parse_args()

    retriever = (GraphRetriever(hub_max=args.hub_max) if args.mode == 'graph'
                 else RETRIEVERS[args.mode]())
    generator = Generator()

    result = retriever.retrieve(args.question, k=args.k)
    print(f"\nRetrieved {len(result.chunks)} chunks via '{retriever.name}' search:")
    for c in result.chunks:
        preview = c.text[:66].replace("\n", " ")
        print(f"  [{c.chunk_id}] score={c.score:.3f}  {preview}...")
    if result.graph_facts:
        print(f"\n+ {len(result.graph_facts)} knowledge-graph facts, e.g.:")
        for f in result.graph_facts[:8]:
            print(f"  - {f}")

    answer = generator.answer(args.question, result)
    print("\n" + "=" * 60 + f"\nANSWER  ({retriever.name})\n" + "=" * 60)
    print(answer)
    print("=" * 60)

    retriever.close()


if __name__ == "__main__":
    main()