"""Phase 4: evaluation harness.

Runs every test question through BOTH retrievers (same generator, same k) and
scores each answer three ways:
  - recall     : fraction of gold items the answer surfaced      (completeness)
  - precision  : fraction of same-type items it named that were correct (noise)
  - judge      : an LLM grader's 1-5 quality score

    uv run python -m graphrag_assistant.evaluate                # both modes, with judge
    uv run python -m graphrag_assistant.evaluate --no-judge     # skip the LLM judge (cheaper)
    uv run python -m graphrag_assistant.evaluate --k 8
"""

import argparse
import csv
import json
import re
import statistics

from .config import settings
from .db import get_driver
from .generate import Generator
from .llm import ChatLLM
from .retrieve import GraphRetriever, VectorRetriever
from .testset import QUESTIONS

RETRIEVERS = {"vector": VectorRetriever, "graph": GraphRetriever}


# --- string matching -----------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _mentions(text: str, name: str) -> bool:
    """Whole-phrase, boundary-aware containment on normalized text."""
    hay = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    needle = " " + _norm(name) + " "
    return needle in hay


def _gold_key(item) -> str:
    """Canonical display key for a gold item that may be a str or alias-list."""
    return item[0] if isinstance(item, list) else item


def _matches_any(text: str, item) -> bool:
    """item may be a string or a list of acceptable surface forms (aliases)."""
    aliases = item if isinstance(item, list) else [item]
    return any(_mentions(text, a) for a in aliases)


def type_vocabulary(driver, item_type: str) -> list[str]:
    with driver.session(database=settings.neo4j_database) as s:
        return [r["name"] for r in
                s.run("MATCH (e:Entity) WHERE e.type = $t RETURN e.name AS name", t=item_type)]


def score_recall_precision(answer: str, gold: list, vocab: list[str]):
    found = [g for g in gold if _matches_any(answer, g)]
    recall = len(found) / len(gold) if gold else None
    # precision: of same-type entities the answer named, how many are gold?
    mentioned = [v for v in vocab if _mentions(answer, v)]
    gold_norms = set()
    for g in gold:
        for a in (g if isinstance(g, list) else [g]):
            gold_norms.add(_norm(a))
    correct = [m for m in mentioned if _norm(m) in gold_norms]
    precision = len(correct) / len(mentioned) if mentioned else None
    return recall, precision


# --- LLM judge -----------------------------------------------------------

JUDGE_SYSTEM = """You grade an answer to a question about a collection of ML \
papers, on a 1-5 scale (5 = fully correct, complete, and grounded; 1 = wrong or \
fabricated). If a reference item list is given, reward covering it and penalize \
fabricated or irrelevant items. For control questions with no answer in the \
papers, a refusal ("I don't know based on the provided documents") deserves a 5.
Return ONLY JSON: {"score": <int 1-5>, "reason": "<one short sentence>"}."""


def judge(llm: ChatLLM, question: str, answer: str, gold) -> tuple[int, str]:
    ref = ", ".join(_gold_key(g) for g in gold) if gold else "(no fixed reference; grade correctness + grounding)"
    user = (f"Question: {question}\nReference correct items: {ref}\n\n"
            f"Answer:\n{answer}")
    try:
        data = json.loads(llm.complete_json(JUDGE_SYSTEM, user))
        return int(data.get("score", 0)), str(data.get("reason", ""))
    except Exception:
        return 0, "judge-parse-error"


# --- runner --------------------------------------------------------------

def run(modes: list[str], k: int, use_judge: bool, hub_max: int = 150):
    driver = get_driver()
    generator = Generator()
    judge_llm = ChatLLM() if use_judge else None

    vocab_cache: dict[str, list[str]] = {}
    rows = []

    for mode in modes:
        retriever = (GraphRetriever(hub_max=hub_max) if mode == 'graph'
                     else RETRIEVERS[mode]())
        print(f"\n=== running mode: {mode} ===")
        for q in QUESTIONS:
            result = retriever.retrieve(q["question"], k=k)
            answer = generator.answer(q["question"], result)

            recall = precision = None
            if q["gold"] is not None and q["item_type"]:
                if q["item_type"] not in vocab_cache:
                    vocab_cache[q["item_type"]] = type_vocabulary(driver, q["item_type"])
                recall, precision = score_recall_precision(
                    answer, q["gold"], vocab_cache[q["item_type"]]
                )
            elif q["gold"] == []:  # control question
                recall = 1.0 if "don't know" in answer.lower() else 0.0

            jscore, jreason = (judge(judge_llm, q["question"], answer, q["gold"])
                               if use_judge else (None, ""))

            rows.append({"mode": mode, "hop": q["hop"], "question": q["question"],
                         "recall": recall, "precision": precision,
                         "judge": jscore, "judge_reason": jreason})
            r = f"{recall:.2f}" if recall is not None else "  - "
            p = f"{precision:.2f}" if precision is not None else "  - "
            j = f"{jscore}" if jscore is not None else "-"
            print(f"  [{q['hop']:<7}] R={r} P={p} J={j}  {q['question'][:48]}")
        retriever.close()

    # --- aggregate ---
    print("\n" + "=" * 64 + "\nAGGREGATE (mean over applicable questions)\n" + "=" * 64)
    print(f"{'mode':<8}{'recall':>10}{'precision':>12}{'judge':>10}")
    for mode in modes:
        mr = [r["recall"] for r in rows if r["mode"] == mode and r["recall"] is not None]
        mp = [r["precision"] for r in rows if r["mode"] == mode and r["precision"] is not None]
        mj = [r["judge"] for r in rows if r["mode"] == mode and r["judge"]]
        def _m(xs): return f"{statistics.mean(xs):.3f}" if xs else "  -  "
        print(f"{mode:<8}{_m(mr):>10}{_m(mp):>12}{_m(mj):>10}")

    with open("eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nPer-question results written to eval_results.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate vector vs graph retrieval.")
    p.add_argument("--mode", choices=["vector", "graph", "both"], default="both")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (cheaper)")
    p.add_argument("--hub-max", type=int, default=150, help="graph: max node degree to traverse through")
    p.add_argument("--sweep-hub", type=str, default="", help="comma list of hub_max values to sweep, e.g. 40,80,150,10000")
    args = p.parse_args()
    modes = ["vector", "graph"] if args.mode == "both" else [args.mode]
    if args.sweep_hub:
        vals = [int(v) for v in args.sweep_hub.split(",")]
        print(f"Sweeping hub_max over {vals} (graph mode, recall/precision only)...")
        import statistics as _st
        from .retrieve import GraphRetriever as _GR
        gen = Generator(); drv = get_driver(); vocab_cache = {}
        for hm in vals:
            r = _GR(hub_max=hm)
            recs, precs = [], []
            for q in QUESTIONS:
                res = r.retrieve(q["question"], k=args.k)
                ans = gen.answer(q["question"], res)
                if q["gold"]:
                    if q["item_type"] and q["item_type"] not in vocab_cache:
                        vocab_cache[q["item_type"]] = type_vocabulary(drv, q["item_type"])
                    voc = vocab_cache.get(q["item_type"], [])
                    rc, pr = score_recall_precision(ans, q["gold"], voc)
                    if rc is not None: recs.append(rc)
                    if pr is not None: precs.append(pr)
            r.close()
            mr = f"{_st.mean(recs):.3f}" if recs else "-"
            mp = f"{_st.mean(precs):.3f}" if precs else "-"
            print(f"  hub_max={hm:<4} recall={mr}  precision={mp}")
        drv.close()
        return
    run(modes, k=args.k, use_judge=not args.no_judge, hub_max=args.hub_max)


if __name__ == "__main__":
    main()