"""Phase 2.5: entity resolution.

Baseline extraction merges entities only on EXACT name match, leaving
surface-variant duplicates ('TokUR (EU)', 'TokUR(EU)', 'TokUR EU', ...).
This pass groups same-type entities whose names normalize to the same key and
merges each group into one canonical node, combining relationships, via APOC's
apoc.refactor.mergeNodes.

    uv run python -m graphrag_assistant.resolve_entities --report   # preview only
    uv run python -m graphrag_assistant.resolve_entities            # apply merges
"""

import argparse
import re
from collections import defaultdict

from .config import settings
from .db import get_driver


def normalize_key(name: str) -> str:
    """Aggressive key for matching surface variants: lowercase, drop the
    '(..., Ours)' aside, then keep only letters and digits. This collapses
    spacing/punctuation/case differences ('Color-Cube' == 'Color Cube') while
    keeping genuinely different names apart ('MATH' != 'MATH500')."""
    s = name.lower()
    s = re.sub(r"\bours\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


FETCH = "MATCH (e:Entity) RETURN e.name AS name, e.type AS type"

MERGE_GROUP = """
MATCH (e:Entity) WHERE e.name IN $names
WITH e ORDER BY size(e.name) ASC, e.name ASC
WITH collect(e) AS nodes
CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true})
YIELD node
SET node.name = $canonical
RETURN node.name AS name
"""


def canonical_name(names: list[str]) -> str:
    return sorted(names, key=lambda n: (len(n), n))[0]  # shortest, then alphabetical


def find_groups(driver) -> dict:
    with driver.session(database=settings.neo4j_database) as s:
        rows = [dict(r) for r in s.run(FETCH)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["type"], normalize_key(r["name"]))].append(r["name"])
    return {k: sorted(set(v)) for k, v in groups.items() if len(set(v)) > 1}


def resolve(report: bool = False) -> None:
    driver = get_driver()
    groups = find_groups(driver)
    print(f"Found {len(groups)} duplicate group(s).")
    merged = 0
    with driver.session(database=settings.neo4j_database) as s:
        for (etype, _key), names in sorted(groups.items()):
            canon = canonical_name(names)
            if report:
                print(f"  [{etype:<11}] {names}  ->  '{canon}'")
            else:
                s.run(MERGE_GROUP, names=names, canonical=canon)
                merged += len(names) - 1
    driver.close()
    print("\n(report only — nothing changed)" if report
          else f"\nMerged away {merged} duplicate node(s).")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge surface-variant duplicate entities.")
    p.add_argument("--report", action="store_true", help="preview proposed merges; change nothing")
    resolve(report=p.parse_args().report)


if __name__ == "__main__":
    main()
