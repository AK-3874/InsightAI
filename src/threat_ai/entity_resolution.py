from difflib import SequenceMatcher
from typing import List, Tuple


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar_pairs(names: List[str], cutoff: float = 0.8) -> List[Tuple[str, str, float]]:
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            s = similarity(names[i], names[j])
            if s >= cutoff:
                pairs.append((names[i], names[j], s))
    return pairs


def auto_resolve(graph, cutoff: float = 0.85):
    """Automatically resolve similar person nodes in the graph.

    - Finds nodes marked as type 'person' and merges aliases with high similarity.
    - Chooses canonical name as the longest name (heuristic).
    """
    people = [n for n, d in graph.graph.nodes(data=True) if d.get("type") == "person"]
    pairs = find_similar_pairs(people, cutoff=cutoff)
    merged = set()
    for a, b, score in pairs:
        if a in merged or b in merged:
            continue
        # choose canonical
        canonical = a if len(a) >= len(b) else b
        alias = b if canonical == a else a
        graph.resolve_entity_alias(alias, canonical)
        merged.add(alias)
    return merged
