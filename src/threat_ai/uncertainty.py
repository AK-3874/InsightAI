import math
from typing import List


def entropy_from_probs(probs: List[float]) -> float:
    # expect probs sum to 1
    h = 0.0
    for p in probs:
        if p <= 0:
            continue
        h -= p * math.log(p + 1e-12)
    return h


def ensemble_disagreement(prob_vectors: List[List[float]]) -> float:
    # measure average pairwise L2 distance between prediction vectors
    import itertools
    import math

    if not prob_vectors:
        return 0.0
    dists = []
    for a, b in itertools.combinations(prob_vectors, 2):
        s = 0.0
        for x, y in zip(a, b):
            s += (x - y) ** 2
        dists.append(math.sqrt(s))
    return sum(dists) / len(dists) if dists else 0.0


def label_uncertainty(score: float, ent: float, disagreement: float) -> str:
    # heuristics: high entropy or high disagreement => high uncertainty
    if ent > 1.0 or disagreement > 0.5:
        return "high"
    if ent > 0.5 or disagreement > 0.2:
        return "medium"
    return "low"
