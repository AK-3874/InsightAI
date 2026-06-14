import random
from typing import List, Dict, Any
from datetime import timedelta
import numpy as np


def simulate_slang_shift(documents: List[Dict[str, Any]], slang_map: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    for d in documents:
        text = d.get("text", "")
        for k, v in slang_map.items():
            text = text.replace(k, v)
        new = dict(d)
        new["text"] = text
        out.append(new)
    return out


def simulate_missing_data(documents: List[Dict[str, Any]], drop_rate: float = 0.1) -> List[Dict[str, Any]]:
    out = []
    for d in documents:
        new = dict(d)
        if random.random() < drop_rate:
            # drop text or timestamp randomly
            if random.random() < 0.5:
                new.pop("text", None)
            else:
                new.pop("timestamp", None)
        out.append(new)
    return out


def simulate_delays(documents: List[Dict[str, Any]], max_delay_seconds: int = 3600) -> List[Dict[str, Any]]:
    out = []
    for d in documents:
        new = dict(d)
        ts = new.get("timestamp")
        if ts is not None:
            # assume timestamp is a datetime or ISO string; add seconds as integer
            try:
                from dateutil import parser

                t = parser.isoparse(ts) if isinstance(ts, str) else ts
                delay = random.randint(0, max_delay_seconds)
                new["timestamp"] = (t + timedelta(seconds=delay)).isoformat()
            except Exception:
                pass
        out.append(new)
    return out


def simulate_partial_graph(graph, node_drop_rate: float = 0.1, edge_drop_rate: float = 0.1):
    # graph is expected to be a networkx-like object with nodes() and edges()
    import networkx as nx

    g = graph.copy()
    nodes = list(g.nodes())
    for n in nodes:
        if random.random() < node_drop_rate:
            if n in g:
                g.remove_node(n)
    edges = list(g.edges())
    for u, v in edges:
        if random.random() < edge_drop_rate and g.has_edge(u, v):
            g.remove_edge(u, v)
    return g


def measure_embedding_drift(emb_old: List[List[float]], emb_new: List[List[float]]):
    # compute mean cosine distance between two embedding sets (must be same length)
    from numpy.linalg import norm

    if len(emb_old) == 0 or len(emb_new) == 0:
        return None
    L = min(len(emb_old), len(emb_new))
    ds = []
    for i in range(L):
        a = np.array(emb_old[i])
        b = np.array(emb_new[i])
        denom = (norm(a) * norm(b))
        if denom == 0:
            continue
        cos = np.dot(a, b) / denom
        ds.append(1.0 - cos)
    return float(sum(ds) / len(ds)) if ds else None
