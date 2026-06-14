from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from .models import Document


def get_embeddings(texts: List[str]):
    if SentenceTransformer:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, convert_to_numpy=True)
    # fallback: return None to signal using text-similarity
    return None


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    dot = float((a * b).sum())
    na = float((a * a).sum())
    nb = float((b * b).sum())
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(token for token in a.lower().split() if token)
    sb = set(token for token in b.lower().split() if token)
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def cluster_documents(
    documents: List[Document],
    window_hours: int = 6,
    similarity_threshold: float = 0.6,
):
    texts = [d.text for d in documents]
    embeddings = get_embeddings(texts)

    clusters: List[List[int]] = []
    cluster_times: List[datetime] = []

    def time_ok(doc_time: Optional[datetime], cluster_time: datetime) -> bool:
        if doc_time is None:
            return True
        return abs((doc_time - cluster_time).total_seconds()) <= window_hours * 3600

    for idx, doc in enumerate(documents):
        placed = False
        for ci, cluster in enumerate(clusters):
            # compute similarity against first member of cluster
            other_idx = cluster[0]
            if embeddings is not None:
                sim = cosine(embeddings[idx], embeddings[other_idx])
            else:
                sim = jaccard_similarity(doc.text, documents[other_idx].text)

            if sim >= similarity_threshold and time_ok(doc.timestamp, cluster_times[ci]):
                cluster.append(idx)
                # update cluster time average
                times = [documents[i].timestamp or datetime.utcnow() for i in cluster]
                avg = sum([t.timestamp() for t in times]) / len(times)
                cluster_times[ci] = datetime.fromtimestamp(avg)
                placed = True
                break

        if not placed:
            clusters.append([idx])
            cluster_times.append(doc.timestamp or datetime.utcnow())

    # convert clusters of indices to lists of documents
    result: List[List[Document]] = []
    for cluster in clusters:
        result.append([documents[i] for i in cluster])
    return result
