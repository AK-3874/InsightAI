from typing import List

from .graph import KnowledgeGraph
from .models import Document, Entity


def answer_natural_language_query(query: str, graph: KnowledgeGraph, documents: List[Document]) -> dict:
    if "connected to" in query.lower():
        tokens = query.split("connected to")
        if len(tokens) > 1:
            target = tokens[1].strip().strip(".?\n")
            related = graph.query_related(target)
            return {"query": query, "results": related}

    if "mentioned explosives" in query.lower():
        matching = []
        for document in documents:
            if "explosive" in document.text.lower() or "explosives" in document.text.lower():
                matching.append({"id": document.id, "text": document.text})
        return {"query": query, "results": matching}

    return {"query": query, "results": [], "note": "Query type not yet supported."}
