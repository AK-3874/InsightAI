from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .storage import store_memory_record, query_memories


@dataclass
class MemoryRecord:
    investigation_id: str
    summary: str
    pattern: str
    outcome: str
    tags: List[str]
    note: Optional[str] = None
    timestamp: str = datetime.utcnow().isoformat()


def remember_investigation(conn, investigation_id: str, summary: str, pattern: str, outcome: str, tags: List[str], note: str = None):
    if tags is None:
        tags = []
    store_memory_record(conn, investigation_id, summary, pattern, outcome, tags, note)


def retrieve_similar_memories(conn, tags: List[str] = None, keywords: List[str] = None):
    return query_memories(conn, tags=tags, keywords=keywords)
