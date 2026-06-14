from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from .models import Document, Event, EventType


def reconstruct_timeline(documents: List[Document], events: Optional[List[Event]] = None) -> Dict[str, List[Event]]:
    timeline: Dict[str, List[Event]] = defaultdict(list)
    included_documents: Set[str] = set()

    if events:
        for event in sorted(events, key=lambda e: e.timestamp or datetime.min):
            date_key = (event.timestamp or datetime.utcnow()).strftime("%Y-%m-%d")
            timeline[date_key].append(event)
            included_documents.update(event.source_ids)

    for document in sorted(documents, key=lambda d: d.timestamp or datetime.min):
        if document.id in included_documents:
            continue
        date_key = (document.timestamp or datetime.utcnow()).strftime("%Y-%m-%d")
        timeline[date_key].append(
            Event(
                id=f"event-{document.id}",
                type=EventType.OTHER,
                description=document.text[:140],
                timestamp=document.timestamp,
                source_ids=[document.id],
            )
        )

    return timeline


def summarize_timeline(timeline: Dict[str, List[Event]]) -> List[str]:
    summary = []
    for date, events in sorted(timeline.items()):
        summary.append(f"{date}:")
        for event in events:
            summary.append(f"  - {event.description}")
    return summary
