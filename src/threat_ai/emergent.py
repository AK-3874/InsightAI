from typing import Dict, List, Any
from collections import Counter

from .graph import KnowledgeGraph
from .models import KnowledgeFact


def find_rapidly_forming_groups(graph: KnowledgeGraph, min_growth: int = 3) -> List[Dict[str, Any]]:
    groups = []
    people = [n for n, d in graph.graph.nodes(data=True) if d.get("type") == "person"]
    for person in people:
        neighbors = [nbr for nbr in graph.graph.neighbors(person) if graph.graph.nodes[nbr].get("type") == "event"]
        if len(neighbors) >= min_growth:
            groups.append({"person": person, "event_count": len(neighbors), "events": neighbors})
    return sorted(groups, key=lambda x: x["event_count"], reverse=True)


def find_location_hubs(graph: KnowledgeGraph, min_events: int = 3) -> List[Dict[str, Any]]:
    hubs = []
    locations = [n for n, d in graph.graph.nodes(data=True) if d.get("type") == "location"]
    for loc in locations:
        events = [u for u, v, d in graph.graph.in_edges(loc, data=True) if d.get("predicate") == "occurred_at"]
        if len(events) >= min_events:
            hubs.append({"location": loc, "event_count": len(events), "events": events})
    return sorted(hubs, key=lambda x: x["event_count"], reverse=True)


def find_unusual_connection_growth(graph: KnowledgeGraph, threshold: int = 5) -> List[Dict[str, Any]]:
    growth = []
    for node, degree in graph.graph.degree():
        if degree >= threshold and graph.graph.nodes[node].get("type") == "person":
            growth.append({"entity": node, "degree": degree})
    return sorted(growth, key=lambda x: x["degree"], reverse=True)


def _event_attr(event, attr, default=None):
    return getattr(event, attr, event.get(attr, default) if isinstance(event, dict) else default)


def find_repeated_low_risk_patterns(events: List[Any], event_alerts: Dict[str, Any], min_repeat: int = 4) -> List[Dict[str, Any]]:
    patterns = []
    event_types = Counter(_event_attr(e, "type", "unknown") for e in events)
    for event_type, count in event_types.items():
        if count >= min_repeat:
            low_risk_count = sum(
                1
                for e in events
                if event_alerts.get(_event_attr(e, "id", ""), {}).get("confidence", 0.0) < 0.4
                and _event_attr(e, "type", "unknown") == event_type
            )
            if low_risk_count >= min_repeat:
                patterns.append({"event_type": event_type, "low_risk_count": low_risk_count, "total_count": count})
    return sorted(patterns, key=lambda x: x["low_risk_count"], reverse=True)
