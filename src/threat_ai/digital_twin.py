from typing import Dict, List, Any

from .graph import KnowledgeGraph
from .models import KnowledgeFact


def _attr(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def create_belief(subject: str, predicate: str, object: str, confidence: float, source_ids: List[str], explanation: str = None) -> KnowledgeFact:
    return KnowledgeFact(subject=subject, predicate=predicate, object=object, source_ids=source_ids, confidence=confidence)


def generate_beliefs_from_graph(graph: KnowledgeGraph, events: List[Any], event_alerts: Dict[str, Any], min_confidence: float = 0.2) -> List[KnowledgeFact]:
    beliefs: List[KnowledgeFact] = []
    person_nodes = [n for n, d in graph.graph.nodes(data=True) if d.get("type") == "person"]
    location_nodes = [n for n, d in graph.graph.nodes(data=True) if d.get("type") == "location"]

    # Person-location affinity beliefs
    for person in person_nodes:
        attended = graph.events_for_person(person)
        location_counts = {}
        for event_id in attended:
            for _, loc, data in graph.graph.out_edges(event_id, data=True):
                if data.get("predicate") == "occurred_at":
                    location_counts[loc] = location_counts.get(loc, 0) + 1
        for loc, count in location_counts.items():
            if count >= 2:
                confidence = min(0.95, 0.2 + 0.15 * count)
                if confidence >= min_confidence:
                    beliefs.append(create_belief(person, "frequently_at", loc, confidence, [event_id for event_id in attended], f"Attended {count} events at {loc}"))

    # Group affinity beliefs based on repeated co-attendance
    for person in person_nodes:
        attended = graph.events_for_person(person)
        if len(attended) < 2:
            continue
        co_attendance = {}
        for event_id in attended:
            for nbr, data in graph.graph.in_edges(event_id, data=True):
                if data.get("predicate") == "attended" and nbr != person:
                    co_attendance[nbr] = co_attendance.get(nbr, 0) + 1
        for peer, count in co_attendance.items():
            if count >= 2:
                confidence = min(0.9, 0.25 + 0.2 * count)
                if confidence >= min_confidence:
                    beliefs.append(create_belief(person, "frequently_with", peer, confidence, [event_id for event_id in attended], f"Attended {count} shared events with {peer}"))

    # Escalation chain beliefs
    escalation_edges = [(u, v) for u, v, d in graph.graph.edges(data=True) if d.get("predicate") == "escalates_to"]
    for source, target in escalation_edges:
        confidence = 0.75
        beliefs.append(create_belief(source, "escalates_to", target, confidence, [], "Observed escalation chain in event graph"))

    # Risk cluster beliefs from event alerts
    for event in events:
        event_id = _attr(event, "id", _attr(event, "event_id", ""))
        alert = event_alerts.get(event_id)
        if not alert:
            continue
        if getattr(alert, "confidence", 0.0) >= 0.7:
            people = _attr(event, "people", []) or []
            belief = create_belief(event_id, "high_risk_cluster", ",".join(sorted(people)), 0.6, getattr(alert, "source_ids", []), "Event is a high-risk cluster with repeated participants")
            beliefs.append(belief)

    return beliefs


def generate_competing_hypotheses(graph: KnowledgeGraph, beliefs: List[KnowledgeFact], events: List[Any], max_hypotheses: int = 3) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    # hypothesis A: group-based coordinated activity
    top_people = graph.top_connected_people(top_n=5)
    if top_people:
        people = [person for person, _ in top_people]
        candidates.append({
            "hypothesis": f"A coordinated activity cluster is forming around {people[:3]}",
            "probability": 0.45,
            "evidence": [f"Top connected people: {', '.join(people[:3])}"],
        })

    # hypothesis B: location-driven escalation
    hubs = graph.locations_with_rising_frequency(window_days=30)
    if hubs:
        candidates.append({
            "hypothesis": f"The location {hubs[0]} is becoming a hub for escalating activity", 
            "probability": 0.3,
            "evidence": [f"Location {hubs[0]} has {len([e for _, e, d in graph.graph.in_edges(hubs[0], data=True) if d.get('predicate') == 'occurred_at'])} events"],
        })

    # hypothesis C: hidden fraud ring or harassment campaign
    repeated_links = [b for b in beliefs if b.predicate in ("frequently_with", "frequently_at")][:3]
    if repeated_links:
        candidates.append({
            "hypothesis": "There is a hidden organized ring behind repeat co-attendance and shared locations.",
            "probability": 0.25,
            "evidence": [f"{b.subject} {b.predicate} {b.object} (confidence {b.confidence:.2f})" for b in repeated_links],
        })

    # normalize probabilities
    total = sum(h["probability"] for h in candidates) or 1.0
    for h in candidates:
        h["probability"] = round(h["probability"] / total, 2)

    # sort and trim
    return sorted(candidates, key=lambda x: x["probability"], reverse=True)[:max_hypotheses]
