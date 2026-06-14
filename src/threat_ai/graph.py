from typing import Dict, List

import networkx as nx

from .models import Entity, Event, KnowledgeFact


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: Entity) -> None:
        self.graph.add_node(entity.text, type=entity.type.value)

    def add_event(self, event: Event) -> None:
        self.graph.add_node(event.id, type="event", description=event.description)
        for person in event.people:
            self.graph.add_node(person, type="person")
            self.graph.add_edge(person, event.id, predicate="attended")
        for location in event.locations:
            self.graph.add_node(location, type="location")
            self.graph.add_edge(event.id, location, predicate="occurred_at")
        for date in event.dates:
            self.graph.add_node(date, type="date")
            self.graph.add_edge(event.id, date, predicate="scheduled_for")
        for document_id in event.source_ids:
            self.graph.add_node(document_id, type="document")
            self.graph.add_edge(event.id, document_id, predicate="mentioned_in")

    def add_fact(self, fact: KnowledgeFact) -> None:
        self.graph.add_node(fact.subject)
        self.graph.add_node(fact.object)
        self.graph.add_edge(
            fact.subject,
            fact.object,
            predicate=fact.predicate,
            source_ids=fact.source_ids,
            confidence=fact.confidence,
        )

    def connect_entities(self, source: Entity, target: Entity, relation: str, metadata: Dict = None) -> None:
        self.add_entity(source)
        self.add_entity(target)
        self.graph.add_edge(
            source.text,
            target.text,
            predicate=relation,
            metadata=metadata or {},
        )

    def connect_event(self, event: Event) -> None:
        self.add_event(event)

    def resolve_entity_alias(self, alias: str, canonical: str) -> None:
        """Merge `alias` node into `canonical` node by reattaching edges and removing alias."""
        if alias not in self.graph or canonical not in self.graph:
            return
        # reattach incoming edges to canonical
        for u, v, d in list(self.graph.in_edges(alias, data=True)):
            self.graph.add_edge(u, canonical, **d)
        # reattach outgoing edges from alias to canonical
        for u, v, d in list(self.graph.out_edges(alias, data=True)):
            self.graph.add_edge(canonical, v, **d)
        # remove alias node
        try:
            self.graph.remove_node(alias)
        except Exception:
            pass

    def link_temporal_events(self, events: List[Event], event_conf_map: Dict[str, float], days_window: int = 7, escalation_delta: float = 0.2) -> None:
        """Create temporal links between events based on shared participants, locations, and time proximity.

        - Adds edge predicate 'followed_by' when events share participants/locations within `days_window` days.
        - Adds edge predicate 'escalates_to' when risk/confidence increases by >= `escalation_delta`.
        """
        # sort events by timestamp
        evs = sorted(events, key=lambda e: e.timestamp or datetime.min)
        for i, a in enumerate(evs):
            for b in evs[i + 1 :]:
                if not a.timestamp or not b.timestamp:
                    continue
                delta_days = (b.timestamp - a.timestamp).days
                if delta_days < 0 or delta_days > days_window:
                    continue
                # shared participants or locations
                shared_people = set(getattr(a, "people", [])) & set(getattr(b, "people", []))
                shared_locations = set(getattr(a, "locations", [])) & set(getattr(b, "locations", []))
                if shared_people or shared_locations:
                    self.graph.add_edge(a.id, b.id, predicate="followed_by")
                    # escalation check
                    ca = event_conf_map.get(a.id, 0.0)
                    cb = event_conf_map.get(b.id, 0.0)
                    if cb - ca >= escalation_delta:
                        self.graph.add_edge(a.id, b.id, predicate="escalates_to")

    def events_for_person(self, person_text: str) -> List[str]:
        """Return event ids connected to a person via 'attended' edge."""
        if person_text not in self.graph:
            return []
        results = []
        for nbr, data in self.graph[person_text].items():
            if data.get("predicate") == "attended":
                results.append(nbr)
        return results

    def top_connected_people(self, top_n: int = 5) -> List[tuple]:
        # compute degree for nodes typed as person
        people = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "person"]
        degrees = [(p, self.graph.degree(p)) for p in people]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    def locations_with_rising_frequency(self, window_days: int = 30) -> List[str]:
        # naive: return locations with many 'occurred_at' incoming edges
        locs = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "location"]
        counts = []
        for loc in locs:
            incoming = [u for u, v, d in self.graph.in_edges(loc, data=True) if d.get("predicate") == "occurred_at"]
            counts.append((loc, len(incoming)))
        return [l for l, c in sorted(counts, key=lambda x: x[1], reverse=True) if c > 0]

    def find_escalation_chains(self, min_length: int = 2) -> List[List[str]]:
        # find paths that include 'escalates_to' edges
        chains = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("predicate") == "escalates_to":
                chains.append([u, v])
        return chains

    def degree_centrality(self):
        return nx.degree_centrality(self.graph)

    def basic_anomaly_detection(self):
        # Return nodes with unusually high degree centrality (top 1%)
        dc = self.degree_centrality()
        if not dc:
            return []
        vals = list(dc.values())
        threshold = sorted(vals)[max(0, int(len(vals) * 0.99) - 1)]
        return [n for n, v in dc.items() if v >= threshold]

    def query_related(self, entity_text: str) -> List[Dict]:
        if entity_text not in self.graph:
            return []
        return [
            {
                "target": target,
                "predicate": data.get("predicate"),
                "metadata": data,
            }
            for target, data in self.graph[entity_text].items()
        ]

    def facts(self) -> List[Dict]:
        return [
            {
                "subject": u,
                "object": v,
                "predicate": d.get("predicate"),
                "metadata": d,
            }
            for u, v, d in self.graph.edges(data=True)
        ]
