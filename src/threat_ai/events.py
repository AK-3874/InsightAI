from collections import defaultdict
from typing import Dict, List, Optional, Set

from .models import Document, Entity, Event, EventType, EntityType, RiskType
from .risk import detect_risk_indicators

EVENT_TYPE_KEYWORDS = {
    EventType.MEETING: ["meet", "meeting", "gather", "assemble", "schedule", "arrive", "attend", "warehouse"],
    EventType.THREAT: ["threat", "warn", "danger", "consequences", "regret", "final warning", "price to pay"],
    EventType.RECRUITMENT: ["recruit", "join", "volunteer", "team up", "support", "expand", "coordinate"],
    EventType.PURCHASE: ["order", "buy", "purchase", "funds", "payment", "transfer"],
    EventType.TRAVEL: ["arrive", "leave", "travel", "depart", "visit", "transport"],
}


def extract_event_entities(entities: List[Entity]) -> Dict[str, List[str]]:
    people = []
    locations = []
    dates = []

    seen: Set[str] = set()
    for entity in entities:
        normalized = entity.text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        if entity.type == EntityType.PERSON:
            people.append(normalized)
        elif entity.type == EntityType.LOCATION:
            locations.append(normalized)
        elif entity.type == EntityType.DATE:
            dates.append(normalized)

    return {
        "people": people,
        "locations": locations,
        "dates": dates,
    }


def suggest_event_type(document: Document) -> EventType:
    text = document.text.lower()
    risk_type, _ = detect_risk_indicators(document.text)

    if risk_type == RiskType.SUSPICIOUS_MEETING:
        return EventType.MEETING
    if risk_type == RiskType.THREAT:
        return EventType.THREAT
    if risk_type == RiskType.ORGANIZED_ACTIVITY:
        return EventType.RECRUITMENT
    if risk_type == RiskType.FRAUD:
        return EventType.PURCHASE
    if risk_type == RiskType.SELF_HARM:
        return EventType.OTHER

    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return event_type
    return EventType.OTHER


def merge_entities(existing: List[Entity], new_entities: List[Entity]) -> List[Entity]:
    seen = {(ent.text, ent.type, ent.start, ent.end) for ent in existing}
    merged = existing.copy()
    for entity in new_entities:
        key = (entity.text, entity.type, entity.start, entity.end)
        if key not in seen:
            merged.append(entity)
            seen.add(key)
    return merged


def group_documents_into_events(documents: List[Document], document_entities: Dict[str, List[Entity]]) -> List[Event]:
    events: List[Event] = []
    next_id = 1

    for document in sorted(documents, key=lambda d: d.timestamp or None):
        entities = document_entities.get(document.id, [])
        extracted = extract_event_entities(entities)
        event_type = suggest_event_type(document)

        matched_event: Optional[Event] = None
        for event in events:
            shared_people = set(event.people) & set(extracted["people"])
            shared_locations = set(event.locations) & set(extracted["locations"])
            shared_dates = set(event.dates) & set(extracted["dates"])
            shared_entities = {ent.text for ent in event.entities} & {ent.text for ent in entities}
            if event.type == event_type and (shared_people or shared_locations or shared_dates or shared_entities):
                matched_event = event
                break

        if matched_event is None:
            matched_event = Event(
                id=f"event-{next_id}",
                type=event_type,
                description=document.text[:140],
                timestamp=document.timestamp,
                source_ids=[document.id],
                entities=entities,
                people=extracted["people"],
                locations=extracted["locations"],
                dates=extracted["dates"],
                related_message_ids=[document.id],
            )
            next_id += 1
            events.append(matched_event)
        else:
            matched_event.source_ids.append(document.id)
            matched_event.related_message_ids.append(document.id)
            matched_event.entities = merge_entities(matched_event.entities, entities)
            matched_event.people = sorted(set(matched_event.people) | set(extracted["people"]))
            matched_event.locations = sorted(set(matched_event.locations) | set(extracted["locations"]))
            matched_event.dates = sorted(set(matched_event.dates) | set(extracted["dates"]))
            if not matched_event.description and document.text:
                matched_event.description = document.text[:140]

    return events
