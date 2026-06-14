from datetime import datetime
from typing import List, Tuple

from .models import Alert, Document, Entity, EntityType, RiskLevel, RiskType

THREAT_KEYWORDS = [
    "kill",
    "explode",
    "bomb",
    "shoot",
    "attack",
    "weapon",
    "raid",
    "threat",
    "hostage",
    "pay the price",
    "regret it",
]

HARASSMENT_KEYWORDS = [
    "idiot",
    "stupid",
    "shut up",
    "worthless",
    "loser",
    "harass",
    "insult",
    "hate",
]

FRAUD_KEYWORDS = [
    "bank details",
    "credit card",
    "account number",
    "social security",
    "wire transfer",
    "send me money",
    "password",
    "verify your identity",
]

SUSPICIOUS_MEETING_KEYWORDS = [
    "meet behind",
    "midnight",
    "warehouse",
    "alley",
    "after hours",
    "private meeting",
    "secret location",
]

SELF_HARM_KEYWORDS = [
    "kill myself",
    "end it",
    "give up",
    "can't go on",
    "depressed",
]

ORGANIZED_ACTIVITY_KEYWORDS = [
    "coordinate",
    "organized",
    "team up",
    "group session",
    "plan together",
    "recruit",
]

CATEGORY_KEYWORDS = [
    (RiskType.THREAT, THREAT_KEYWORDS),
    (RiskType.HARASSMENT, HARASSMENT_KEYWORDS),
    (RiskType.FRAUD, FRAUD_KEYWORDS),
    (RiskType.SUSPICIOUS_MEETING, SUSPICIOUS_MEETING_KEYWORDS),
    (RiskType.SELF_HARM, SELF_HARM_KEYWORDS),
    (RiskType.ORGANIZED_ACTIVITY, ORGANIZED_ACTIVITY_KEYWORDS),
]


def detect_risk_indicators(text: str) -> Tuple[RiskType, List[str]]:
    lower = text.lower()
    reasons = []
    risk_type = RiskType.NORMAL

    for category, keywords in CATEGORY_KEYWORDS:
        hits = [kw for kw in keywords if kw in lower]
        if hits:
            if risk_type == RiskType.NORMAL or category != RiskType.NORMAL:
                risk_type = category
            for hit in hits:
                reasons.append(f"{category.value.replace('_', ' ').title()} indicator: '{hit}'")

    if any(word in lower for word in ["weapon", "bomb", "shoot"]):
        reasons.append("Weapon-related language detected")

    return risk_type, reasons


def score_risk(document: Document, entities: List[Entity]) -> Alert:
    risk_type, reasons = detect_risk_indicators(document.text)

    if any(e.type == EntityType.WEAPON for e in entities):
        reasons.append("Weapon reference found")
        if risk_type == RiskType.NORMAL:
            risk_type = RiskType.THREAT

    if not reasons:
        level = RiskLevel.LOW
        summary = "No strong risk indicators detected."
    elif len(reasons) == 1:
        level = RiskLevel.MEDIUM
        summary = "A potential risk indicator was found."
    elif len(reasons) <= 3:
        level = RiskLevel.HIGH
        summary = "Multiple risk indicators were detected."
    else:
        level = RiskLevel.CRITICAL
        summary = "Critical risk because multiple indicators were found."

    explanation = (
        "; ".join(reasons) if reasons else "No risk indicators were identified."
    )

    return Alert(
        id=f"alert-{document.id}",
        level=level,
        risk_type=risk_type,
        confidence=min(0.99, 0.5 + 0.1 * len(reasons)) if reasons else 0.0,
        summary=summary,
        reasons=reasons,
        source_ids=[document.id],
        created_at=datetime.utcnow(),
        explanation=explanation,
    )


def score_event(event, documents_by_id: dict, document_entities: dict) -> Alert:
    # Aggregate reasons and entities across messages in the event
    all_reasons = []
    all_entities = []
    for msg_id in getattr(event, "related_message_ids", []) or []:
        doc = documents_by_id.get(msg_id)
        if not doc:
            continue
        alert = score_risk(doc, document_entities.get(msg_id, []))
        all_reasons.extend(alert.reasons)
        all_entities.extend(document_entities.get(msg_id, []))

    # dedupe reasons
    reasons = []
    for r in all_reasons:
        if r not in reasons:
            reasons.append(r)

    # basic fusion for confidence: combine evidence count and presence of weapon/entity
    base_conf = min(0.99, 0.3 + 0.15 * len(reasons))
    if any(e.type == EntityType.WEAPON for e in all_entities):
        base_conf = min(0.99, base_conf + 0.2)

    if not reasons:
        level = RiskLevel.LOW
        summary = "No significant risk at event level."
    elif len(reasons) == 1:
        level = RiskLevel.MEDIUM
        summary = "Potential risk observed at event level."
    elif len(reasons) <= 3:
        level = RiskLevel.HIGH
        summary = "High risk observed across event messages."
    else:
        level = RiskLevel.CRITICAL
        summary = "Critical risk observed across event messages."

    event_type = getattr(event, "type", None)
    if hasattr(event_type, "value"):
        event_type_value = event_type.value
    else:
        event_type_value = str(event_type).lower()

    risk_type = RiskType.OTHER
    if event_type_value == "meeting":
        risk_type = RiskType.SUSPICIOUS_MEETING
    elif event_type_value == "threat":
        risk_type = RiskType.THREAT
    elif event_type_value == "recruitment":
        risk_type = RiskType.ORGANIZED_ACTIVITY
    elif event_type_value == "purchase":
        risk_type = RiskType.FRAUD
    elif event_type_value == "other":
        risk_type = RiskType.OTHER

    return Alert(
        id=f"event-alert-{getattr(event, 'id', 'unknown')}",
        level=level,
        risk_type=risk_type,
        confidence=base_conf,
        summary=summary,
        reasons=reasons,
        source_ids=getattr(event, "source_ids", []),
        created_at=datetime.utcnow(),
        explanation=("; ".join(reasons) if reasons else None),
    )
