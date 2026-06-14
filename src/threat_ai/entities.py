import re
from typing import List

try:
    import spacy
except ImportError:
    spacy = None

from .models import Entity, EntityType, Document

if spacy:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
else:
    nlp = None

PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\-\.\s]{7,}\d)\b")


def extract_entities(document: Document) -> List[Entity]:
    text = document.text
    entities = []

    for match in PHONE_PATTERN.finditer(text):
        entities.append(
            Entity(
                text=match.group(0),
                type=EntityType.PHONE,
                start=match.start(),
                end=match.end(),
                source_id=document.id,
            )
        )

    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            ent_type = map_spacy_label(ent.label_)
            entities.append(
                Entity(
                    text=ent.text,
                    type=ent_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    source_id=document.id,
                )
            )

    return entities


def map_spacy_label(label: str) -> EntityType:
    if label in {"PERSON"}:
        return EntityType.PERSON
    if label in {"GPE", "LOC", "FAC"}:
        return EntityType.LOCATION
    if label in {"ORG"}:
        return EntityType.ORGANIZATION
    if label in {"DATE", "TIME"}:
        return EntityType.DATE
    return EntityType.OTHER


def detect_weapons(text: str) -> List[Entity]:
    weapon_terms = ["gun", "rifle", "knife", "explosive", "bomb", "weapon"]
    entities = []
    for term in weapon_terms:
        for match in re.finditer(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            entities.append(
                Entity(
                    text=match.group(0),
                    type=EntityType.WEAPON,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return entities
