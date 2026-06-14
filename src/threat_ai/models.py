from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class SourceType(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    CALL = "call"
    SOCIAL = "social"
    INCIDENT = "incident"


class Document(BaseModel):
    id: str
    source_type: SourceType
    source_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    text: str
    metadata: dict = {}


class SpeechSegment(BaseModel):
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    language: Optional[str] = None
    text: str


class EntityType(str, Enum):
    PERSON = "person"
    LOCATION = "location"
    DATE = "date"
    ORGANIZATION = "organization"
    PHONE = "phone"
    ADDRESS = "address"
    VEHICLE = "vehicle"
    WEAPON = "weapon"
    OTHER = "other"


class Entity(BaseModel):
    text: str
    type: EntityType
    start: int
    end: int
    source_id: Optional[str] = None
    confidence: Optional[float] = None


class EventType(str, Enum):
    PURCHASE = "purchase"
    MEETING = "meeting"
    TRAVEL = "travel"
    THREAT = "threat"
    PAYMENT = "payment"
    RECRUITMENT = "recruitment"
    OTHER = "other"


class Event(BaseModel):
    id: str
    type: EventType
    description: str
    timestamp: Optional[datetime] = None
    source_ids: List[str] = []
    entities: List[Entity] = []
    people: List[str] = []
    locations: List[str] = []
    dates: List[str] = []
    related_message_ids: List[str] = []


class RiskType(str, Enum):
    NORMAL = "normal"
    THREAT = "threat"
    HARASSMENT = "harassment"
    FRAUD = "fraud"
    SUSPICIOUS_MEETING = "suspicious_meeting"
    SELF_HARM = "self_harm"
    ORGANIZED_ACTIVITY = "organized_activity"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Alert(BaseModel):
    id: str
    level: RiskLevel
    risk_type: RiskType
    confidence: float
    summary: str
    reasons: List[str]
    source_ids: List[str]
    created_at: datetime
    explanation: Optional[str] = None


class KnowledgeFact(BaseModel):
    subject: str
    predicate: str
    object: str
    source_ids: List[str] = []
    confidence: Optional[float] = None
