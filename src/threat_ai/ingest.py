from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Document, SourceType


def normalize_text(text: str) -> str:
    return text.strip()


def ingest_email(email: Dict[str, Any]) -> Document:
    return Document(
        id=email.get("id", "email-unknown"),
        source_type=SourceType.EMAIL,
        source_name=email.get("from"),
        timestamp=email.get("timestamp", datetime.utcnow()),
        text=normalize_text(email.get("body", "")),
        metadata={"subject": email.get("subject")},
    )


def ingest_chat(chat: Dict[str, Any]) -> Document:
    return Document(
        id=chat.get("id", "chat-unknown"),
        source_type=SourceType.CHAT,
        source_name=chat.get("channel"),
        timestamp=chat.get("timestamp", datetime.utcnow()),
        text=normalize_text(chat.get("message", "")),
        metadata={"sender": chat.get("sender")},
    )


def ingest_incident(incident: Dict[str, Any]) -> Document:
    return Document(
        id=incident.get("id", "incident-unknown"),
        source_type=SourceType.INCIDENT,
        source_name=incident.get("source"),
        timestamp=incident.get("timestamp", datetime.utcnow()),
        text=normalize_text(incident.get("description", "")),
        metadata={"severity": incident.get("severity")},
    )


def ingest_social_post(post: Dict[str, Any]) -> Document:
    return Document(
        id=post.get("id", "social-unknown"),
        source_type=SourceType.SOCIAL,
        source_name=post.get("platform"),
        timestamp=post.get("timestamp", datetime.utcnow()),
        text=normalize_text(post.get("text", "")),
        metadata={"author": post.get("author")},
    )


def ingest_call_transcript(transcript: Dict[str, Any]) -> Document:
    return Document(
        id=transcript.get("id", "call-unknown"),
        source_type=SourceType.CALL,
        source_name=transcript.get("caller"),
        timestamp=transcript.get("timestamp", datetime.utcnow()),
        text=normalize_text(transcript.get("transcript", "")),
        metadata={"duration": transcript.get("duration")},
    )


def unify_documents(documents: List[Document]) -> List[Document]:
    return documents
