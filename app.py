import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from threat_ai.graph import KnowledgeGraph
from threat_ai.ingest import ingest_chat, ingest_email, ingest_incident, ingest_social_post
from threat_ai.models import Document
from threat_ai.pipeline import ThreatPipeline
from threat_ai.query import answer_natural_language_query

app = FastAPI(title="Threat Detection AI")


class IngestPayload(BaseModel):
    documents: list[dict]


class QueryPayload(BaseModel):
    query: str


@app.post("/process")
def process_documents(payload: IngestPayload):
    documents = []
    for item in payload.documents:
        source_type = item.get("source_type")
        if source_type == "email":
            documents.append(ingest_email(item))
        elif source_type == "chat":
            documents.append(ingest_chat(item))
        elif source_type == "incident":
            documents.append(ingest_incident(item))
        elif source_type == "social":
            documents.append(ingest_social_post(item))
        else:
            documents.append(Document(**item))

    pipeline = ThreatPipeline()
    return pipeline.run(documents)


@app.post("/query")
def query_graph(payload: QueryPayload):
    pipeline = ThreatPipeline()
    graph = KnowledgeGraph()
    return answer_natural_language_query(payload.query, graph, [])


@app.get("/")
def root():
    return {"status": "Threat Detection AI is ready"}
