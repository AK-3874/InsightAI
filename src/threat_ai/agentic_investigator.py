from typing import Dict, List, Optional

from .analysis_memory import retrieve_similar_memories
from .digital_twin import generate_competing_hypotheses
from .graph import KnowledgeGraph


def investigate_question(question: str, pipeline_results: Dict[str, any], graph: KnowledgeGraph, conn=None) -> Dict[str, any]:
    lower = question.lower()
    report = {
        "question": question,
        "summary": None,
        "observations": [],
        "hypotheses": [],
        "analogue_memories": [],
    }

    anomalies = graph.basic_anomaly_detection()
    top_people = [person for person, _ in graph.top_connected_people(5)]
    escalation_chains = graph.find_escalation_chains(min_length=1)

    if "why" in lower and "communication" in lower:
        report["summary"] = "Communication volume appears elevated in a small group of tightly connected people."
        report["observations"].append(f"Anomalous high-degree nodes: {anomalies}")
        report["observations"].append(f"Top connected people: {', '.join(top_people)}")
        report["observations"].append(f"Escalation chains detected: {escalation_chains}")
    elif "why" in lower and "connected" in lower:
        report["summary"] = "The graph shows repeated shared events between the same participants, suggesting emerging clusters."
        report["observations"].append(f"Top people: {', '.join(top_people)}")
        report["observations"].append(f"Anomaly candidates: {anomalies}")
    elif "unknown entity" in lower:
        report["summary"] = "A likely unknown entity is a common hub linking several high-risk events."
        report["observations"].append(f"Potential hubs: {anomalies}")
        report["observations"].append(f"Escalation chains: {escalation_chains}")
    else:
        report["summary"] = "The system has identified risk clusters, escalation paths, and high-confidence relationships to explore."
        report["observations"].append(f"Top people: {', '.join(top_people)}")
        report["observations"].append(f"Anomalies: {anomalies}")
        report["observations"].append(f"Escalation chains: {escalation_chains}")

    beliefs = pipeline_results.get("beliefs") or []
    report["hypotheses"] = generate_competing_hypotheses(graph, beliefs, pipeline_results.get("events", []))

    if conn is not None:
        report["analogue_memories"] = retrieve_similar_memories(conn, keywords=[w for w in lower.split() if len(w) > 4])[:3]

    return report
