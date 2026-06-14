import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .digital_twin import generate_beliefs_from_graph, generate_competing_hypotheses
from .analysis_memory import remember_investigation
from .agentic_investigator import investigate_question
from .graph import KnowledgeGraph
from .models import Document
from .pipeline import ThreatPipeline
from .storage import init_db, store_system_metric, store_belief, store_hypothesis
from .world_sim import build_city_world


def run_continuous_deployment(
    persist_db_path: str = "deployment_env.db",
    num_people: int = 1000,
    days: int = 7,
    events_per_day: int = 150,
    batch_hours: int = 6,
    hidden_scenarios: Optional[List[str]] = None,
    scale_mode: str = "medium",
):
    if hidden_scenarios is None:
        hidden_scenarios = ["fraud_ring", "harassment_campaign", "coordinated_activity", "escalating_threat"]

    # generate a synthetic city and scenario ground truth
    documents, ground_truth, city = build_city_world(
        num_people=num_people,
        days=days,
        events_per_day=events_per_day,
        hidden_scenarios=hidden_scenarios,
    )

    if os.path.exists(persist_db_path):
        os.remove(persist_db_path)
    conn = init_db(persist_db_path)

    pipeline = ThreatPipeline()
    pipeline.graph = KnowledgeGraph()
    now = min(doc.timestamp for doc in documents)
    end_time = max(doc.timestamp for doc in documents)
    current_time = now

    timeline_documents = sorted(documents, key=lambda d: d.timestamp or datetime.min)
    metrics_samples = []
    memory_conn = conn

    while current_time <= end_time:
        window_end = current_time + timedelta(hours=batch_hours)
        window_docs = [doc for doc in timeline_documents if doc.timestamp and current_time <= doc.timestamp < window_end]
        if window_docs:
            result = pipeline.run(window_docs, persist_db_path=persist_db_path)
            beliefs = generate_beliefs_from_graph(pipeline.graph, [e for e in result.get("events", [])], {ea["id"]: ea for ea in result.get("event_alerts", [])})
            hypotheses = generate_competing_hypotheses(pipeline.graph, beliefs, [e for e in result.get("events", [])])
            for belief in beliefs:
                store_belief(conn, belief)
            for hyp in hypotheses:
                store_hypothesis(conn, hyp)

            event_docs = {}
            for event in result.get("events", []):
                for message_id in event.get("related_message_ids", []):
                    event_docs[message_id] = event.get("id")

            for doc in window_docs:
                event_id = event_docs.get(doc.id)
                pred = result.get("decisions", {}).get(event_id, {"action": "MONITOR"})
                metrics_samples.append((doc.id, pred["action"], ground_truth.get(doc.id, {})))

        current_time = window_end

    evaluation = evaluate_continuous_metrics(metrics_samples)
    store_system_metric(conn, "detection_lead_seconds", evaluation.get("average_lead", 0.0))
    store_system_metric(conn, "false_alarm_rate", evaluation.get("false_alarm_rate", 0.0))
    store_system_metric(conn, "miss_rate", evaluation.get("miss_rate", 0.0))

    # analytical memory record
    remember_investigation(
        memory_conn,
        investigation_id="deployment-1",
        summary="Continuous city deployment evaluation",
        pattern="synthetic hidden scenario detection",
        outcome="completed",
        tags=["deployment", "synthetic city", "hidden scenarios"],
        note=f"Detected {evaluation.get('true_positives')} positive signals and missed {evaluation.get('misses')} actual escalations.",
    )

    report = investigate_question(
        "Why is risk increasing around the most connected people?",
        {"beliefs": beliefs, "events": [e for e in result.get("events", [])]},
        pipeline.graph,
        conn=memory_conn,
    )

    return {
        "evaluation": evaluation,
        "report": report,
        "city": city,
        "hidden_scenarios": hidden_scenarios,
    }


def evaluate_continuous_metrics(samples: List[tuple]) -> Dict[str, float]:
    tp = fp = fn = 0
    leads = []
    for doc_id, action, truth in samples:
        positive = action not in ("IGNORE", "MONITOR", "negative")
        actual = truth.get("escalated", False)
        if positive and actual:
            tp += 1
            if truth.get("escalation_time") and truth.get("timestamp"):
                leads.append((truth["escalation_time"] - truth["timestamp"]).total_seconds())
        if positive and not actual:
            fp += 1
        if not positive and actual:
            fn += 1
    total = len(samples)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "misses": fn,
        "false_alarm_rate": fp / total if total else 0.0,
        "miss_rate": fn / total if total else 0.0,
        "average_lead": sum(leads) / len(leads) if leads else 0.0,
    }


if __name__ == "__main__":
    summary = run_continuous_deployment(num_people=2000, days=5, events_per_day=120, batch_hours=8)
    print("Deployment evaluation:")
    print(summary["evaluation"])
    print("Agentic investigator report:")
    print(summary["report"])
