"""Failure analysis: trace injected scenarios through the pipeline."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .models import Document


def analyze_scenario_injection(documents: List[Document], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Check if injected scenarios were actually generated."""
    injected_by_scenario = defaultdict(list)
    injected_ids = set()

    for doc_id, truth in ground_truth.items():
        scenario = truth.get("scenario")
        if scenario:
            injected_by_scenario[scenario].append(doc_id)
            injected_ids.add(doc_id)

    scenario_counts = {scenario: len(docs) for scenario, docs in injected_by_scenario.items()}

    # check if they're in the documents
    doc_ids = {doc.id for doc in documents}
    injected_in_docs = injected_ids & doc_ids
    missing = injected_ids - doc_ids

    return {
        "scenario_counts": scenario_counts,
        "total_injected": len(injected_ids),
        "injected_in_documents": len(injected_in_docs),
        "missing_from_documents": len(missing),
        "missing_ids": list(missing)[:10],
        "by_scenario": {scenario: len(docs) for scenario, docs in injected_by_scenario.items()},
    }


def analyze_event_creation(pipeline_results: Dict[str, Any], documents: List[Document], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Trace: injected documents → events."""
    events = pipeline_results.get("events", [])
    doc_to_event = {}

    for event in events:
        for doc_id in event.get("related_message_ids", []):
            doc_to_event[doc_id] = event.get("id")

    # check if injected scenario docs made it into events
    injected_docs = {doc_id for doc_id, truth in ground_truth.items() if truth.get("scenario")}
    injected_in_events = injected_docs & set(doc_to_event.keys())

    # check if those events were flagged high-risk
    event_alerts = pipeline_results.get("event_alerts", [])
    alert_by_event_id = {}
    for ea in event_alerts:
        # ea["id"] is like "event-alert-event-1", extract event_id
        alert_id = ea.get("id", "")
        if alert_id.startswith("event-alert-"):
            event_id = alert_id.replace("event-alert-", "")
            alert_by_event_id[event_id] = ea

    injected_event_risks = {}
    for doc_id in injected_in_events:
        event_id = doc_to_event[doc_id]
        alert = alert_by_event_id.get(event_id)
        if alert:
            injected_event_risks[event_id] = {
                "confidence": alert.get("confidence"),
                "level": alert.get("level"),
                "reasons": alert.get("reasons"),
            }

    return {
        "total_events": len(events),
        "total_alerts": len(event_alerts),
        "injected_docs": len(injected_docs),
        "injected_docs_in_events": len(injected_in_events),
        "injected_docs_lost": len(injected_docs - set(doc_to_event.keys())),
        "injected_event_ids": list(set(doc_to_event[d] for d in injected_in_events if d in doc_to_event)),
        "injected_event_risks": injected_event_risks,
    }


def analyze_risk_distribution(pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
    """Check if predictions collapsed (all similar) or spread."""
    event_alerts = pipeline_results.get("event_alerts", [])
    confidences = [ea.get("confidence", 0.0) for ea in event_alerts]
    decisions = pipeline_results.get("decisions", {})
    decision_actions = [d.get("action") for d in decisions.values()]

    if not confidences:
        return {"status": "no_alerts"}

    return {
        "alert_count": len(confidences),
        "confidence_min": min(confidences),
        "confidence_max": max(confidences),
        "confidence_mean": sum(confidences) / len(confidences),
        "confidence_stddev": (sum((x - sum(confidences) / len(confidences)) ** 2 for x in confidences) / len(confidences)) ** 0.5,
        "confidence_quartiles": sorted(confidences)[len(confidences) // 4 :: len(confidences) // 4],
        "decision_distribution": dict(Counter(decision_actions)),
    }


def analyze_predictions_vs_thresholds(pipeline_results: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """Check if thresholds suppress detections."""
    event_alerts = pipeline_results.get("event_alerts", [])
    confidences = [ea.get("confidence", 0.0) for ea in event_alerts]
    decisions = pipeline_results.get("decisions", {})

    above_threshold = [c for c in confidences if c >= threshold]
    below_threshold = [c for c in confidences if c < threshold]

    escalate_actions = sum(1 for d in decisions.values() if d.get("action") in ("ESCALATE", "PRIORITY_ALERT"))

    return {
        "threshold": threshold,
        "above_threshold": len(above_threshold),
        "below_threshold": len(below_threshold),
        "escalate_decisions": escalate_actions,
        "monitor_decisions": sum(1 for d in decisions.values() if d.get("action") == "MONITOR"),
        "ignore_decisions": sum(1 for d in decisions.values() if d.get("action") == "IGNORE"),
    }


def trace_injected_scenario_signal(documents: List[Document], ground_truth: Dict[str, Any], pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detailed trace of each injected scenario through the pipeline."""
    traces = []
    doc_to_event = {}

    for event in pipeline_results.get("events", []):
        for doc_id in event.get("related_message_ids", []):
            doc_to_event[doc_id] = event.get("id")

    # Build alert_by_event_id map
    event_alerts = pipeline_results.get("event_alerts", [])
    alert_by_event_id = {}
    for ea in event_alerts:
        alert_id = ea.get("id", "")
        if alert_id.startswith("event-alert-"):
            event_id = alert_id.replace("event-alert-", "")
            alert_by_event_id[event_id] = ea

    decisions = pipeline_results.get("decisions", {})

    for doc in documents:
        truth = ground_truth.get(doc.id)
        if not truth or not truth.get("scenario"):
            continue

        event_id = doc_to_event.get(doc.id)
        alert = alert_by_event_id.get(event_id) if event_id else None
        decision = decisions.get(event_id) if event_id else None

        trace = {
            "doc_id": doc.id,
            "scenario": truth.get("scenario"),
            "text": doc.text[:80],
            "doc_created": True,
            "event_created": event_id is not None,
            "event_id": event_id,
            "alert_confidence": alert.get("confidence") if alert else None,
            "alert_level": alert.get("level") if alert else None,
            "decision_action": decision.get("action") if decision else None,
            "detected_as_positive": decision.get("action") in ("ESCALATE", "PRIORITY_ALERT") if decision else False,
        }
        traces.append(trace)

    return traces


def run_failure_analysis(persist_db_path: str = "failure_analysis.db", num_people: int = 200, days: int = 2, events_per_day: int = 50):
    """Full failure analysis on a deployment simulation."""
    import os

    from .deployment_runner import run_continuous_deployment
    from .world_sim import build_city_world

    if os.path.exists(persist_db_path):
        os.remove(persist_db_path)

    city = build_city_world.__code__

    documents, ground_truth, city = build_city_world(
        num_people=num_people,
        days=days,
        events_per_day=events_per_day,
        hidden_scenarios=["fraud_ring", "harassment_campaign", "coordinated_activity", "escalating_threat"],
    )

    from .pipeline import ThreatPipeline

    pipeline = ThreatPipeline()
    result = pipeline.run(documents, persist_db_path=persist_db_path)

    print("=== SCENARIO INJECTION ANALYSIS ===")
    inj = analyze_scenario_injection(documents, ground_truth)
    print(f"Total injected scenarios: {inj['total_injected']}")
    print(f"Injected in documents: {inj['injected_in_documents']}")
    print(f"Missing from documents: {inj['missing_from_documents']}")
    print(f"By scenario type: {inj['by_scenario']}")

    print("\n=== EVENT CREATION TRACE ===")
    evt = analyze_event_creation(result, documents, ground_truth)
    print(f"Total events created: {evt['total_events']}")
    print(f"Injected docs in events: {evt['injected_docs_in_events']}")
    print(f"Injected docs lost: {evt['injected_docs_lost']}")
    print(f"Injected event IDs: {evt['injected_event_ids'][:5]}")
    print(f"Injected event risks (top 5): {dict(list(evt['injected_event_risks'].items())[:5])}")

    print("\n=== RISK DISTRIBUTION ===")
    risk = analyze_risk_distribution(result)
    if risk.get("status") != "no_alerts":
        print(f"Alert count: {risk['alert_count']}")
        print(f"Confidence range: [{risk['confidence_min']:.3f}, {risk['confidence_max']:.3f}]")
        print(f"Confidence mean: {risk['confidence_mean']:.3f}")
        print(f"Confidence stddev: {risk['confidence_stddev']:.3f}")
        print(f"Decision distribution: {risk['decision_distribution']}")
    else:
        print("No alerts generated")

    print("\n=== THRESHOLD ANALYSIS ===")
    thresh = analyze_predictions_vs_thresholds(result, threshold=0.5)
    print(f"Threshold: {thresh['threshold']}")
    print(f"Above threshold: {thresh['above_threshold']}")
    print(f"Below threshold: {thresh['below_threshold']}")
    print(f"Escalate decisions: {thresh['escalate_decisions']}")
    print(f"Monitor decisions: {thresh['monitor_decisions']}")

    print("\n=== INJECTED SCENARIO SIGNAL TRACES ===")
    traces = trace_injected_scenario_signal(documents, ground_truth, result)
    fraud_traces = [t for t in traces if t["scenario"] == "fraud_ring"]
    harassment_traces = [t for t in traces if t["scenario"] == "harassment_campaign"]
    coordinated_traces = [t for t in traces if t["scenario"] == "coordinated_activity"]
    threat_traces = [t for t in traces if t["scenario"] == "escalating_threat"]

    print(f"\nFraud rings ({len(fraud_traces)} docs):")
    for t in fraud_traces[:3]:
        print(f"  Doc: {t['doc_id']}, Text: {t['text']}, Event: {t['event_id']}, Conf: {t['alert_confidence']}, Decision: {t['decision_action']}, Detected: {t['detected_as_positive']}")

    print(f"\nHarassment campaigns ({len(harassment_traces)} docs):")
    for t in harassment_traces[:3]:
        print(f"  Doc: {t['doc_id']}, Text: {t['text']}, Event: {t['event_id']}, Conf: {t['alert_confidence']}, Decision: {t['decision_action']}, Detected: {t['detected_as_positive']}")

    print(f"\nCoordinated activities ({len(coordinated_traces)} docs):")
    for t in coordinated_traces[:3]:
        print(f"  Doc: {t['doc_id']}, Text: {t['text']}, Event: {t['event_id']}, Conf: {t['alert_confidence']}, Decision: {t['decision_action']}, Detected: {t['detected_as_positive']}")

    print(f"\nEscalating threats ({len(threat_traces)} docs):")
    for t in threat_traces[:3]:
        print(f"  Doc: {t['doc_id']}, Text: {t['text']}, Event: {t['event_id']}, Conf: {t['alert_confidence']}, Decision: {t['decision_action']}, Detected: {t['detected_as_positive']}")

    return {
        "scenario_injection": inj,
        "event_creation": evt,
        "risk_distribution": risk,
        "threshold_analysis": thresh,
        "traces": traces,
    }


if __name__ == "__main__":
    analysis = run_failure_analysis(num_people=200, days=2, events_per_day=50)
