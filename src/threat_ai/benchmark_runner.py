import sys
from datetime import datetime

from .benchmark import RuleBasedBaseline, MLBaseline, FullIntelligenceSystem, compare_systems
from .world_sim import build_world, generate_missing_message_scenario, reassign_entities
from .emergent import find_rapidly_forming_groups, find_location_hubs, find_unusual_connection_growth, find_repeated_low_risk_patterns
from .scenario import prioritize_scenarios
from .models import Document


def print_metrics(results):
    for name, metrics in results.items():
        print(f"=== {name} ===")
        print(f"precision: {metrics.precision}")
        print(f"recall: {metrics.recall}")
        print(f"false_positives: {metrics.false_positives}")
        print(f"false_positive_rate: {metrics.false_positive_rate}")
        print(f"detection_lead_seconds: {metrics.detection_lead_seconds}")
        print(f"workload_reduction: {metrics.workload_reduction}")
        print(f"total_events: {metrics.total_events}")
        print(f"predicted_positive: {metrics.predicted_positive}")
        print()


def run_small_benchmark():
    # generate a modest synthetic world for benchmarking
    documents, ground_truth = build_world(num_people=200, num_events=400, escalations=0.2)
    train_docs = documents[:200]
    train_labels = [1 if ground_truth[doc.id]["escalated"] else 0 for doc in train_docs]
    ml = MLBaseline()
    ml.train(train_docs, train_labels)

    systems = {
        "rule_baseline": RuleBasedBaseline(threshold=0.5),
        "ml_baseline": ml,
        "full_system": FullIntelligenceSystem(),
    }

    print("Running benchmark on base synthetic world...")
    results = compare_systems(systems, documents, ground_truth)
    print_metrics(results)

    print("Running drift and missing data scenarios...")
    missing_docs = generate_missing_message_scenario(documents, missing_rate=0.3)
    missing_truth = {doc.id: ground_truth[doc.id] for doc in missing_docs}
    results_missing = compare_systems(systems, missing_docs, missing_truth)
    print("--- Missing message scenario ---")
    print_metrics(results_missing)

    aliased_docs = reassign_entities(documents, change_rate=0.2)
    aliased_truth = ground_truth
    results_aliased = compare_systems(systems, aliased_docs, aliased_truth)
    print("--- Entity alias scenario ---")
    print_metrics(results_aliased)

    # emergent pattern analysis uses the full system event clustering
    full = FullIntelligenceSystem()
    full_results = full.pipeline.run(documents)
    events = full_results.get("events", [])
    event_alerts = {ea["id"]: ea for ea in full_results.get("event_alerts", [])}
    print("Emergent pattern samples:")
    print("- repeated low-risk patterns:", find_repeated_low_risk_patterns(events, event_alerts, min_repeat=4)[:5])

    scenarios = [
        {"probability": 0.2, "impact": 0.9, "time_to_escalation": 12},
        {"probability": 0.6, "impact": 0.4, "time_to_escalation": 24},
        {"probability": 0.35, "impact": 0.7, "time_to_escalation": 8},
    ]
    ranked = prioritize_scenarios(scenarios)
    print("Ranked future scenarios:")
    for s in ranked:
        print(s)

    print("Generated missing and alias scenarios for scale testing.")


if __name__ == "__main__":
    run_small_benchmark()
