"""Reproducible evidence harness for synthetic threat detection trials.

The harness keeps labels outside the detector inputs, runs several systems over
the same challenge set, and produces scenario-level scorecards plus artifacts
that can be inspected or visualized.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Document, SourceType
from .risk import detect_risk_indicators, score_risk


SCENARIO_MIX = {
    "normal": 0.30,
    "fraud": 0.20,
    "harassment": 0.20,
    "escalating_threat": 0.20,
    "coordinated_group": 0.10,
}

DISPLAY_NAMES = {
    "normal": "Normal",
    "fraud": "Fraud",
    "harassment": "Harassment",
    "escalating_threat": "Threat",
    "coordinated_group": "Coordinated Group",
}

SOURCE_TYPES = [SourceType.CHAT, SourceType.EMAIL, SourceType.SOCIAL, SourceType.INCIDENT]

LOCATIONS = [
    "Harbor Warehouse",
    "Metro Station",
    "North Office",
    "Market Annex",
    "Airport Hotel",
    "Old Factory",
    "Central Mall",
    "Depot 14",
    "Civic Center",
    "Training Hall",
]

ORGANIZATIONS = [
    "Northstar Logistics",
    "Kestrel Services",
    "Brightline Partners",
    "Atlas Accounts",
    "Silver Channel",
    "Civic Volunteer Hub",
    "Eastway Group",
]

CLASSIFIER_PROFILES = {
    "fraud": {
        "invoice": 0.14,
        "payment": 0.16,
        "account": 0.16,
        "verify": 0.18,
        "identity": 0.16,
        "wire": 0.18,
        "transfer": 0.16,
        "password": 0.20,
        "security code": 0.22,
        "authentication": 0.18,
        "routing": 0.14,
        "urgent": 0.12,
        "processing fee": 0.16,
    },
    "harassment": {
        "pressure": 0.15,
        "target": 0.12,
        "nobody": 0.12,
        "stop talking": 0.18,
        "quit": 0.18,
        "useless": 0.22,
        "worthless": 0.24,
        "loser": 0.22,
        "shut up": 0.22,
        "laughs": 0.16,
        "burden": 0.18,
        "does not belong": 0.20,
        "keep isolating": 0.20,
        "left out": 0.16,
        "not a fit": 0.18,
        "stay quiet": 0.16,
        "step away": 0.16,
        "isolating": 0.18,
    },
    "escalating_threat": {
        "warning": 0.16,
        "final warning": 0.22,
        "regret": 0.22,
        "pay the price": 0.24,
        "trouble": 0.16,
        "consequences": 0.18,
        "settle this": 0.14,
        "dangerous": 0.16,
        "do not ignore": 0.18,
        "last reminder": 0.14,
        "weather changes": 0.14,
    },
    "coordinated_group": {
        "coordinate": 0.18,
        "gather": 0.14,
        "team": 0.10,
        "leaders": 0.12,
        "assign responsibilities": 0.18,
        "back entrance": 0.16,
        "confidential": 0.14,
        "same group": 0.16,
        "secret location": 0.20,
        "next action": 0.16,
        "handoff": 0.12,
    },
}

NEGATIVE_TERMS = {
    "warning",
    "regret",
    "trouble",
    "consequences",
    "pressure",
    "worthless",
    "loser",
    "dangerous",
    "quiet",
    "confidential",
}

CODED_SEQUENCE_TERMS = {
    "package",
    "warehouse",
    "tomorrow",
    "handoff",
    "weather changes",
    "last reminder",
    "access note",
    "delivery route",
}


@dataclass(frozen=True)
class ChallengeConfig:
    seed: int = 12345
    scenario_count: int = 1000
    population: int = 10_000
    start_at: datetime = datetime(2026, 1, 1, 9, 0, 0)


@dataclass
class Scenario:
    id: str
    scenario_type: str
    documents: List[Document]
    participants: List[str]
    location: str
    start_at: datetime
    obvious_at: Optional[datetime]
    escalation_at: Optional[datetime]

    @property
    def positive(self) -> bool:
        return self.scenario_type != "normal"


@dataclass
class ChallengeSet:
    seed: int
    population: int
    scenarios: List[Scenario]
    ground_truth: Dict[str, Dict[str, Any]]

    @property
    def documents(self) -> List[Document]:
        docs: List[Document] = []
        for scenario in self.scenarios:
            docs.extend(scenario.documents)
        return sorted(docs, key=lambda doc: doc.timestamp or datetime.min)


@dataclass
class ScenarioPrediction:
    scenario_id: str
    system_name: str
    score: float
    risk_type: str
    detected_at: Optional[datetime]
    reasons: List[str] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SystemMetrics:
    name: str
    precision: float
    recall: float
    false_positives: int
    false_negatives: int
    true_positives: int
    predicted_positive: int
    total_scenarios: int
    average_lead_days: float
    early_detections: int
    recall_by_type: Dict[str, Dict[str, Any]]


def build_ground_truth_challenge(config: ChallengeConfig) -> ChallengeSet:
    """Create a deterministic scenario set and separate ground-truth labels."""

    if config.scenario_count <= 0:
        raise ValueError("scenario_count must be positive")

    rng = random.Random(config.seed)
    counts = _scaled_counts(config.scenario_count)
    scenario_types: List[str] = []
    for scenario_type, count in counts.items():
        scenario_types.extend([scenario_type] * count)
    rng.shuffle(scenario_types)

    population = [f"Person_{i:05d}" for i in range(1, config.population + 1)]
    scenarios: List[Scenario] = []
    ground_truth: Dict[str, Dict[str, Any]] = {}

    for index, scenario_type in enumerate(scenario_types, start=1):
        scenario_id = f"scenario-{index:04d}"
        participants = _choose_participants(rng, population, scenario_type)
        location = rng.choice(LOCATIONS)
        organization = rng.choice(ORGANIZATIONS)
        start_at = config.start_at + timedelta(hours=index * 3, minutes=rng.randint(0, 59))

        documents, obvious_at, escalation_at = _build_scenario_documents(
            rng=rng,
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            participants=participants,
            location=location,
            organization=organization,
            start_at=start_at,
        )

        scenario = Scenario(
            id=scenario_id,
            scenario_type=scenario_type,
            documents=documents,
            participants=participants,
            location=location,
            start_at=start_at,
            obvious_at=obvious_at,
            escalation_at=escalation_at,
        )
        scenarios.append(scenario)
        ground_truth[scenario_id] = {
            "scenario_type": scenario_type,
            "display_name": DISPLAY_NAMES[scenario_type],
            "positive": scenario.positive,
            "obvious_at": obvious_at,
            "escalation_at": escalation_at,
            "participants": participants,
            "participant_count": len(participants),
            "location": location,
            "document_ids": [doc.id for doc in documents],
        }

    scenarios.sort(key=lambda item: item.start_at)
    return ChallengeSet(
        seed=config.seed,
        population=config.population,
        scenarios=scenarios,
        ground_truth=ground_truth,
    )


class KeywordRulesDetector:
    name = "Keyword Rules"

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def predict(self, scenarios: Sequence[Scenario]) -> Dict[str, ScenarioPrediction]:
        predictions: Dict[str, ScenarioPrediction] = {}
        for scenario in scenarios:
            best_score = 0.0
            best_type = "normal"
            detected_at: Optional[datetime] = None
            reasons: List[str] = []
            trajectory = []

            for doc in sorted(scenario.documents, key=lambda item: item.timestamp or datetime.min):
                risk_type, doc_reasons = detect_risk_indicators(doc.text)
                score = min(0.99, 0.35 * len(doc_reasons))
                trajectory.append(
                    {
                        "timestamp": doc.timestamp,
                        "score": score,
                        "source_id": doc.id,
                        "signals": doc_reasons,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_type = risk_type.value
                    reasons = doc_reasons[:]
                if detected_at is None and score >= self.threshold:
                    detected_at = doc.timestamp

            predictions[scenario.id] = ScenarioPrediction(
                scenario_id=scenario.id,
                system_name=self.name,
                score=best_score,
                risk_type=best_type,
                detected_at=detected_at,
                reasons=reasons,
                trajectory=trajectory,
            )
        return predictions


class BasicClassifierDetector:
    name = "Basic Classifier"

    def __init__(self, threshold: float = 0.62):
        self.threshold = threshold

    def predict(self, scenarios: Sequence[Scenario]) -> Dict[str, ScenarioPrediction]:
        predictions: Dict[str, ScenarioPrediction] = {}
        for scenario in scenarios:
            detected_at: Optional[datetime] = None
            best_score = 0.0
            best_type = "normal"
            reasons: List[str] = []
            trajectory: List[Dict[str, Any]] = []
            rolling_scores: List[float] = []

            for doc in sorted(scenario.documents, key=lambda item: item.timestamp or datetime.min):
                risk_type, score, doc_reasons = _classifier_score(doc.text)
                rolling_scores.append(score)
                rolling_avg = sum(rolling_scores[-3:]) / min(3, len(rolling_scores))
                scenario_score = min(0.99, max(score, rolling_avg + 0.06 * max(0, len(rolling_scores) - 2)))
                trajectory.append(
                    {
                        "timestamp": doc.timestamp,
                        "score": scenario_score,
                        "source_id": doc.id,
                        "signals": doc_reasons,
                    }
                )

                if scenario_score > best_score:
                    best_score = scenario_score
                    best_type = risk_type
                    reasons = doc_reasons[:]
                if detected_at is None and scenario_score >= self.threshold:
                    detected_at = doc.timestamp

            predictions[scenario.id] = ScenarioPrediction(
                scenario_id=scenario.id,
                system_name=self.name,
                score=best_score,
                risk_type=best_type,
                detected_at=detected_at,
                reasons=reasons,
                trajectory=trajectory,
            )
        return predictions


class FullPlatformDetector:
    """Fusion detector using risk scoring plus temporal and network signals."""

    name = "Full Platform"

    def __init__(self, threshold: float = 0.67):
        self.threshold = threshold

    def predict(self, scenarios: Sequence[Scenario]) -> Dict[str, ScenarioPrediction]:
        predictions: Dict[str, ScenarioPrediction] = {}
        for scenario in scenarios:
            detected_at: Optional[datetime] = None
            best_score = 0.0
            best_type = "normal"
            best_reasons: List[str] = []
            trajectory: List[Dict[str, Any]] = []
            prefix_docs: List[Document] = []
            classifier_scores: List[float] = []

            for doc in sorted(scenario.documents, key=lambda item: item.timestamp or datetime.min):
                prefix_docs.append(doc)
                direct_alert = score_risk(doc, [])
                classifier_type, classifier_score, classifier_reasons = _classifier_score(doc.text)
                classifier_scores.append(classifier_score)

                temporal_score, temporal_reasons = _temporal_network_score(prefix_docs, scenario.participants)
                direct_score = float(direct_alert.confidence)
                rolling_classifier = max(classifier_scores[-3:] or [0.0])
                trend_bonus = _trend_bonus(classifier_scores)

                score = min(
                    0.99,
                    0.50 * direct_score
                    + 0.45 * rolling_classifier
                    + temporal_score
                    + trend_bonus,
                )
                risk_type = classifier_type if classifier_score >= direct_score else direct_alert.risk_type.value
                reasons = list(dict.fromkeys(direct_alert.reasons + classifier_reasons + temporal_reasons))
                if trend_bonus:
                    reasons.append("risk trajectory is rising")

                trajectory.append(
                    {
                        "timestamp": doc.timestamp,
                        "score": score,
                        "source_id": doc.id,
                        "signals": reasons,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_type = risk_type
                    best_reasons = reasons[:]
                if detected_at is None and score >= self.threshold:
                    detected_at = doc.timestamp

            predictions[scenario.id] = ScenarioPrediction(
                scenario_id=scenario.id,
                system_name=self.name,
                score=best_score,
                risk_type=best_type,
                detected_at=detected_at,
                reasons=best_reasons,
                trajectory=trajectory,
            )
        return predictions


def evaluate_predictions(
    challenge: ChallengeSet,
    predictions: Dict[str, ScenarioPrediction],
    system_name: str,
) -> SystemMetrics:
    tp = fp = fn = predicted_positive = 0
    lead_days: List[float] = []
    early_detections = 0
    by_type: Dict[str, Dict[str, Any]] = {}

    for scenario_type in SCENARIO_MIX:
        by_type[scenario_type] = {
            "scenario_type": DISPLAY_NAMES[scenario_type],
            "actual": 0,
            "detected": 0,
            "recall": None,
        }

    for scenario in challenge.scenarios:
        prediction = predictions.get(scenario.id)
        detected = bool(prediction and prediction.detected_at)
        by_type[scenario.scenario_type]["actual"] += 1

        if detected:
            predicted_positive += 1
            if scenario.positive:
                by_type[scenario.scenario_type]["detected"] += 1
                tp += 1
                if scenario.obvious_at and prediction and prediction.detected_at:
                    lead = (scenario.obvious_at - prediction.detected_at).total_seconds() / 86400.0
                    lead_days.append(lead)
                    if lead > 0:
                        early_detections += 1
            else:
                fp += 1
                by_type[scenario.scenario_type]["detected"] += 1
        elif scenario.positive:
            fn += 1

    for row in by_type.values():
        actual = row["actual"]
        row["recall"] = row["detected"] / actual if actual else None

    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    average_lead = sum(lead_days) / len(lead_days) if lead_days else 0.0

    return SystemMetrics(
        name=system_name,
        precision=precision,
        recall=recall,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        predicted_positive=predicted_positive,
        total_scenarios=len(challenge.scenarios),
        average_lead_days=average_lead,
        early_detections=early_detections,
        recall_by_type=by_type,
    )


def run_evidence_challenge(
    config: ChallengeConfig,
    output_dir: Optional[Path] = None,
    investigation_scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    challenge = build_ground_truth_challenge(config)
    detectors = [KeywordRulesDetector(), BasicClassifierDetector(), FullPlatformDetector()]

    predictions_by_system = {
        detector.name: detector.predict(challenge.scenarios)
        for detector in detectors
    }
    metrics_by_system = {
        name: evaluate_predictions(challenge, predictions, name)
        for name, predictions in predictions_by_system.items()
    }

    red_team_challenge = build_red_team_challenge(challenge, seed=config.seed + 404)
    red_team_predictions = FullPlatformDetector().predict(red_team_challenge.scenarios)
    red_team_metrics = evaluate_predictions(red_team_challenge, red_team_predictions, "Full Platform Red Team")

    full_predictions = predictions_by_system["Full Platform"]
    investigation = build_investigation_report(
        challenge,
        full_predictions,
        scenario_id=investigation_scenario_id,
    )
    visual_proof = build_visual_proof(challenge, full_predictions, investigation["scenario_id"])
    scorecard = build_scorecard(
        challenge,
        metrics_by_system["Full Platform"],
        metrics_by_system,
        red_team_metrics,
        predictions_by_system,
    )
    scorecard_text = render_scorecard_text(scorecard)

    result = {
        "config": {
            "seed": config.seed,
            "scenario_count": config.scenario_count,
            "population": config.population,
            "start_at": config.start_at,
        },
        "scorecard": scorecard,
        "scorecard_text": scorecard_text,
        "system_metrics": metrics_by_system,
        "red_team_metrics": red_team_metrics,
        "investigation_report": investigation,
        "visual_proof": visual_proof,
        "challenge": challenge,
        "predictions": predictions_by_system,
    }

    if output_dir:
        write_evidence_artifacts(output_dir, result)

    return result


def build_red_team_challenge(challenge: ChallengeSet, seed: int) -> ChallengeSet:
    rng = random.Random(seed)
    scenarios: List[Scenario] = []

    for scenario in challenge.scenarios:
        documents = []
        for doc in sorted(scenario.documents, key=lambda item: item.timestamp or datetime.min):
            text = doc.text
            if scenario.positive:
                text = _red_team_text(text)
            new_doc = _copy_document(doc, text=text, doc_id=f"rt-{doc.id}")
            documents.append(new_doc)

        if scenario.positive and len(documents) >= 4 and rng.random() < 0.75:
            target = documents[-2]
            words = target.text.split()
            if len(words) >= 6:
                midpoint = len(words) // 2
                first = _copy_document(target, text=" ".join(words[:midpoint]), doc_id=f"{target.id}-a")
                second = _copy_document(
                    target,
                    text=" ".join(words[midpoint:]),
                    doc_id=f"{target.id}-b",
                    timestamp=(target.timestamp or scenario.start_at) + timedelta(minutes=3),
                )
                documents = [doc for doc in documents if doc.id != target.id]
                documents.extend([first, second])
                documents.sort(key=lambda item: item.timestamp or datetime.min)

        scenarios.append(
            Scenario(
                id=scenario.id,
                scenario_type=scenario.scenario_type,
                documents=documents,
                participants=scenario.participants[:],
                location=scenario.location,
                start_at=scenario.start_at,
                obvious_at=scenario.obvious_at,
                escalation_at=scenario.escalation_at,
            )
        )

    return ChallengeSet(
        seed=challenge.seed,
        population=challenge.population,
        scenarios=scenarios,
        ground_truth=challenge.ground_truth,
    )


def build_investigation_report(
    challenge: ChallengeSet,
    predictions: Dict[str, ScenarioPrediction],
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    scenario = _select_investigation_scenario(challenge, predictions, scenario_id)
    prediction = predictions[scenario.id]
    top_location, location_count = _most_common_location(scenario.documents)
    first_half, second_half = _split_activity_counts(scenario.documents)
    frequency_increase = _percent_increase(first_half, second_half)
    sentiment_points = _risk_points(prediction)
    risk_start = sentiment_points[0]["score"] if sentiment_points else 0.0
    risk_end = sentiment_points[-1]["score"] if sentiment_points else 0.0
    top_evidence = [
        {
            "timestamp": doc.timestamp,
            "source_id": doc.id,
            "text": doc.text,
        }
        for doc in sorted(scenario.documents, key=lambda item: item.timestamp or datetime.min)[-3:]
    ]

    bullets = [
        f"{len(scenario.participants)} connected individuals in the scenario graph",
        f"communication frequency increased {frequency_increase:.0f}%",
        f"{location_count} repeated meetings or references at {top_location}",
        f"risk trajectory moved from {risk_start:.2f} to {risk_end:.2f}",
        "new high-risk cluster emerged" if len(scenario.participants) >= 8 else "high-risk interaction pattern emerged",
    ]
    if prediction.detected_at and scenario.obvious_at:
        lead = (scenario.obvious_at - prediction.detected_at).total_seconds() / 86400.0
        bullets.append(f"flagged {lead:.1f} days before the obvious injected escalation")

    return {
        "scenario_id": scenario.id,
        "actual_type": DISPLAY_NAMES[scenario.scenario_type],
        "detected_type": prediction.risk_type,
        "score": prediction.score,
        "detected_at": prediction.detected_at,
        "obvious_at": scenario.obvious_at,
        "bullets": bullets,
        "reasons": prediction.reasons[:8],
        "top_evidence": top_evidence,
    }


def build_visual_proof(
    challenge: ChallengeSet,
    predictions: Dict[str, ScenarioPrediction],
    scenario_id: str,
) -> Dict[str, Any]:
    scenario = next(item for item in challenge.scenarios if item.id == scenario_id)
    prediction = predictions[scenario.id]
    location_counter = Counter(_document_location(doc) for doc in scenario.documents)
    nodes = [{"id": person, "type": "person"} for person in scenario.participants]
    nodes.append({"id": scenario.location, "type": "location"})

    edges = []
    participants = scenario.participants
    for index, person in enumerate(participants):
        if index + 1 < len(participants):
            edges.append({"source": person, "target": participants[index + 1], "weight": 1})
        edges.append({"source": person, "target": scenario.location, "weight": location_counter[scenario.location]})

    timeline = [
        {
            "timestamp": doc.timestamp,
            "source_id": doc.id,
            "text": doc.text,
            "risk_score": _trajectory_score(prediction, doc.id),
            "injected_obvious_escalation": bool(scenario.obvious_at and doc.timestamp == scenario.obvious_at),
        }
        for doc in sorted(scenario.documents, key=lambda item: item.timestamp or datetime.min)
    ]

    return {
        "scenario_id": scenario.id,
        "network_graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "event_timeline": timeline,
        "risk_trajectory": _risk_points(prediction),
        "actual_injected_escalation_at": scenario.obvious_at,
    }


def build_scorecard(
    challenge: ChallengeSet,
    full_metrics: SystemMetrics,
    metrics_by_system: Dict[str, SystemMetrics],
    red_team_metrics: SystemMetrics,
    predictions_by_system: Dict[str, Dict[str, ScenarioPrediction]],
) -> Dict[str, Any]:
    injected = {
        DISPLAY_NAMES[key]: value["actual"]
        for key, value in full_metrics.recall_by_type.items()
        if key != "normal"
    }
    detected = {
        DISPLAY_NAMES[key]: value["detected"]
        for key, value in full_metrics.recall_by_type.items()
        if key != "normal"
    }

    full_predictions = predictions_by_system["Full Platform"]
    rule_predictions = predictions_by_system["Keyword Rules"]
    emergent_clusters = sum(
        1
        for scenario in challenge.scenarios
        if full_predictions[scenario.id].detected_at
        and any("cluster" in reason or "participant" in reason for reason in full_predictions[scenario.id].reasons)
    )
    unknown_groups = sum(
        1
        for scenario in challenge.scenarios
        if scenario.scenario_type == "coordinated_group"
        and full_predictions[scenario.id].detected_at
        and not rule_predictions[scenario.id].detected_at
    )

    return {
        "population": challenge.population,
        "scenario_seed": challenge.seed,
        "scenario_count": len(challenge.scenarios),
        "injected_scenarios": injected,
        "detected": detected,
        "precision": full_metrics.precision,
        "recall": full_metrics.recall,
        "false_positives": full_metrics.false_positives,
        "false_negatives": full_metrics.false_negatives,
        "average_lead_days": full_metrics.average_lead_days,
        "early_detections": full_metrics.early_detections,
        "emergent_clusters_found": emergent_clusters,
        "unknown_risk_groups_identified": unknown_groups,
        "recall_by_type": full_metrics.recall_by_type,
        "system_comparison": {
            name: {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "false_positives": metrics.false_positives,
                "average_lead_days": metrics.average_lead_days,
            }
            for name, metrics in metrics_by_system.items()
        },
        "red_team": {
            "full_platform_recall_before": full_metrics.recall,
            "full_platform_recall_after": red_team_metrics.recall,
            "full_platform_precision_after": red_team_metrics.precision,
            "false_positives_after": red_team_metrics.false_positives,
        },
    }


def render_scorecard_text(scorecard: Dict[str, Any]) -> str:
    lines = [
        f"Scenario Seed: {scorecard['scenario_seed']}",
        f"Population: {scorecard['population']:,}",
        "",
        "Injected Scenarios:",
    ]
    for label, count in scorecard["injected_scenarios"].items():
        lines.append(f"  {label}: {count}")

    lines.extend(["", "Detected:"])
    for label, count in scorecard["detected"].items():
        lines.append(f"  {label}: {count}")

    lines.extend(
        [
            "",
            f"Precision: {_pct(scorecard['precision'])}",
            f"Recall: {_pct(scorecard['recall'])}",
            f"False Positives: {scorecard['false_positives']}",
            f"False Negatives: {scorecard['false_negatives']}",
            f"Average Lead Time: {scorecard['average_lead_days']:.2f} days",
            f"Early Detections: {scorecard['early_detections']}",
            "",
            f"Emergent Clusters Found: {scorecard['emergent_clusters_found']}",
            f"Unknown Risk Groups Identified: {scorecard['unknown_risk_groups_identified']}",
            "",
            "Scenario Type Recall:",
            "| Scenario Type | Actual | Detected | Recall |",
            "| ------------- | ------ | -------- | ------ |",
        ]
    )
    for row in scorecard["recall_by_type"].values():
        lines.append(
            f"| {row['scenario_type']} | {row['actual']} | {row['detected']} | {_pct(row['recall'])} |"
        )

    lines.extend(["", "System Comparison:", "| System | Precision | Recall | Avg Lead |"])
    lines.append("| ------ | --------- | ------ | -------- |")
    for name, metrics in scorecard["system_comparison"].items():
        lines.append(
            f"| {name} | {_pct(metrics['precision'])} | {_pct(metrics['recall'])} | {metrics['average_lead_days']:.2f}d |"
        )

    red_team = scorecard["red_team"]
    lines.extend(
        [
            "",
            "Red Team Robustness:",
            f"  Detection before evasion: {_pct(red_team['full_platform_recall_before'])}",
            f"  Detection after evasion: {_pct(red_team['full_platform_recall_after'])}",
        ]
    )
    return "\n".join(lines)


def write_evidence_artifacts(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    challenge: ChallengeSet = result["challenge"]

    documents_path = output_dir / "challenge_documents.jsonl"
    with documents_path.open("w", encoding="utf-8") as handle:
        for doc in challenge.documents:
            handle.write(json.dumps(_document_to_public_dict(doc), default=_json_default) + "\n")

    _write_json(output_dir / "ground_truth_labels.json", challenge.ground_truth)
    _write_json(output_dir / "scorecard.json", result["scorecard"])
    (output_dir / "scorecard.txt").write_text(result["scorecard_text"], encoding="utf-8")
    _write_json(output_dir / "system_metrics.json", result["system_metrics"])
    _write_json(output_dir / "investigation_report.json", result["investigation_report"])
    _write_json(output_dir / "visual_proof.json", result["visual_proof"])

    report = _render_markdown_report(result)
    (output_dir / "evidence_report.md").write_text(report, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible synthetic evidence challenge.")
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic scenario seed.")
    parser.add_argument("--scenarios", type=int, default=1000, help="Number of synthetic scenarios.")
    parser.add_argument("--population", type=int, default=10_000, help="Synthetic city population size.")
    parser.add_argument("--output", type=Path, default=Path("evidence_output"), help="Artifact output directory.")
    parser.add_argument("--investigate", default=None, help="Optional scenario id for investigation report.")
    args = parser.parse_args(argv)

    config = ChallengeConfig(
        seed=args.seed,
        scenario_count=args.scenarios,
        population=args.population,
    )
    result = run_evidence_challenge(
        config=config,
        output_dir=args.output,
        investigation_scenario_id=args.investigate,
    )

    print(result["scorecard_text"])
    print()
    print(f"Artifacts written to: {args.output.resolve()}")
    return 0


def _build_scenario_documents(
    rng: random.Random,
    scenario_id: str,
    scenario_type: str,
    participants: List[str],
    location: str,
    organization: str,
    start_at: datetime,
) -> Tuple[List[Document], Optional[datetime], Optional[datetime]]:
    if scenario_type == "normal":
        texts = [
            f"{participants[0]} and {participants[1]} will review the quarterly report at {location}.",
            "The package arrived safely and the client signed the receipt.",
            "Please send the agenda before tomorrow's planning meeting.",
            f"{organization} confirmed the event schedule for next week.",
        ]
        obvious_at = None
        escalation_at = None
    elif scenario_type == "fraud":
        variant = rng.random()
        if variant < 0.12:
            texts = [
                f"Confirm the vendor paperwork for {organization} before lunch.",
                "Move the package through the alternate delivery route.",
                "Confirm the handoff with the access note.",
                "The receipt number arrives tomorrow.",
                "Close the route quietly after the materials clear.",
            ]
        elif variant < 0.32:
            texts = [
                f"Confirm the vendor invoice for {organization} before lunch.",
                "Use the revised payment routing for the new account ending 31.",
                "Confirm the handoff before the receipt number arrives tomorrow.",
                "Move the package after the delivery route is approved.",
                "Close the route quietly after the materials clear.",
            ]
        else:
            texts = [
                f"Confirm the vendor invoice for {organization} before lunch.",
                "Use the revised payment routing for the new account ending 31.",
                "Urgent payment required today; verify your identity through this form.",
                "Wire transfer the funds and send the password for confirmation.",
                "Send the security code after the transfer clears.",
            ]
        obvious_at = start_at + timedelta(days=3)
        escalation_at = start_at + timedelta(days=4)
    elif scenario_type == "harassment":
        target = participants[0]
        variant = rng.random()
        if variant < 0.16:
            texts = [
                f"Keep extra attention on {target} during every check-in.",
                f"Make sure {target} is left out of the group thread.",
                f"{target} is not a fit here and should stay quiet.",
                f"Everyone should make {target} step away from the project.",
                f"Keep isolating {target}; they do not belong here.",
            ]
        else:
            texts = [
                f"Keep the pressure on {target} during every check-in.",
                f"Nobody wants {target} in this group anymore.",
                f"Stop talking, {target}; you are worthless and a loser.",
                f"{target} should quit before everyone laughs again.",
                f"Keep isolating {target}; they do not belong here.",
            ]
        obvious_at = start_at + timedelta(days=3)
        escalation_at = start_at + timedelta(days=4)
    elif scenario_type == "escalating_threat":
        target = participants[0]
        variant = rng.random()
        if variant < 0.14:
            texts = [
                f"The dispute with {target} is getting tense; meet privately at {location}.",
                f"The package arrives tomorrow near {location}.",
                f"This is the last reminder for {target}.",
                "The weather changes if the window is missed.",
                f"Keep the handoff quiet until {target} understands.",
            ]
        else:
            texts = [
                f"The dispute with {target} is getting tense; meet privately at {location}.",
                f"We need to settle this with {target} before the week ends.",
                f"This is a warning; {target} should avoid creating more problems.",
                f"Do not ignore this or you will regret it, {target}.",
                f"Final warning: {target} will pay the price if this continues.",
            ]
        obvious_at = start_at + timedelta(days=4)
        escalation_at = start_at + timedelta(days=5)
    elif scenario_type == "coordinated_group":
        variant = rng.random()
        if variant < 0.18:
            texts = [
                f"{participants[0]} will arrange logistics for the volunteer session at {location}.",
                f"Bring the larger group to {location} after the public event.",
                "Use the side door and keep a low profile.",
                f"The members return to {location}; share tasks before launch.",
                "Arrange the next step with the full team in the quiet room.",
            ]
        else:
            texts = [
                f"{participants[0]} will coordinate logistics for the volunteer session at {location}.",
                f"Gather the team leaders at {location} after hours.",
                "Use the back entrance and keep the location confidential.",
                f"The same group returns to {location}; assign responsibilities before launch.",
                "Coordinate the next action with the full team at the secret location.",
            ]
        obvious_at = start_at + timedelta(days=4)
        escalation_at = start_at + timedelta(days=5)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    documents = []
    for index, text in enumerate(texts, start=1):
        timestamp = start_at + timedelta(days=index - 1, minutes=rng.randint(0, 45))
        documents.append(
            Document(
                id=f"{scenario_id}-msg-{index:02d}",
                source_type=rng.choice(SOURCE_TYPES),
                source_name=organization,
                timestamp=timestamp,
                text=text,
                metadata={
                    "scenario_id": scenario_id,
                    "participants": participants,
                    "location": location,
                    "organization": organization,
                },
            )
        )

    return documents, obvious_at, escalation_at


def _scaled_counts(total: int) -> Dict[str, int]:
    raw = {key: total * ratio for key, ratio in SCENARIO_MIX.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(SCENARIO_MIX, key=lambda key: (raw[key] - counts[key], SCENARIO_MIX[key]), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _choose_participants(rng: random.Random, population: Sequence[str], scenario_type: str) -> List[str]:
    if scenario_type == "normal":
        count = rng.randint(2, 4)
    elif scenario_type == "coordinated_group":
        count = rng.randint(9, 12)
    else:
        count = rng.randint(3, 6)
    return rng.sample(list(population), count)


def _classifier_score(text: str) -> Tuple[str, float, List[str]]:
    lower = text.lower()
    scored: Dict[str, float] = {}
    reasons_by_type: Dict[str, List[str]] = {}

    for risk_type, profile in CLASSIFIER_PROFILES.items():
        score = 0.0
        reasons: List[str] = []
        for phrase, weight in profile.items():
            if phrase in lower:
                score += weight
                reasons.append(f"{risk_type} phrase: '{phrase}'")
        if score:
            scored[risk_type] = score
            reasons_by_type[risk_type] = reasons

    negative_bonus = min(0.16, 0.03 * sum(1 for term in NEGATIVE_TERMS if term in lower))
    if scored:
        for risk_type in scored:
            scored[risk_type] += negative_bonus

    if not scored:
        return "normal", 0.0, []

    best_type = max(scored, key=scored.get)
    return best_type, min(0.95, scored[best_type]), reasons_by_type.get(best_type, [])


def _temporal_network_score(prefix_docs: Sequence[Document], participants: Sequence[str]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    message_count = len(prefix_docs)
    if message_count >= 3:
        score += min(0.20, 0.06 * (message_count - 2))
        reasons.append("communication frequency increased")

    participant_count = len(set(participants))
    if participant_count >= 6:
        score += min(0.22, 0.035 * (participant_count - 5))
        reasons.append(f"{participant_count} connected participants")

    locations = [_document_location(doc) for doc in prefix_docs]
    top_location, count = Counter(locations).most_common(1)[0]
    if count >= 3:
        score += 0.14
        reasons.append(f"repeated meetings at {top_location}")

    joined_text = " ".join(doc.text.lower() for doc in prefix_docs)
    coded_hits = [term for term in CODED_SEQUENCE_TERMS if term in joined_text]
    if len(coded_hits) >= 3:
        score += 0.34
        reasons.append("coded multi-message sequence matched")

    cluster_terms = [
        "coordinate",
        "arrange",
        "gather",
        "handoff",
        "same group",
        "larger group",
        "full team",
        "share tasks",
        "members return",
    ]
    if participant_count >= 8 and any(term in joined_text for term in cluster_terms):
        score += 0.22
        reasons.append("new high-risk cluster emerged")

    return min(0.62, score), reasons


def _trend_bonus(scores: Sequence[float]) -> float:
    if len(scores) < 3:
        return 0.0
    start = sum(scores[:2]) / 2
    end = sum(scores[-2:]) / 2
    if end - start >= 0.18:
        return 0.08
    return 0.0


def _red_team_text(text: str) -> str:
    replacements = {
        "Urgent payment": "Priority delivery",
        "payment": "delivery",
        "verify your identity": "confirm the handoff",
        "Wire transfer": "Move the package",
        "transfer": "route",
        "funds": "materials",
        "password": "access note",
        "security code": "receipt number",
        "pressure": "attention",
        "worthless": "not a fit",
        "loser": "problem",
        "quit": "step away",
        "Stop talking": "Stay quiet",
        "warning": "last reminder",
        "Final warning": "Last reminder",
        "regret it": "miss the window",
        "pay the price": "deal with the weather changes",
        "consequences": "weather changes",
        "Coordinate": "Arrange",
        "coordinate": "arrange",
        "secret location": "quiet room",
        "confidential": "low profile",
        "back entrance": "side door",
    }
    out = text
    for source, target in replacements.items():
        out = out.replace(source, target)
    return out


def _copy_document(
    doc: Document,
    text: Optional[str] = None,
    doc_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Document:
    return Document(
        id=doc_id or doc.id,
        source_type=doc.source_type,
        source_name=doc.source_name,
        timestamp=timestamp or doc.timestamp,
        text=text if text is not None else doc.text,
        metadata=dict(doc.metadata),
    )


def _select_investigation_scenario(
    challenge: ChallengeSet,
    predictions: Dict[str, ScenarioPrediction],
    scenario_id: Optional[str],
) -> Scenario:
    if scenario_id:
        for scenario in challenge.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise ValueError(f"Unknown scenario id for investigation: {scenario_id}")

    candidates = [
        scenario
        for scenario in challenge.scenarios
        if scenario.positive and predictions.get(scenario.id) and predictions[scenario.id].detected_at
    ]
    if not candidates:
        return next(scenario for scenario in challenge.scenarios if scenario.positive)
    return max(candidates, key=lambda item: predictions[item.id].score)


def _most_common_location(documents: Sequence[Document]) -> Tuple[str, int]:
    locations = [_document_location(doc) for doc in documents]
    return Counter(locations).most_common(1)[0]


def _split_activity_counts(documents: Sequence[Document]) -> Tuple[int, int]:
    ordered = sorted(documents, key=lambda item: item.timestamp or datetime.min)
    midpoint = max(1, len(ordered) // 2)
    return len(ordered[:midpoint]), len(ordered[midpoint:])


def _percent_increase(first: int, second: int) -> float:
    return ((second - first) / max(1, first)) * 100.0


def _risk_points(prediction: ScenarioPrediction) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp": point["timestamp"],
            "score": point["score"],
            "source_id": point["source_id"],
        }
        for point in prediction.trajectory
    ]


def _trajectory_score(prediction: ScenarioPrediction, source_id: str) -> float:
    for point in prediction.trajectory:
        if point["source_id"] == source_id:
            return float(point["score"])
    return 0.0


def _document_location(doc: Document) -> str:
    return str(doc.metadata.get("location") or "unknown")


def _document_to_public_dict(doc: Document) -> Dict[str, Any]:
    data = _model_dump(doc)
    metadata = dict(data.get("metadata") or {})
    metadata.pop("scenario_type", None)
    data["metadata"] = metadata
    return data


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: getattr(value, key)
            for key in value.__dataclass_fields__
            if key not in {"challenge", "predictions"}
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _render_markdown_report(result: Dict[str, Any]) -> str:
    investigation = result["investigation_report"]
    lines = [
        "# Synthetic Evidence Report",
        "",
        "## Scorecard",
        "",
        "```text",
        result["scorecard_text"],
        "```",
        "",
        "## Investigation Sample",
        "",
        f"Scenario: {investigation['scenario_id']} ({investigation['actual_type']})",
        "",
    ]
    for bullet in investigation["bullets"]:
        lines.append(f"- {bullet}")
    lines.extend(["", "Reasons:"])
    for reason in investigation["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", "Top Evidence:"])
    for evidence in investigation["top_evidence"]:
        lines.append(f"- {evidence['timestamp']}: {evidence['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
