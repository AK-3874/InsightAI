import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    CountVectorizer = None
    LogisticRegression = None
    train_test_split = None

from .models import Document
from .risk import score_risk
from .prediction import compute_sentiment
from .pipeline import ThreatPipeline


@dataclass
class BenchmarkMetrics:
    precision: Optional[float]
    recall: Optional[float]
    false_positives: int
    false_positive_rate: Optional[float]
    detection_lead_seconds: Optional[float]
    workload_reduction: Optional[float]
    total_events: int
    predicted_positive: int


class RuleBasedBaseline:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, documents: List[Document]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for doc in documents:
            alert = score_risk(doc, [])
            score = float(alert.confidence)
            action = "positive" if score >= self.threshold else "negative"
            results[doc.id] = {
                "score": score,
                "action": action,
                "timestamp": doc.timestamp,
            }
        return results


class MLBaseline:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.keyword_weights = None

    def build_features(self, texts: List[str]) -> List[List[float]]:
        if HAS_SKLEARN and self.vectorizer:
            return self.vectorizer.transform(texts).toarray().tolist()

        features = []
        for text in texts:
            lower = text.lower()
            features.append([
                float(lower.count("threat")),
                float(lower.count("weapon")),
                float(lower.count("meet")),
                float(lower.count("loan")),
                float(lower.count("bank")),
                float(lower.count("urgent")),
                float(len(lower.split())),
                float(sum(1 for c in lower if c in "!,?")),
            ])
        return features

    def train(self, documents: List[Document], labels: List[int]):
        texts = [doc.text for doc in documents]
        if HAS_SKLEARN:
            self.vectorizer = CountVectorizer(max_features=500)
            X = self.vectorizer.fit_transform(texts)
            self.model = LogisticRegression(max_iter=200)
            self.model.fit(X, labels)
        else:
            self.keyword_weights = {
                "threat": 2.0,
                "weapon": 1.5,
                "meet": 1.2,
                "urgent": 1.0,
                "bank": 0.8,
                "loan": 0.8,
                "danger": 1.3,
                "regret": 1.0,
            }

    def predict(self, documents: List[Document]) -> Dict[str, Dict[str, Any]]:
        results = {}
        texts = [doc.text for doc in documents]
        if HAS_SKLEARN and self.model and self.vectorizer:
            X = self.vectorizer.transform(texts)
            prob = self.model.predict_proba(X)[:, 1].tolist()
            for doc, p in zip(documents, prob):
                results[doc.id] = {"score": p, "action": "positive" if p >= 0.5 else "negative", "timestamp": doc.timestamp}
            return results

        for doc in documents:
            lower = doc.text.lower()
            score = 0.0
            for keyword, weight in (self.keyword_weights or {}).items():
                score += lower.count(keyword) * weight
            score = min(0.99, score / 10.0)
            results[doc.id] = {"score": score, "action": "positive" if score >= 0.5 else "negative", "timestamp": doc.timestamp}
        return results


class FullIntelligenceSystem:
    def __init__(self):
        self.pipeline = ThreatPipeline()

    def predict(self, documents: List[Document]) -> Dict[str, Dict[str, Any]]:
        results = self.pipeline.run(documents)
        decisions = results.get("decisions", {})
        event_docs = {}
        for event in results.get("events", []):
            for message_id in event.get("related_message_ids", []):
                event_docs[message_id] = event.get("id")

        predictions = {}
        for doc in documents:
            event_id = event_docs.get(doc.id)
            decision = decisions.get(event_id, {"action": "MONITOR", "reasons": ["no event mapping"]})
            predictions[doc.id] = {
                "score": None,
                "action": decision["action"],
                "reasons": decision.get("reasons", []),
                "timestamp": doc.timestamp,
            }
        return predictions


def compute_metrics(predictions: Dict[str, Dict[str, Any]], ground_truth: Dict[str, Any]) -> BenchmarkMetrics:
    tp = 0
    fp = 0
    fn = 0
    detection_leads = []
    predicted_positive = 0

    for doc_id, pred in predictions.items():
        positive = pred["action"] != "IGNORE" and pred["action"] != "negative"
        actual = ground_truth.get(doc_id, {}).get("escalated", False)
        if positive:
            predicted_positive += 1
        if positive and actual:
            tp += 1
            doc_time = pred.get("timestamp")
            esc_time = ground_truth.get(doc_id, {}).get("escalation_time")
            if esc_time and doc_time:
                detection_leads.append((esc_time - doc_time).total_seconds())
        if positive and not actual:
            fp += 1
        if not positive and actual:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    fp_rate = fp / (len(predictions) or 1)
    avg_lead = sum(detection_leads) / len(detection_leads) if detection_leads else None
    workload_reduction = 1.0 - (predicted_positive / (len(predictions) or 1))
    return BenchmarkMetrics(
        precision=precision,
        recall=recall,
        false_positives=fp,
        false_positive_rate=fp_rate,
        detection_lead_seconds=avg_lead,
        workload_reduction=workload_reduction,
        total_events=len(predictions),
        predicted_positive=predicted_positive,
    )


def compare_systems(systems: Dict[str, Any], documents: List[Document], ground_truth: Dict[str, Any]) -> Dict[str, BenchmarkMetrics]:
    results = {}
    for name, system in systems.items():
        predictions = system.predict(documents)
        metrics = compute_metrics(predictions, ground_truth)
        results[name] = metrics
    return results
