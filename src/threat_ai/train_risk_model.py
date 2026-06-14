import json
from datetime import datetime, timedelta
from typing import List, Dict

from .models_ml import XGBoostPredictor, LSTMPredictor, HeuristicPredictor
from .prediction import compute_sentiment


def build_training_data(events: List, event_alerts: Dict[str, object], escalation_threshold: float = 0.2):
    samples = []
    labels = []
    persons = {}
    for event in events:
        for person in getattr(event, "people", []):
            persons.setdefault(person, []).append(event)

    for person, evs in persons.items():
        evs = sorted(evs, key=lambda e: e.timestamp or datetime.min)
        for idx, event in enumerate(evs):
            alert = event_alerts.get(event.id)
            if not alert:
                continue
            current_risk = getattr(alert, "confidence", 0.0)
            current_time = event.timestamp or datetime.min
            future_risks = []
            for future in evs[idx + 1 :]:
                if not future.timestamp:
                    continue
                if (future.timestamp - current_time).days <= 7 and (future.timestamp - current_time).days >= 0:
                    future_alert = event_alerts.get(future.id)
                    if future_alert:
                        future_risks.append(getattr(future_alert, "confidence", 0.0))
            if not future_risks:
                continue
            avg_future = sum(future_risks) / len(future_risks)
            label = 1 if avg_future - current_risk >= escalation_threshold else 0
            features = {
                "current_risk": current_risk,
                "participant_count": len(getattr(event, "people", [])),
                "location_count": len(getattr(event, "locations", [])),
                "num_messages": len(getattr(event, "related_message_ids", [])),
                "sentiment": compute_sentiment(getattr(event, "description", "")),
                "high_risk_count": len([1 for r in future_risks if r >= 0.7]),
            }
            samples.append(features)
            labels.append(label)
    return samples, labels


def train_xgboost_model(samples: List[Dict], labels: List[int]):
    try:
        predictor = XGBoostPredictor()
        # build numeric feature array
        if not hasattr(predictor, "build"):
            raise RuntimeError("XGBoost predictor unavailable")
        predictor.build()
        import numpy as np
        X = np.array([[s["current_risk"], s["participant_count"], s["location_count"], s["num_messages"], s["sentiment"], s["high_risk_count"]] for s in samples])
        y = np.array(labels)
        predictor.train(X, y)
        return predictor
    except Exception as exc:
        print(f"XGBoost unavailable or failed: {exc}")
        return HeuristicPredictor()


def train_lstm_model(samples: List[Dict], labels: List[int], seq_length: int = 7):
    predictor = LSTMPredictor(seq_length=seq_length, feature_dim=6)
    if not hasattr(predictor, "build"):
        raise RuntimeError("LSTM predictor unavailable")
    predictor.build()
    import numpy as np
    # create dummy sequences by repeating current features
    base = np.array([[s["current_risk"], s["participant_count"], s["location_count"], s["num_messages"], s["sentiment"], s["high_risk_count"]] for s in samples])
    X = np.repeat(base[:, None, :], seq_length, axis=1)
    y = np.array(labels)
    predictor.train(X, y)
    return predictor


def save_training_data(samples: List[Dict], labels: List[int], path: str = "training_data.json"):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"samples": samples, "labels": labels}, handle, indent=2)

if __name__ == "__main__":
    print("train_risk_model.py is a helper module; import and use from your training pipeline.")
