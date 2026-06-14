from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import math

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


def compute_sentiment(text: str) -> float:
    """Return sentiment polarity in [-1, 1]; -1 is most negative."""
    if TextBlob:
        try:
            return float(TextBlob(text).sentiment.polarity)
        except Exception:
            return 0.0
    return 0.0


def events_for_person(person: str, events: List) -> List:
    result = []
    for e in events:
        if person in getattr(e, "people", []):
            result.append(e)
    return sorted(result, key=lambda ev: ev.timestamp or datetime.min)


def days_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def linear_slope(xs: List[float], ys: List[float]) -> float:
    # simple linear regression slope
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def extract_features_for_person(
    person: str,
    events: List,
    event_alerts: Dict[str, object],
    lookback_days: int = 30,
    now: Optional[datetime] = None,
) -> Dict:
    if now is None:
        now = datetime.utcnow()
    cutoff = now - timedelta(days=lookback_days)

    person_events = [e for e in events if person in getattr(e, "people", [])]
    recent = [e for e in person_events if (e.timestamp or now) >= cutoff]

    # event count
    event_count = len(recent)

    # events per day time series
    days = {}
    for e in recent:
        d = (e.timestamp or now).date()
        days.setdefault(d, 0)
        days[d] += 1
    if days:
        sorted_days = sorted(days.items())
        xs = list(range(len(sorted_days)))
        ys = [c for _, c in sorted_days]
        freq_slope = linear_slope(xs, ys)
    else:
        freq_slope = 0.0

    # risk trend
    risk_points_x = []
    risk_points_y = []
    high_risk_count = 0
    for e in recent:
        alert = event_alerts.get(e.id)
        if alert:
            t = days_between(cutoff, e.timestamp or now)
            risk = float(getattr(alert, "confidence", 0.0))
            risk_points_x.append(t)
            risk_points_y.append(risk)
            if risk >= 0.7:
                high_risk_count += 1
    if risk_points_x:
        risk_slope = linear_slope(risk_points_x, risk_points_y)
        current_risk = sum(risk_points_y) / len(risk_points_y)
    else:
        risk_slope = 0.0
        current_risk = 0.0

    # location repetition
    locs = []
    for e in recent:
        locs.extend(getattr(e, "locations", []))
    distinct_locs = len(set(locs))

    # new connections: participants in recent events excluding the person
    participants = set()
    for e in recent:
        participants.update(getattr(e, "people", []))
    participants.discard(person)
    new_connections = len(participants)

    features = {
        "person": person,
        "event_count": event_count,
        "freq_slope": freq_slope,
        "current_risk": current_risk,
        "risk_slope": risk_slope,
        "high_risk_count": high_risk_count,
        "distinct_locations": distinct_locs,
        "new_connections": new_connections,
        "lookback_days": lookback_days,
    }
    return features


def forecast_trajectory_from_features(features: Dict, days: List[int] = [1, 7, 30]) -> Dict:
    # Heuristic forecasting: linear extrapolation of current risk by risk_slope
    current = features.get("current_risk", 0.0)
    slope = features.get("risk_slope", 0.0)
    results = {}
    for d in days:
        pred = current + slope * d
        pred = max(0.0, min(1.0, pred))
        results[d] = pred
    # confidence: based on event_count and lookback
    ec = features.get("event_count", 0)
    confidence = min(0.99, 0.3 + 0.1 * min(ec, 10))
    return {"current": current, "forecast": results, "confidence": confidence}


def predict_for_top_people(events: List, event_alerts: Dict[str, object], top_people: List[str], lookback_days: int = 30):
    forecasts = {}
    for person in top_people:
        feats = extract_features_for_person(person, events, event_alerts, lookback_days=lookback_days)
        fc = forecast_trajectory_from_features(feats)
        forecasts[person] = {"features": feats, "forecast": fc}
    return forecasts


def extract_sentiment_trend(person: str, events: List, lookback_days: int = 30, now: Optional[datetime] = None) -> float:
    """Compute average sentiment polarity across messages in events."""
    if now is None:
        now = datetime.utcnow()
    cutoff = now - timedelta(days=lookback_days)
    person_events = [e for e in events if person in getattr(e, "people", [])]
    recent = [e for e in person_events if (e.timestamp or now) >= cutoff]
    sentiments = [compute_sentiment(e.description or "") for e in recent]
    return sum(sentiments) / len(sentiments) if sentiments else 0.0


def extract_connection_timestamps(person: str, events: List, lookback_days: int = 30, now: Optional[datetime] = None) -> Dict[str, datetime]:
    """Return dict of person -> first timestamp of interaction."""
    if now is None:
        now = datetime.utcnow()
    cutoff = now - timedelta(days=lookback_days)
    person_events = [e for e in events if person in getattr(e, "people", [])]
    recent = [e for e in person_events if (e.timestamp or now) >= cutoff]
    connections = {}
    for e in recent:
        for p in getattr(e, "people", []):
            if p != person and p not in connections:
                connections[p] = e.timestamp or now
    return connections


def predict_group_trajectory(group: List[str], events: List, event_alerts: Dict, lookback_days: int = 30) -> Dict:
    """Predict risk for a group of people as the avg of individual predictions."""
    individual_risks = []
    for person in group:
        feats = extract_features_for_person(person, events, event_alerts, lookback_days=lookback_days)
        individual_risks.append(feats.get("current_risk", 0.0))
    group_risk = sum(individual_risks) / len(individual_risks) if individual_risks else 0.0
    return {"group": group, "members": len(group), "predicted_risk": group_risk}


def simulate_graph_change(person: str, removed_connection: str, events: List, event_alerts: Dict, lookback_days: int = 30) -> Dict:
    """Simulate the impact of removing a person connection on risk trajectory.
    
    Returns the predicted new risk score and impact estimate.
    """
    current_feats = extract_features_for_person(person, events, event_alerts, lookback_days=lookback_days)
    current_risk = current_feats.get("current_risk", 0.0)
    current_new_conns = current_feats.get("new_connections", 0)
    
    # simple heuristic: removing a connection reduces new_connections by 1
    simulated_new_conns = max(0, current_new_conns - 1)
    reduction_pct = (current_new_conns - simulated_new_conns) / max(1, current_new_conns)
    simulated_risk = max(0.0, current_risk - reduction_pct * 0.1)
    
    return {
        "person": person,
        "removed_connection": removed_connection,
        "current_risk": current_risk,
        "simulated_risk": simulated_risk,
        "impact": current_risk - simulated_risk
    }


def suggest_interventions(person: str, events: List, event_alerts: Dict, top_n: int = 3) -> List[Dict]:
    """Suggest interventions to reduce risk trajectory.
    
    Examples:
    - Reduce contact with high-risk person X
    - Change meeting location to lower-risk area
    - Monitor for escalation patterns
    """
    feats = extract_features_for_person(person, events, event_alerts)
    suggestions = []
    
    if feats.get("risk_slope", 0) > 0.1:
        suggestions.append({
            "priority": "high",
            "action": "Monitor escalation",
            "reason": "Risk score increasing over time",
            "impact": -0.2
        })
    
    if feats.get("new_connections", 0) > 5:
        suggestions.append({
            "priority": "medium",
            "action": "Reduce new contacts",
            "reason": "Rapid expansion of contacts",
            "impact": -0.15
        })
    
    if feats.get("high_risk_count", 0) > 2:
        suggestions.append({
            "priority": "high",
            "action": "Investigate high-risk events",
            "reason": "Multiple high-confidence alerts",
            "impact": -0.25
        })
    
    if feats.get("distinct_locations", 0) > 5:
        suggestions.append({
            "priority": "low",
            "action": "Stabilize meeting locations",
            "reason": "High location variance",
            "impact": -0.1
        })
    
    return sorted(suggestions, key=lambda x: x["priority"] == "high", reverse=True)[:top_n]

