from typing import List, Dict, Any


def prioritize_scenarios(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank scenarios by three axes: worst-case (max impact), most-likely (probability), fastest (time_to_escalation).

    Each scenario is expected to have keys: 'probability', 'impact', 'time_to_escalation'
    Returns list sorted by a composite score and annotated ranks.
    """
    for s in scenarios:
        prob = float(s.get("probability", 0.0))
        impact = float(s.get("impact", 0.0))
        time = float(s.get("time_to_escalation", 1e6))
        # composite: weight worst-case (impact), then probability, then speed
        s["score"] = impact * 0.6 + prob * 0.3 + max(0, (1.0 - min(time / 30.0, 1.0))) * 0.1
    ranked = sorted(scenarios, key=lambda x: x["score"], reverse=True)
    return ranked
