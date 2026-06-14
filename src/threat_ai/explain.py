from typing import Dict, Any, List


def compress_explanation(signals: Dict[str, float]) -> Dict[str, Any]:
    """Convert raw signal values into a compact structured justification.

    signals: dict like {"sentiment_spike": 0.62, "new_connections":4, ...}
    Returns: {"summary": str, "signals": List[{'name', 'delta', 'impact'}]}
    """
    parts: List[Dict[str, Any]] = []
    for k, v in signals.items():
        impact = "neutral"
        if isinstance(v, (int, float)):
            if v >= 0.7:
                impact = "strong_positive"
            elif v >= 0.3:
                impact = "positive"
            elif v <= -0.7:
                impact = "strong_negative"
            elif v <= -0.3:
                impact = "negative"
            else:
                impact = "neutral"
        parts.append({"name": k, "value": float(v) if isinstance(v, (int, float)) else v, "impact": impact})

    # build short summary
    positives = [p for p in parts if p.get("impact") in ("strong_positive", "positive")]
    negatives = [p for p in parts if p.get("impact") in ("strong_negative", "negative")]
    summary = []
    if positives:
        summary.append("Signals increasing: " + ", ".join(p["name"] for p in positives[:3]))
    if negatives:
        summary.append("Signals decreasing: " + ", ".join(p["name"] for p in negatives[:3]))
    if not summary:
        summary = ["No major signal changes detected"]

    return {"summary": "; ".join(summary), "signals": parts}
