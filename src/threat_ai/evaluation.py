from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import math


def build_labeled_windows(events: List[Any], event_alerts: Dict[str, Any], window_days: int = 7):
    # Group events into time windows and attach alerts/labels
    windows = []
    events_sorted = sorted(events, key=lambda e: e.timestamp or datetime.min)
    if not events_sorted:
        return windows
    start = events_sorted[0].timestamp or datetime.min
    end = (start + timedelta(days=window_days))
    current = []
    for e in events_sorted:
        t = e.timestamp or datetime.min
        if t <= end:
            current.append((e, event_alerts.get(e.id)))
        else:
            windows.append(current)
            current = [(e, event_alerts.get(e.id))]
            start = t
            end = start + timedelta(days=window_days)
    if current:
        windows.append(current)
    return windows


def evaluate_events(events: List[Any], event_alerts: Dict[str, Any], ground_truth: Dict[str, Dict[str, Any]]):
    """
    Compute evaluation metrics given events, predicted alerts, and ground-truth escalation labels.
    ground_truth: mapping event_id -> {"escalated": bool, "escalation_time": datetime}
    Returns metrics and per-event records.
    """
    total_events = len(events)
    tp_high = 0
    fp_high = 0
    detected_times = []
    confidences = []
    records = []

    for e in events:
        alert = event_alerts.get(e.id)
        gt = ground_truth.get(e.id, {})
        escalated = bool(gt.get("escalated", False))
        escalation_time = gt.get("escalation_time")
        conf = getattr(alert, "confidence", 0.0) if alert else 0.0
        level = getattr(alert, "level", None) if alert else None
        confidences.append(conf)

        is_high = level is not None and str(level).lower() in ("high", "critical")

        if is_high and escalated:
            tp_high += 1
        if is_high and not escalated:
            fp_high += 1

        # time-to-detection: if escalated, check if alert exists before escalation
        detection_lead = None
        if escalated and escalation_time:
            if is_high and getattr(alert, "created_at", None):
                detection_lead = (escalation_time - alert.created_at).total_seconds()
                detected_times.append(detection_lead)

        records.append({"event_id": e.id, "escalated": escalated, "pred_confidence": conf, "pred_level": str(level) if level else None, "detection_lead_seconds": detection_lead})

    precision_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else None
    fp_rate_per_1000 = (fp_high / total_events * 1000.0) if total_events > 0 else None
    avg_time_to_detection = (sum(detected_times) / len(detected_times)) if detected_times else None

    # stability: compute stddev of confidences over whole period
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    var = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences) if confidences else 0.0
    stddev_conf = math.sqrt(var)

    metrics = {
        "precision_high": precision_high,
        "false_positive_rate_per_1000_events": fp_rate_per_1000,
        "avg_time_to_detection_seconds": avg_time_to_detection,
        "confidence_stddev": stddev_conf,
        "total_events": total_events,
    }
    return metrics, records
