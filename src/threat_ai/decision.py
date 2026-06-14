from typing import Dict, Any
from .storage import init_db, query_feedback


DEFAULT_CONFIG = {
    "escalate_threshold": 0.75,
    "monitor_threshold": 0.4,
    "uncertainty_limit": "medium",  # allow escalate if uncertainty <= this
    "max_fp_rate_recent": 0.2,  # max fraction of recent feedback marked false_positive
}


def _recent_false_positive_rate(conn, lookback_events=100):
    # compute recent false positive fraction from feedback table
    cur = conn.cursor()
    cur.execute("SELECT label FROM feedback ORDER BY timestamp DESC LIMIT ?", (lookback_events,))
    rows = cur.fetchall()
    if not rows:
        return 0.0
    total = len(rows)
    fps = sum(1 for r in rows if r[0] and r[0].lower().startswith("false"))
    return float(fps) / total


def decide_event(event: Any, alert: Any, uncertainty: Dict[str, Any], features: Dict[str, Any], config: Dict[str, Any] = None, db_path: str = None):
    """Return decision action and reasons for an event.
    Actions: IGNORE, MONITOR, ESCALATE, PRIORITY_ALERT
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    score = float(getattr(alert, "confidence", 0.0)) if alert else 0.0
    slope = float(features.get("risk_slope", 0.0))
    unc_label = uncertainty.get("uncertainty", "high") if uncertainty else "high"

    reasons = []
    # check recent feedback false positive rate
    fp_rate = 0.0
    if db_path:
        conn = init_db(db_path)
        try:
            fp_rate = _recent_false_positive_rate(conn)
        finally:
            conn.close()

    # Priority alert: very high confidence and rising slope and low uncertainty
    if score >= 0.95 and slope > 0 and unc_label == "low" and fp_rate <= cfg["max_fp_rate_recent"]:
        reasons.append("very high confidence and rising trend")
        return "PRIORITY_ALERT", reasons

    # Escalate rules
    if score >= cfg["escalate_threshold"] and unc_label in ("low", "medium") and slope >= 0:
        if fp_rate > cfg["max_fp_rate_recent"]:
            reasons.append("recent false-positive rate high; suppressing escalate")
        else:
            reasons.append("meets threshold, uncertainty acceptable, trend non-negative")
            return "ESCALATE", reasons

    # Monitor: mid confidence or high uncertainty
    if score >= cfg["monitor_threshold"] or unc_label != "low":
        reasons.append("monitor: mid confidence or elevated uncertainty")
        return "MONITOR", reasons

    reasons.append("below thresholds")
    return "IGNORE", reasons
