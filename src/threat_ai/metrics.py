from datetime import datetime, timedelta
from typing import Dict, Any
from .storage import init_db, store_system_metric


def collect_system_metrics(db_path: str):
    conn = init_db(db_path)
    cur = conn.cursor()
    # alert volume per day (last 7 days)
    cur.execute("SELECT timestamp FROM event_risk")
    rows = cur.fetchall()
    times = [r[0] for r in rows if r and r[0]]
    # naive volume: count total in last 24h
    now = datetime.utcnow()
    day_ago = (now - timedelta(days=1)).isoformat()
    cur.execute("SELECT COUNT(*) FROM event_risk WHERE timestamp >= ?", (day_ago,))
    alert_vol = cur.fetchone()[0]
    store_system_metric(conn, "alert_volume_last_24h", float(alert_vol))

    # false positive ratio based on feedback last 100
    cur.execute("SELECT label FROM feedback ORDER BY timestamp DESC LIMIT 100")
    labels = [r[0] for r in cur.fetchall()]
    fps = sum(1 for l in labels if l and l.lower().startswith("false"))
    fp_ratio = float(fps) / len(labels) if labels else 0.0
    store_system_metric(conn, "false_positive_ratio_recent", float(fp_ratio))

    # model confidence stability (stddev) from evaluation snapshots if present
    # look for pipeline_evaluation_snapshot.json? For now compute stddev from event_risk
    cur.execute("SELECT risk_score FROM event_risk")
    scores = [r[0] for r in cur.fetchall() if r and r[0] is not None]
    if scores:
        import math

        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var)
    else:
        std = 0.0
    store_system_metric(conn, "model_confidence_stddev", float(std))
    conn.close()
