from datetime import datetime, timedelta
from typing import Optional
from types import SimpleNamespace
from .storage import init_db, query_feedback, append_audit
from .train_risk_model import build_training_data, train_xgboost_model, train_lstm_model


def retrain_if_needed(db_path: str, min_feedback_samples: int = 50):
    conn = init_db(db_path)
    feedback = query_feedback(conn)
    if len(feedback) < min_feedback_samples:
        conn.close()
        return None

    # assemble events and event_alerts from storage
    cur = conn.cursor()
    cur.execute("SELECT id, summary, start_time FROM events")
    events_rows = cur.fetchall()
    events = []
    for row in events_rows:
        eid = row[0]
        summary = row[1]
        start_time = row[2]
        # participants
        cur.execute("SELECT participant FROM event_participants WHERE event_id = ?", (eid,))
        people = [r[0] for r in cur.fetchall()]
        # related messages
        cur.execute("SELECT message_id FROM event_message_map WHERE event_id = ?", (eid,))
        msgs = [r[0] for r in cur.fetchall()]
        # risk
        cur.execute("SELECT risk_score FROM event_risk WHERE event_id = ?", (eid,))
        rr = cur.fetchone()
        risk_score = rr[0] if rr else 0.0

        ev = SimpleNamespace(id=eid, description=summary or "", timestamp=start_time, people=people, related_message_ids=msgs)
        events.append(ev)

    # build event_alerts mapping expected by build_training_data (objects with .confidence)
    event_alerts = {}
    for ev in events:
        cur.execute("SELECT risk_score FROM event_risk WHERE event_id = ?", (ev.id,))
        r = cur.fetchone()
        if r:
            event_alerts[ev.id] = SimpleNamespace(confidence=r[0])

    samples, labels = build_training_data(events, event_alerts)
    append_audit(conn, event_id="*", action="retrain_started", payload=f"feedback_count={len(feedback)},samples={len(samples)}", timestamp=datetime.utcnow().isoformat())

    model = None
    try:
        if len(samples) >= 20:
            model = train_xgboost_model(samples, labels)
        elif len(samples) >= 10:
            model = train_lstm_model(samples, labels)
        else:
            # not enough samples; fallback
            model = None
    except Exception as exc:
        append_audit(conn, event_id="*", action="retrain_error", payload=str(exc), timestamp=datetime.utcnow().isoformat())

    append_audit(conn, event_id="*", action="retrain_finished", payload=f"model_trained={bool(model)}", timestamp=datetime.utcnow().isoformat())
    conn.close()
    return model
