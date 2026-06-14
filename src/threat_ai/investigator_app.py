from flask import Flask, jsonify, request, render_template_string
import sqlite3
from typing import List, Dict

from .entities import extract_entities
from .feedback import record_feedback

app = Flask(__name__)

TEMPLATE = """
<html>
<head>
    <title>Investigator View</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; }
        .event { border: 1px solid #ccc; padding: 10px; margin: 5px 0; }
        canvas { max-width: 600px; }
    </style>
</head>
<body>
<div class="container">
<h1>Person: {{name}}</h1>
<h2>Risk Trend</h2>
<canvas id="riskChart"></canvas>
<h2>Events</h2>
<ul>
{% for ev in events %}
  <div class="event">
    <strong>{{ev.id}}</strong>: {{ev.summary}} (risk: {{ev.risk_score or 'N/A'}})
    <br/>Messages: {{ev.messages|join(', ')}}
  </div>
{% endfor %}
</ul>
<script>
var riskData = {{risk_data | tojson}};
var ctx = document.getElementById('riskChart').getContext('2d');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: riskData.dates,
        datasets: [{
            label: 'Risk Score',
            data: riskData.scores,
            borderColor: 'rgb(255,99,132)',
            backgroundColor: 'rgba(255,99,132,0.1)',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: { min: 0, max: 1 }
        }
    }
});
</script>
</div>
</body>
</html>
"""


def get_db(path: str = "threat_detector.db"):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def events_for_person(conn, person_name: str) -> List[Dict]:
    cur = conn.cursor()
    # Find messages containing person, then find events
    cur.execute("SELECT id, text FROM messages WHERE text LIKE ?", (f"%{person_name}%",))
    messages = cur.fetchall()
    if not messages:
        return []
    msg_ids = [m[0] for m in messages]
    # Find events referencing these message ids
    events = {}
    for mid in msg_ids:
        cur.execute("SELECT event_id FROM event_message_map WHERE message_id = ?", (mid,))
        for row in cur.fetchall():
            eid = row[0]
            if eid not in events:
                cur2 = conn.cursor()
                cur2.execute("SELECT id, summary FROM events WHERE id = ?", (eid,))
                r = cur2.fetchone()
                cur3 = conn.cursor()
                cur3.execute("SELECT risk_score FROM event_risk WHERE event_id = ?", (eid,))
                rs = cur3.fetchone()
                events[eid] = {
                    "id": eid,
                    "summary": r[1] if r else "",
                    "risk_score": rs[0] if rs else 0.0,
                    "messages": []
                }
            events[eid]["messages"].append(mid)
    return list(events.values())


def get_risk_trend(conn, person_name: str) -> Dict:
    """Get risk trend data for a person."""
    cur = conn.cursor()
    # Query risk scores over time from event_risk joined with event_participants
    cur.execute("""
        SELECT er.timestamp, er.risk_score FROM event_risk er
        JOIN event_participants ep ON er.event_id = ep.event_id
        WHERE ep.participant = ?
        ORDER BY er.timestamp ASC
    """, (person_name,))
    rows = cur.fetchall()
    return {
        "dates": [r[0] or "Unknown" for r in rows],
        "scores": [r[1] or 0.0 for r in rows]
    }


@app.route("/person/<name>")
def person_view(name):
    db_path = request.args.get("db", "threat_detector.db")
    conn = get_db(db_path)
    events = events_for_person(conn, name)
    risk_data = get_risk_trend(conn, name)
    return render_template_string(TEMPLATE, name=name, events=events, risk_data=risk_data)


@app.route("/api/person/<name>")
def person_api(name):
    db_path = request.args.get("db", "threat_detector.db")
    conn = get_db(db_path)
    events = events_for_person(conn, name)
    risk_data = get_risk_trend(conn, name)
    return jsonify({"person": name, "events": events, "risk_data": risk_data})


@app.route("/api/feedback", methods=["POST"])
def post_feedback():
    data = request.get_json() or {}
    event_id = data.get("event_id")
    user = data.get("user", "analyst")
    label = data.get("label")
    note = data.get("note")
    db_path = data.get("db", "threat_detector.db")
    if not event_id or not label:
        return jsonify({"error": "event_id and label are required"}), 400
    try:
        record_feedback(db_path, event_id, user, label, note)
        return jsonify({"status": "ok"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/tasks")
def get_tasks_api():
    db_path = request.args.get("db", "threat_detector.db")
    conn = get_db(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, event_id, action, priority, assigned_to, created_at, status, dedup_key FROM task_queue ORDER BY priority DESC, created_at ASC")
    rows = cur.fetchall()
    tasks = [dict(r) for r in rows]
    return jsonify({"tasks": tasks})


@app.route("/api/override", methods=["POST"])
def post_override():
    data = request.get_json() or {}
    event_id = data.get("event_id")
    user = data.get("user", "analyst")
    action = data.get("action")
    reason = data.get("reason")
    db_path = data.get("db", "threat_detector.db")
    if not event_id or not action:
        return jsonify({"error": "event_id and action are required"}), 400
    try:
        from .task_queue import override_event

        res = override_event(db_path, event_id, user, action, reason)
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def run(host: str = "127.0.0.1", port: int = 5000):
    app.run(host=host, port=port)


if __name__ == "__main__":
    run()
