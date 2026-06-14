import sqlite3
from typing import List

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, text TEXT, source_type TEXT, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, type TEXT, summary TEXT, start_time TEXT, end_time TEXT, confidence REAL);",
    "CREATE TABLE IF NOT EXISTS event_message_map (event_id TEXT, message_id TEXT, PRIMARY KEY(event_id,message_id));",
    "CREATE TABLE IF NOT EXISTS event_participants (event_id TEXT, participant TEXT, PRIMARY KEY(event_id, participant));",
    "CREATE TABLE IF NOT EXISTS event_risk (event_id TEXT PRIMARY KEY, risk_level TEXT, risk_score REAL, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, type TEXT, source_id TEXT);",
    "CREATE TABLE IF NOT EXISTS entity_relations (subject TEXT, predicate TEXT, object TEXT);",
    "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, user TEXT, label TEXT, note TEXT, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, action TEXT, payload TEXT, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS overrides (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, user TEXT, action TEXT, reason TEXT, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS task_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, action TEXT, priority INTEGER, assigned_to TEXT, created_at TEXT, status TEXT, dedup_key TEXT);",
    "CREATE TABLE IF NOT EXISTS system_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, metric_name TEXT, metric_value REAL, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS beliefs (id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT, confidence REAL, source_ids TEXT, created_at TEXT, explanation TEXT);",
    "CREATE TABLE IF NOT EXISTS hypotheses (id TEXT PRIMARY KEY, description TEXT, probability REAL, evidence TEXT, timestamp TEXT);",
    "CREATE TABLE IF NOT EXISTS analysis_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, investigation_id TEXT, summary TEXT, pattern TEXT, outcome TEXT, tags TEXT, note TEXT, timestamp TEXT);",
]


def init_db(path: str = "threat_detector.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    for s in SCHEMA:
        cur.execute(s)
    conn.commit()
    return conn


def store_messages(conn: sqlite3.Connection, messages: List[dict]):
    cur = conn.cursor()
    for m in messages:
        cur.execute(
            "INSERT OR REPLACE INTO messages (id, text, source_type, timestamp) VALUES (?, ?, ?, ?)",
            (m.get("id"), m.get("message") or m.get("text"), m.get("source_type"), m.get("timestamp")),
        )
    conn.commit()


def store_events(conn: sqlite3.Connection, events: List[dict]):
    cur = conn.cursor()
    for e in events:
        start = None
        end = None
        cur.execute(
            "INSERT OR REPLACE INTO events (id, type, summary, start_time, end_time, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (e.get("id"), str(e.get("type")), e.get("description"), start, end, e.get("confidence") or 0.0),
        )
        for mid in e.get("related_message_ids", []):
            cur.execute(
                "INSERT OR REPLACE INTO event_message_map (event_id, message_id) VALUES (?, ?)",
                (e.get("id"), mid),
            )
        for participant in e.get("people", []):
            cur.execute(
                "INSERT OR REPLACE INTO event_participants (event_id, participant) VALUES (?, ?)",
                (e.get("id"), participant),
            )
    conn.commit()


def store_event_risk(conn: sqlite3.Connection, event_id: str, risk_level: str, risk_score: float, timestamp: str = None):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO event_risk (event_id, risk_level, risk_score, timestamp) VALUES (?, ?, ?, ?)",
        (event_id, risk_level, risk_score, timestamp),
    )
    conn.commit()


def store_feedback(conn: sqlite3.Connection, event_id: str, user: str, label: str, note: str = None, timestamp: str = None):
    cur = conn.cursor()
    if not timestamp:
        from datetime import datetime

        timestamp = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO feedback (event_id, user, label, note, timestamp) VALUES (?, ?, ?, ?, ?)",
        (event_id, user, label, note, timestamp),
    )
    conn.commit()


def query_feedback(conn: sqlite3.Connection, event_id: str = None) -> List[dict]:
    cur = conn.cursor()
    if event_id:
        cur.execute("SELECT id, event_id, user, label, note, timestamp FROM feedback WHERE event_id = ?", (event_id,))
    else:
        cur.execute("SELECT id, event_id, user, label, note, timestamp FROM feedback")
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def append_audit(conn: sqlite3.Connection, event_id: str, action: str, payload: str, timestamp: str = None):
    cur = conn.cursor()
    if not timestamp:
        from datetime import datetime

        timestamp = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO audit_log (event_id, action, payload, timestamp) VALUES (?, ?, ?, ?)",
        (event_id, action, payload, timestamp),
    )
    conn.commit()


def store_override(conn: sqlite3.Connection, event_id: str, user: str, action: str, reason: str = None, timestamp: str = None):
    cur = conn.cursor()
    if not timestamp:
        from datetime import datetime

        timestamp = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO overrides (event_id, user, action, reason, timestamp) VALUES (?, ?, ?, ?, ?)",
        (event_id, user, action, reason, timestamp),
    )
    conn.commit()


def enqueue_task(conn: sqlite3.Connection, event_id: str, action: str, priority: int = 5, assigned_to: str = None, dedup_key: str = None, status: str = "queued"):
    cur = conn.cursor()
    from datetime import datetime

    created_at = datetime.utcnow().isoformat()
    # deduplication: if dedup_key provided and a recent task exists, return existing
    if dedup_key:
        cur.execute("SELECT id, status FROM task_queue WHERE dedup_key = ?", (dedup_key,))
        r = cur.fetchone()
        if r:
            return {"existing_task_id": r[0], "status": r[1]}

    cur.execute(
        "INSERT INTO task_queue (event_id, action, priority, assigned_to, created_at, status, dedup_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (event_id, action, priority, assigned_to, created_at, status, dedup_key),
    )
    conn.commit()
    return {"task_id": cur.lastrowid, "status": status}


def get_tasks(conn: sqlite3.Connection, assigned_to: str = None, status: str = None):
    cur = conn.cursor()
    q = "SELECT id, event_id, action, priority, assigned_to, created_at, status, dedup_key FROM task_queue"
    params = []
    clauses = []
    if assigned_to:
        clauses.append("assigned_to = ?")
        params.append(assigned_to)
    if status:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY priority DESC, created_at ASC"
    cur.execute(q, tuple(params))
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def query_overrides(conn: sqlite3.Connection, event_id: str = None):
    cur = conn.cursor()
    if event_id:
        cur.execute("SELECT id, event_id, user, action, reason, timestamp FROM overrides WHERE event_id = ? ORDER BY timestamp DESC", (event_id,))
    else:
        cur.execute("SELECT id, event_id, user, action, reason, timestamp FROM overrides ORDER BY timestamp DESC")
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def store_belief(conn: sqlite3.Connection, belief):
    cur = conn.cursor()
    from datetime import datetime

    timestamp = datetime.utcnow().isoformat()
    source_ids = ",".join(belief.source_ids or [])
    cur.execute(
        "INSERT OR REPLACE INTO beliefs (id, subject, predicate, object, confidence, source_ids, created_at, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (belief.subject + ":" + belief.predicate + ":" + belief.object, belief.subject, belief.predicate, belief.object, belief.confidence or 0.0, source_ids, timestamp, getattr(belief, "explanation", None)),
    )
    conn.commit()


def store_hypothesis(conn: sqlite3.Connection, hypothesis: dict):
    cur = conn.cursor()
    from datetime import datetime

    timestamp = datetime.utcnow().isoformat()
    evidence = "; ".join(hypothesis.get("evidence", []))
    hyp_id = hypothesis.get("hypothesis", str(timestamp))
    cur.execute(
        "INSERT OR REPLACE INTO hypotheses (id, description, probability, evidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (hyp_id, hypothesis.get("hypothesis"), hypothesis.get("probability", 0.0), evidence, timestamp),
    )
    conn.commit()


def store_memory_record(conn: sqlite3.Connection, investigation_id: str, summary: str, pattern: str, outcome: str, tags: List[str], note: str = None, timestamp: str = None):
    cur = conn.cursor()
    from datetime import datetime

    if not timestamp:
        timestamp = datetime.utcnow().isoformat()
    tag_text = ",".join(tags or [])
    cur.execute(
        "INSERT INTO analysis_memory (investigation_id, summary, pattern, outcome, tags, note, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (investigation_id, summary, pattern, outcome, tag_text, note, timestamp),
    )
    conn.commit()


def query_memories(conn: sqlite3.Connection, tags: List[str] = None, keywords: List[str] = None):
    cur = conn.cursor()
    clauses = []
    params = []
    if tags:
        for tag in tags:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
    if keywords:
        for keyword in keywords:
            clauses.append("(summary LIKE ? OR pattern LIKE ? OR outcome LIKE ? OR note LIKE ?)")
            params.extend([f"%{keyword}%"] * 4)
    query = "SELECT id, investigation_id, summary, pattern, outcome, tags, note, timestamp FROM analysis_memory"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp DESC"
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def store_system_metric(conn: sqlite3.Connection, metric_name: str, metric_value: float, timestamp: str = None):
    cur = conn.cursor()
    if not timestamp:
        from datetime import datetime

        timestamp = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO system_metrics (metric_name, metric_value, timestamp) VALUES (?, ?, ?)",
        (metric_name, metric_value, timestamp),
    )
    conn.commit()


def query_person_events(conn: sqlite3.Connection, person: str) -> List[dict]:
    """Query all events where person participated."""
    cur = conn.cursor()
    cur.execute(
        """SELECT e.id, e.summary, e.type, er.risk_score FROM events e
           JOIN event_participants ep ON e.id = ep.event_id
           LEFT JOIN event_risk er ON e.id = er.event_id
           WHERE ep.participant = ?
           ORDER BY e.start_time DESC""",
        (person,),
    )
    return [dict(row) for row in cur.fetchall()]
