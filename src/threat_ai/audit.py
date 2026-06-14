import json
from datetime import datetime
from typing import Any
from .storage import init_db, append_audit


def audit_prediction(db_path: str, event_id: str, action: str, payload: Any):
    conn = init_db(db_path)
    append_audit(conn, event_id=event_id, action=action, payload=json.dumps(payload), timestamp=datetime.utcnow().isoformat())
    conn.close()


def create_snapshot(path: str, pipeline_state: dict):
    # write a JSON snapshot of pipeline inputs/graph/model state for audits
    obj = {
        "timestamp": datetime.utcnow().isoformat(),
        "state": pipeline_state,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
