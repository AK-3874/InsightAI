from typing import Optional
from .storage import init_db, store_feedback


def record_feedback(db_path: str, event_id: str, user: str, label: str, note: Optional[str] = None):
    conn = init_db(db_path)
    store_feedback(conn, event_id, user, label, note)
    conn.close()
