from typing import Optional, Dict, Any
from .storage import init_db, enqueue_task, get_tasks, store_override


def enqueue_for_event(db_path: str, event_id: str, action: str, priority: int = 5, assigned_to: Optional[str] = None, dedup_key: Optional[str] = None, max_per_user_per_day: int = 50):
    conn = init_db(db_path)
    try:
        # enforce per-user daily limit if assigned
        if assigned_to:
            # count existing queued tasks for user today
            tasks = get_tasks(conn, assigned_to=assigned_to, status="queued")
            if len(tasks) >= max_per_user_per_day:
                return {"enqueued": False, "reason": "user workload limit reached"}

        res = enqueue_task(conn, event_id=event_id, action=action, priority=priority, assigned_to=assigned_to, dedup_key=dedup_key)
        return {"enqueued": True, "task": res}
    finally:
        conn.close()


def override_event(db_path: str, event_id: str, user: str, action: str, reason: str = None):
    conn = init_db(db_path)
    try:
        store_override(conn, event_id, user, action, reason)
        return {"overridden": True}
    finally:
        conn.close()
