import time
from datetime import datetime
from app.services.weaviate_setup import client, CHATLOG_CLASS

def _season_from_ts(ts: float) -> str:
    """Convert timestamp to season string, e.g. 2025-Q4."""
    dt = datetime.utcfromtimestamp(ts)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"

def log_exchange(*, session_id: str, bot_key: str | None, question: str, answer: str):
    """Store one question/answer pair for admin analytics."""
    try:
        coll = client.collections.get(CHATLOG_CLASS)
        now = int(time.time())
        props = {
            "session_id": session_id or "default",
            "season": _season_from_ts(now),
            "ts": now,
            "bot_key": (bot_key or "").upper()[:8],
            "question": question or "",
            "answer": answer or "",
        }
        coll.data.insert(properties=props)
    except Exception as e:
        print(f"⚠️ admin log failed: {e}")


# view: http://localhost:8000/admin/logs
# filter: http://localhost:8000/admin/logs?season=2025-Q4
# download: http://localhost:8000/admin/logs/export?season=2025-Q4