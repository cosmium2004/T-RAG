"""
Module 3.4 — Update Tracker for T-RAG.
Logs fact verifications and tracks update history.
Uses a lightweight SQLite database (no PostgreSQL required).
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    source TEXT,
    old_value TEXT,
    new_value TEXT,
    change_type TEXT DEFAULT 'verified',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_update_log_fact ON update_log(fact_id);
CREATE INDEX IF NOT EXISTS idx_update_log_time ON update_log(verified_at);
"""


class UpdateTracker:
    """Tracks fact verification history in SQLite."""

    def __init__(self, db_path: str = "data/cache/update_tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)
        logger.info(f"UpdateTracker ready: {self.db_path}")

    def log_verification(
        self, fact_id: str, source: str = "auto",
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        change_type: str = "verified",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO update_log (fact_id, verified_at, source, "
                "old_value, new_value, change_type) VALUES (?,?,?,?,?,?)",
                (fact_id, now, source, old_value, new_value, change_type),
            )

    def get_history(self, fact_id: str, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM update_log WHERE fact_id=? "
                "ORDER BY verified_at DESC LIMIT ?",
                (fact_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT count(*) FROM update_log").fetchone()[0]
            unique = conn.execute(
                "SELECT count(DISTINCT fact_id) FROM update_log"
            ).fetchone()[0]
            return {"total_entries": total, "unique_facts_tracked": unique}
