"""
memory_store.py — Skill 使用记录持久化（SQLite）

记录每次 skill 触发与执行结果，用于：
  1. 统计哪些 skill 最常用
  2. 回看历史执行记录
  3. 为后续"基于历史的 skill 推荐"留接口

表结构：
  skill_usage(id, skill_name, user_input, status, result_summary, duration_ms, ts)
"""

from __future__ import annotations
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name      TEXT NOT NULL,
    user_input      TEXT NOT NULL,
    status          TEXT NOT NULL,           -- running / success / failed
    result_summary  TEXT,
    duration_ms     INTEGER,
    ts              TEXT NOT NULL            -- ISO 时间戳
);
CREATE INDEX IF NOT EXISTS idx_skill_name ON skill_usage(skill_name);
CREATE INDEX IF NOT EXISTS idx_ts          ON skill_usage(ts);
"""


class MemoryStore:
    def __init__(self, db_path: str | Path = "outputs/skill_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def record_start(self, skill_name: str, user_input: str) -> int:
        """记录一次 skill 开始执行，返回 id。"""
        ts = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO skill_usage(skill_name, user_input, status, ts) "
                "VALUES(?, ?, 'running', ?)",
                (skill_name, user_input, ts),
            )
            conn.commit()
            return cur.lastrowid

    def record_finish(
        self,
        usage_id: int,
        status: str,
        result_summary: str = "",
        duration_ms: int = 0,
    ):
        """更新执行结果。status: success / failed"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE skill_usage SET status=?, result_summary=?, duration_ms=? "
                "WHERE id=?",
                (status, result_summary, duration_ms, usage_id),
            )
            conn.commit()

    def recent_usage(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM skill_usage ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def skill_stats(self) -> list[dict]:
        """每个 skill 的总触发次数、成功次数、最近一次使用时间。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT skill_name, "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS success, "
                "  MAX(ts) AS last_used "
                "FROM skill_usage GROUP BY skill_name ORDER BY total DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def reset(self):
        """清空所有使用记录（保留 schema）。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM skill_usage")
            conn.commit()
        logger.info("[MemoryStore] 已清空所有使用记录")
