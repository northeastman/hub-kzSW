"""
会话管理器：内存版多轮对话历史存储

设计要点：
  1. 每个 session_id 对应一段对话历史（user/assistant 消息对）
  2. 历史只保留最终的问答，不存中间 ReAct 步骤（避免上下文爆炸）
  3. 加锁保证线程安全（FastAPI 多请求并发）
  4. 限制每会话最大轮数，自动淘汰最旧的消息
  5. 提供 TTL 自动清理（默认 1 小时无活动即过期）

使用方式：
  from session import session_manager
  sid = session_manager.create_session()
  history = session_manager.get_history(sid)
  session_manager.add_turn(sid, "用户问题", "助手回答")
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional


class SessionManager:
    """线程安全的内存会话管理器"""

    def __init__(
        self,
        max_turns_per_session: int = 10,
        ttl_seconds: int = 3600,
    ):
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_turns = max_turns_per_session
        self._ttl = ttl_seconds

    def create_session(self) -> str:
        """新建一个会话，返回 session_id"""
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self._sessions[sid] = {
                "history": [],
                "created_at": time.time(),
                "last_active": time.time(),
            }
        return sid

    def get_history(self, session_id: str) -> list[dict]:
        """获取会话的对话历史（user/assistant 消息列表）"""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return []
            s["last_active"] = time.time()
            # 返回副本，避免外部修改污染内部状态
            return [dict(m) for m in s["history"]]

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> bool:
        """
        在会话末尾追加一轮问答

        Returns:
            True 如果会话存在并成功追加；False 如果会话不存在
        """
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return False
            s["history"].append({"role": "user", "content": user_msg})
            s["history"].append({"role": "assistant", "content": assistant_msg})
            # 超过最大轮数时，保留最近 N 轮（每轮 = user + assistant 两条消息）
            max_msgs = self._max_turns * 2
            if len(s["history"]) > max_msgs:
                s["history"] = s["history"][-max_msgs:]
            s["last_active"] = time.time()
            return True

    def clear_session(self, session_id: str) -> bool:
        """清空指定会话的所有历史"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> list[dict]:
        """列出所有会话的元信息（不含历史内容）"""
        with self._lock:
            return [
                {
                    "session_id": sid,
                    "turns": len(s["history"]) // 2,
                    "created_at": s["created_at"],
                    "last_active": s["last_active"],
                }
                for sid, s in self._sessions.items()
            ]

    def cleanup_expired(self) -> int:
        """清理所有过期会话，返回清理数量"""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s["last_active"] > self._ttl
            ]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)


# 全局单例，供 serve.py 直接使用
session_manager = SessionManager(max_turns_per_session=10, ttl_seconds=3600)
