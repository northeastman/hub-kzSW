from __future__ import annotations

from logging import Logger

import pymysql

from knowledge_assistant.core.settings import MysqlSettings


class MysqlConversationHistory:
    def __init__(self, settings: MysqlSettings, logger: Logger, max_turns: int = 5):
        self.logger = logger
        self.max_turns = max_turns
        self.connection = pymysql.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            charset="utf8mb4",
        )
        self.cursor = self.connection.cursor()
        self.ensure_table()

    def ensure_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_session_id (session_id)
            )
            """
        )
        self.connection.commit()

    def get_recent(self, session_id: str) -> list[dict[str, str]]:
        self.cursor.execute(
            """
            SELECT question, answer
            FROM conversations
            WHERE session_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (session_id, self.max_turns),
        )
        rows = self.cursor.fetchall()
        return [{"question": row[0], "answer": row[1]} for row in rows][::-1]

    def append(self, session_id: str, question: str, answer: str) -> None:
        self.cursor.execute(
            "INSERT INTO conversations (session_id, question, answer) VALUES (%s, %s, %s)",
            (session_id, question, answer),
        )
        self.cursor.execute(
            """
            DELETE FROM conversations
            WHERE session_id = %s AND id NOT IN (
                SELECT id FROM (
                    SELECT id
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                ) recent
            )
            """,
            (session_id, session_id, self.max_turns),
        )
        self.connection.commit()

    def clear(self, session_id: str) -> None:
        self.cursor.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

