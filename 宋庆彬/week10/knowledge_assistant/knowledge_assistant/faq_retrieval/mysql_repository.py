from __future__ import annotations

from logging import Logger

import pandas as pd
import pymysql

from knowledge_assistant.core.settings import MysqlSettings


class MysqlFaqRepository:
    def __init__(self, settings: MysqlSettings, logger: Logger):
        self.settings = settings
        self.logger = logger
        self.connection = pymysql.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            charset="utf8mb4",
        )
        self.cursor = self.connection.cursor()
        self.logger.info("MySQL connected")

    def ensure_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jpkb (
                id INT AUTO_INCREMENT PRIMARY KEY,
                subject_name VARCHAR(50),
                question VARCHAR(255),
                answer TEXT,
                INDEX idx_question (question)
            )
            """
        )
        self.connection.commit()

    def import_csv(self, csv_path: str) -> int:
        data = pd.read_csv(csv_path)
        inserted = 0
        for _, row in data.iterrows():
            self.cursor.execute(
                "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)",
                (row["学科名称"], row["问题"], row["答案"]),
            )
            inserted += 1
        self.connection.commit()
        return inserted

    def fetch_questions(self) -> list[str]:
        self.cursor.execute("SELECT question FROM jpkb")
        return [row[0] for row in self.cursor.fetchall()]

    def fetch_answer(self, question: str) -> str | None:
        self.cursor.execute("SELECT answer FROM jpkb WHERE question=%s LIMIT 1", (question,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self) -> None:
        self.connection.close()
        self.logger.info("MySQL connection closed")

