from __future__ import annotations

from collections.abc import Iterator
from logging import Logger

from knowledge_assistant.conversation.history_repository import MysqlConversationHistory
from knowledge_assistant.faq_retrieval.bm25_retriever import FaqBm25Retriever
from knowledge_assistant.rag_retrieval.rag_pipeline import RagAnswerPipeline


class QuestionAnsweringOrchestrator:
    def __init__(
        self,
        faq_retriever: FaqBm25Retriever,
        rag_pipeline: RagAnswerPipeline,
        history_repository: MysqlConversationHistory,
        logger: Logger,
    ):
        self.faq_retriever = faq_retriever
        self.rag_pipeline = rag_pipeline
        self.history_repository = history_repository
        self.logger = logger

    def stream_answer(
        self,
        query: str,
        session_id: str,
        source_filter: str | None = None,
    ) -> Iterator[str]:
        answer, need_rag = self.faq_retriever.search(query)
        if answer and not need_rag:
            self.history_repository.append(session_id, query, answer)
            yield answer
            return

        history = self.history_repository.get_recent(session_id)
        collected = []
        for token in self.rag_pipeline.stream_answer(query, source_filter=source_filter, history=history):
            collected.append(token)
            yield token
        self.history_repository.append(session_id, query, "".join(collected))

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return self.history_repository.get_recent(session_id)

    def clear_history(self, session_id: str) -> None:
        self.history_repository.clear(session_id)

