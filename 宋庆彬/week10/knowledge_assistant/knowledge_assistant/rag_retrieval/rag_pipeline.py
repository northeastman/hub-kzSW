from __future__ import annotations

from collections.abc import Iterator
from logging import Logger

from knowledge_assistant.core.settings import AppSettings, RetrievalSettings
from knowledge_assistant.llm.dashscope_client import DashScopeChatClient
from knowledge_assistant.rag_retrieval.prompts import (
    build_answer_prompt,
    build_backtracking_prompt,
    build_hyde_prompt,
    build_subquery_prompt,
)
from knowledge_assistant.rag_retrieval.query_type_classifier import QueryTypeClassifier
from knowledge_assistant.rag_retrieval.retrieval_strategy import RetrievalStrategySelector
from knowledge_assistant.rag_retrieval.vector_index import MilvusVectorIndex


class RagAnswerPipeline:
    def __init__(
        self,
        vector_index: MilvusVectorIndex,
        llm: DashScopeChatClient,
        classifier: QueryTypeClassifier,
        strategy_selector: RetrievalStrategySelector,
        retrieval_settings: RetrievalSettings,
        app_settings: AppSettings,
        logger: Logger,
    ):
        self.vector_index = vector_index
        self.llm = llm
        self.classifier = classifier
        self.strategy_selector = strategy_selector
        self.retrieval_settings = retrieval_settings
        self.app_settings = app_settings
        self.logger = logger

    def stream_answer(
        self,
        query: str,
        source_filter: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        query_type = self.classifier.classify(query)
        context = "" if query_type == "general" else self._retrieve_context(query, source_filter)
        prompt = build_answer_prompt(
            context=context,
            history=self._format_history(history or []),
            question=query,
            support_phone=self.app_settings.support_phone,
        )
        prompt = prompt[: self.app_settings.max_prompt_chars]
        yield from self.llm.stream_text(prompt)

    def _retrieve_context(self, query: str, source_filter: str | None) -> str:
        strategy = self.strategy_selector.select(query)
        if strategy == "假设答案检索":
            search_query = self.llm.generate_text(build_hyde_prompt(query)).strip()
            docs = self.vector_index.hybrid_search(search_query, self.retrieval_settings.top_k, source_filter)
        elif strategy == "子问题检索":
            subqueries = [
                line.strip()
                for line in self.llm.generate_text(build_subquery_prompt(query)).splitlines()
                if line.strip()
            ]
            docs = []
            for subquery in subqueries:
                docs.extend(self.vector_index.hybrid_search(subquery, self.retrieval_settings.top_k, source_filter))
            docs = list({doc.page_content: doc for doc in docs}.values())
        elif strategy == "回溯检索":
            search_query = self.llm.generate_text(build_backtracking_prompt(query)).strip()
            docs = self.vector_index.hybrid_search(search_query, self.retrieval_settings.top_k, source_filter)
        else:
            docs = self.vector_index.hybrid_search(query, self.retrieval_settings.top_k, source_filter)

        selected_docs = docs[: self.retrieval_settings.final_context_count]
        self.logger.info(f"RAG context documents selected: {len(selected_docs)}")
        return "\n\n".join(doc.page_content for doc in selected_docs)

    @staticmethod
    def _format_history(history: list[dict[str, str]]) -> str:
        return "\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in history)

