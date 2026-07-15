from __future__ import annotations

from logging import Logger

from knowledge_assistant.llm.dashscope_client import DashScopeChatClient
from knowledge_assistant.rag_retrieval.prompts import build_strategy_prompt


class RetrievalStrategySelector:
    allowed_strategies = ("直接检索", "假设答案检索", "子问题检索", "回溯检索")

    def __init__(self, llm: DashScopeChatClient, logger: Logger):
        self.llm = llm
        self.logger = logger

    def select(self, query: str) -> str:
        try:
            raw_strategy = self.llm.generate_text(build_strategy_prompt(query)).strip()
        except Exception:
            return "直接检索"

        for strategy in self.allowed_strategies:
            if strategy in raw_strategy:
                self.logger.info(f"Retrieval strategy selected: {strategy}")
                return strategy
        self.logger.warning(f"Unexpected retrieval strategy '{raw_strategy}', fallback to direct search")
        return "直接检索"

