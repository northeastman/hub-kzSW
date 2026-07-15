from __future__ import annotations

from logging import Logger

import numpy as np
from rank_bm25 import BM25Okapi

from knowledge_assistant.faq_retrieval.mysql_repository import MysqlFaqRepository
from knowledge_assistant.faq_retrieval.redis_cache import RedisAnswerCache
from knowledge_assistant.faq_retrieval.text_preprocessor import tokenize_text


class FaqBm25Retriever:
    def __init__(
        self,
        repository: MysqlFaqRepository,
        cache: RedisAnswerCache,
        logger: Logger,
        threshold: float = 0.85,
    ):
        self.repository = repository
        self.cache = cache
        self.logger = logger
        self.threshold = threshold
        self.questions: list[str] = []
        self.tokenized_questions: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.reload()

    def reload(self) -> None:
        cached_questions = self.cache.get_json("faq:questions")
        cached_tokens = self.cache.get_json("faq:tokenized_questions")

        if cached_questions and cached_tokens:
            self.questions = [str(question) for question in cached_questions]
            self.tokenized_questions = cached_tokens
        else:
            self.questions = self.repository.fetch_questions()
            self.tokenized_questions = [tokenize_text(question) for question in self.questions]
            self.cache.set_json("faq:questions", self.questions)
            self.cache.set_json("faq:tokenized_questions", self.tokenized_questions)

        if not self.questions:
            self.logger.warning("No FAQ questions loaded")
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.tokenized_questions)
        self.logger.info(f"FAQ BM25 loaded with {len(self.questions)} questions")

    def search(self, query: str) -> tuple[str | None, bool]:
        if not query or not isinstance(query, str):
            return None, False

        cached_answer = self.cache.get_answer(query)
        if cached_answer:
            return cached_answer, False

        if self.bm25 is None:
            return None, True

        scores = self.bm25.get_scores(tokenize_text(query))
        if len(scores) == 0:
            return None, True

        softmax_scores = self._softmax(scores)
        best_index = int(softmax_scores.argmax())
        best_score = float(softmax_scores[best_index])
        best_question = self.questions[best_index]
        self.logger.info(f"FAQ best score={best_score:.3f}, question={best_question}")

        if best_score >= self.threshold:
            answer = self.repository.fetch_answer(best_question)
            if answer:
                self.cache.set_answer(query, answer)
                return answer, False

        return None, True

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

