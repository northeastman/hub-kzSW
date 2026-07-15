from __future__ import annotations

from knowledge_assistant.conversation.history_repository import MysqlConversationHistory
from knowledge_assistant.core.logging import setup_logger
from knowledge_assistant.core.settings import Settings, load_settings
from knowledge_assistant.faq_retrieval.bm25_retriever import FaqBm25Retriever
from knowledge_assistant.faq_retrieval.mysql_repository import MysqlFaqRepository
from knowledge_assistant.faq_retrieval.redis_cache import RedisAnswerCache
from knowledge_assistant.llm.dashscope_client import DashScopeChatClient
from knowledge_assistant.orchestrator.qa_orchestrator import QuestionAnsweringOrchestrator
from knowledge_assistant.rag_retrieval.query_type_classifier import QueryTypeClassifier
from knowledge_assistant.rag_retrieval.rag_pipeline import RagAnswerPipeline
from knowledge_assistant.rag_retrieval.retrieval_strategy import RetrievalStrategySelector
from knowledge_assistant.rag_retrieval.vector_index import MilvusVectorIndex


def build_orchestrator(settings: Settings | None = None) -> QuestionAnsweringOrchestrator:
    settings = settings or load_settings()
    logger = setup_logger(settings.app.log_file)

    faq_repository = MysqlFaqRepository(settings.mysql, logger)
    answer_cache = RedisAnswerCache(settings.redis, logger)
    faq_retriever = FaqBm25Retriever(
        repository=faq_repository,
        cache=answer_cache,
        logger=logger,
        threshold=settings.retrieval.faq_threshold,
    )

    llm = DashScopeChatClient(settings.llm, logger)
    vector_index = MilvusVectorIndex(settings.milvus, settings.models, settings.retrieval, logger)
    classifier = QueryTypeClassifier(settings.models, logger)
    strategy_selector = RetrievalStrategySelector(llm, logger)
    rag_pipeline = RagAnswerPipeline(
        vector_index=vector_index,
        llm=llm,
        classifier=classifier,
        strategy_selector=strategy_selector,
        retrieval_settings=settings.retrieval,
        app_settings=settings.app,
        logger=logger,
    )
    history_repository = MysqlConversationHistory(settings.mysql, logger, settings.app.max_history_turns)
    return QuestionAnsweringOrchestrator(faq_retriever, rag_pipeline, history_repository, logger)

