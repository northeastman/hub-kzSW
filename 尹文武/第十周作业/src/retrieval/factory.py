from pathlib import Path

from src.retrieval.config import RetrievalConfig
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.reranker import BGEReranker


class RetrieverFactory:
    @staticmethod
    def create(
        faiss_dir: str | Path,
        embedding_model_path: str | Path,
        config: RetrievalConfig,
        llm_client=None,
        reranker_model_path: str | Path | None = None,
    ) -> RetrievalPipeline:

        mode = config.mode.lower()

        dense_retriever = None
        bm25_retriever = None

        if mode in ["dense", "rrf"]:
            dense_retriever = DenseRetriever(
                faiss_dir=faiss_dir,
                embedding_model_path=embedding_model_path,
            )

        if mode in ["bm25", "rrf"]:
            bm25_retriever = BM25Retriever(
                faiss_dir=faiss_dir,
            )

        reranker = None

        if config.enable_rerank:
            if not reranker_model_path:
                raise ValueError("启用 rerank 时必须提供 reranker_model_path")

            reranker = BGEReranker(
                model_path=reranker_model_path,
            )

        return RetrievalPipeline(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            llm_client=llm_client,
            config=config,
        )
