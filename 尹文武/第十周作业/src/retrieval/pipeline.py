from typing import List, Dict, Any

from src.retrieval.config import RetrievalConfig
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.rrf import RRFFusion
from src.retrieval.query_transform import QueryTransformer
from src.retrieval.base import deduplicate_results


class RetrievalPipeline:
    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker=None,
        llm_client=None,
        config: RetrievalConfig | None = None,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.config = config or RetrievalConfig()

        self.query_transformer = QueryTransformer(
            llm_client=llm_client
        )

        self.rrf = RRFFusion(
            rrf_k=self.config.rrf_k
        )

    def retrieve(
        self,
        query: str
    ) -> List[Dict[str, Any]]:

        print(f"[QUERY] {query}")
        print(f"[MODE] {self.config.mode}")

        retrieval_queries = [query]

        if self.config.enable_multi_query:
            extra_queries = self.query_transformer.multi_query(
                query,
                self.config.multi_query_num
            )
            print(f"[MULTI QUERY] {extra_queries}")
            retrieval_queries.extend(extra_queries)

        if self.config.enable_hyde:
            hyde_doc = self.query_transformer.hyde(query)
            print(f"[HYDE] {hyde_doc[:100]}...")
            retrieval_queries.append(hyde_doc)

        filters = {}

        if self.config.enable_self_query:
            filters = self.query_transformer.self_query(query)
            print(f"[SELF QUERY FILTER] {filters}")

        all_results = []

        for q in retrieval_queries:
            one_query_results = self._retrieve_single_query(q)
            all_results.extend(one_query_results)

        results = deduplicate_results(all_results)

        results.sort(
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        if filters:
            results = self._apply_filters(results, filters)

        candidate_k = (
            self.config.rerank_candidate_k
            if self.config.enable_rerank
            else self.config.top_k
        )

        results = results[:candidate_k]

        if self.config.enable_rerank and self.reranker:
            results = self.reranker.rerank(
                query=query,
                documents=results,
                top_k=self.config.rerank_top_k,
            )

        return results[:self.config.top_k]

    def _retrieve_single_query(
        self,
        query: str
    ) -> List[Dict[str, Any]]:

        mode = self.config.mode.lower()

        if mode == "dense":
            if not self.dense_retriever:
                raise ValueError("DenseRetriever 未初始化")

            return self.dense_retriever.retrieve(
                query,
                top_k=self.config.dense_k
            )

        if mode == "bm25":
            if not self.bm25_retriever:
                raise ValueError("BM25Retriever 未初始化")

            return self.bm25_retriever.retrieve(
                query,
                top_k=self.config.bm25_k
            )

        if mode == "rrf":
            if not self.dense_retriever or not self.bm25_retriever:
                raise ValueError("RRF 需要 DenseRetriever 和 BM25Retriever")

            dense_results = self.dense_retriever.retrieve(
                query,
                top_k=self.config.dense_k
            )

            bm25_results = self.bm25_retriever.retrieve(
                query,
                top_k=self.config.bm25_k
            )

            return self.rrf.fuse(
                [dense_results, bm25_results]
            )

        raise ValueError(f"未知检索模式：{mode}")

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        output = []

        for item in results:
            ok = True

            book = filters.get("book", "")
            chapter = filters.get("chapter", "")
            keyword = filters.get("keyword", "")

            if book:
                ok = ok and book in str(item.get("book", ""))

            if chapter:
                ok = ok and chapter in str(item.get("chapter", ""))

            if keyword:
                ok = ok and keyword in str(item.get("content", ""))

            if ok:
                output.append(item)

        return output or results
