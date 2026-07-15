import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.embedding.embedder import BGEEmbedder


@dataclass
class RetrieverConfig:
    mode: str = "topk"
    top_k: int = 5
    bm25_k: int = 10
    vector_k: int = 10
    rrf_k: int = 60

    enable_multi_query: bool = False
    enable_hyde: bool = False
    enable_self_query: bool = False
    enable_rerank: bool = False

    multi_query_num: int = 3


class Retriever:
    def __init__(
        self,
        faiss_dir: str | Path,
        embedding_model_path: str | Path,
        config: Optional[RetrieverConfig] = None,
        llm_client=None,
        reranker=None,
    ):
        self.faiss_dir = Path(faiss_dir)
        self.config = config or RetrieverConfig()
        self.llm_client = llm_client
        self.reranker = reranker

        self.index = faiss.read_index(
            str(self.faiss_dir / "index.faiss")
        )

        with open(
            self.faiss_dir / "metadata.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.documents = json.load(f)

        self.embedder = BGEEmbedder(
            embedding_model_path
        )

        self.bm25 = self._build_bm25()

    def _build_bm25(self):
        corpus = [
            list(jieba.cut(doc.get("content", "")))
            for doc in self.documents
        ]
        return BM25Okapi(corpus)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        queries = [query]

        if self.config.enable_multi_query:
            queries.extend(
                self._multi_query(query)
            )

        if self.config.enable_hyde:
            hyde_doc = self._hyde(query)
            queries.append(hyde_doc)

        filters = {}

        if self.config.enable_self_query:
            filters = self._self_query(query)

        all_results = []

        for q in queries:
            if self.config.mode == "topk":
                results = self._vector_search(q)

            elif self.config.mode == "bm25":
                results = self._bm25_search(q)

            elif self.config.mode == "rrf":
                vector_results = self._vector_search(
                    q,
                    top_k=self.config.vector_k
                )
                bm25_results = self._bm25_search(
                    q,
                    top_k=self.config.bm25_k
                )
                results = self._rrf_fusion(
                    [vector_results, bm25_results]
                )

            else:
                raise ValueError(
                    f"Unsupported retrieval mode: {self.config.mode}"
                )

            all_results.extend(results)

        results = self._deduplicate(all_results)

        if filters:
            results = self._apply_filters(
                results,
                filters
            )

        results = results[: max(
            self.config.top_k,
            self.config.vector_k,
            self.config.bm25_k
        )]

        if self.config.enable_rerank:
            results = self._rerank(
                query,
                results
            )

        return results[: self.config.top_k]

    def _vector_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.top_k

        query_vec = self.embedder.encode(
            [query],
            batch_size=1
        )

        scores, ids = self.index.search(
            query_vec,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue

            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            doc["retrieval_type"] = "vector"
            doc["doc_id"] = int(idx)

            results.append(doc)

        return results

    def _bm25_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.config.top_k

        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in top_indices:
            doc = self.documents[int(idx)].copy()
            doc["score"] = float(scores[idx])
            doc["retrieval_type"] = "bm25"
            doc["doc_id"] = int(idx)

            results.append(doc)

        return results

    def _rrf_fusion(
        self,
        result_lists: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        scores = {}
        docs = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["doc_id"]

                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    docs[doc_id] = doc

                scores[doc_id] += 1.0 / (
                    self.config.rrf_k + rank + 1
                )

        fused = []

        for doc_id, score in scores.items():
            item = docs[doc_id].copy()
            item["score"] = score
            item["retrieval_type"] = "rrf"
            fused.append(item)

        fused.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return fused

    def _deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []

        for item in results:
            doc_id = item.get("doc_id")

            if doc_id in seen:
                continue

            seen.add(doc_id)
            deduped.append(item)

        deduped.sort(
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return deduped

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        filtered = []

        for item in results:
            ok = True

            if filters.get("book"):
                ok = ok and filters["book"] in item.get("book", "")

            if filters.get("chapter"):
                ok = ok and filters["chapter"] in str(item.get("chapter", ""))

            if filters.get("keyword"):
                ok = ok and filters["keyword"] in item.get("content", "")

            if ok:
                filtered.append(item)

        return filtered or results

    def _multi_query(self, query: str) -> List[str]:
        if not self.llm_client:
            return []

        prompt = f"""
请把下面的问题改写成 {self.config.multi_query_num} 个不同问法。
要求：
1. 保留原意
2. 每行一个问题
3. 不要解释

原问题：{query}
"""

        text = self.llm_client.generate(prompt)

        queries = [
            line.strip(" -0123456789.、")
            for line in text.splitlines()
            if line.strip()
        ]

        return queries[: self.config.multi_query_num]

    def _hyde(self, query: str) -> str:
        if not self.llm_client:
            return query

        prompt = f"""
请根据问题生成一段可能出现在参考资料中的中文学术文本。
不要回答问题，只生成假设性文档片段。

问题：{query}
"""

        return self.llm_client.generate(prompt)

    def _self_query(self, query: str) -> Dict[str, Any]:
        if not self.llm_client:
            return {}

        prompt = f"""
请从用户问题中提取检索过滤条件，只输出JSON。

字段：
- book: 书名，没有则为空字符串
- chapter: 章节名，没有则为空字符串
- keyword: 核心关键词，没有则为空字符串

用户问题：{query}

输出示例：
{{"book":"","chapter":"","keyword":"气韵"}}
"""

        text = self.llm_client.generate(prompt)

        try:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return {}
            return json.loads(match.group())
        except Exception:
            return {}

    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.reranker:
            return results

        pairs = [
            [query, r.get("content", "")]
            for r in results
        ]

        scores = self.reranker.predict(pairs)

        for item, score in zip(results, scores):
            item["rerank_score"] = float(score)

        results.sort(
            key=lambda x: x.get("rerank_score", 0),
            reverse=True
        )

        return results
