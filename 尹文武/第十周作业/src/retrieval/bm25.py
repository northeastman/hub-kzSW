from pathlib import Path
from typing import List, Dict, Any

import json
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        faiss_dir: str | Path,
    ):
        self.faiss_dir = Path(faiss_dir)

        with open(
            self.faiss_dir / "metadata.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.documents = json.load(f)

        corpus = [
            list(jieba.cut(doc.get("content", "")))
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        tokens = list(jieba.cut(query))

        scores = self.bm25.get_scores(tokens)

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for idx in indices:
            idx = int(idx)
            doc = self.documents[idx].copy()

            doc["doc_id"] = idx
            doc["score"] = float(scores[idx])
            doc["retrieval_type"] = "bm25"

            results.append(doc)

        return results
