from pathlib import Path
from typing import List, Dict, Any

import faiss
import json

from src.embedding.embedder import BGEEmbedder
from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        faiss_dir: str | Path,
        embedding_model_path: str | Path,
    ):
        self.faiss_dir = Path(faiss_dir)

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

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

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

            doc = self.documents[int(idx)].copy()
            doc["doc_id"] = int(idx)
            doc["score"] = float(score)
            doc["retrieval_type"] = "dense"

            results.append(doc)

        return results
