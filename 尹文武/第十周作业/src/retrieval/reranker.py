from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder


class BGEReranker:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_path = str(Path(model_path))
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"[LOAD RERANKER] {self.model_path}")
        print(f"[RERANKER DEVICE] {self.device}")

        self.model = CrossEncoder(
            self.model_path,
            device=self.device,
            max_length=self.max_length,
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:

        if not documents:
            return []

        print(f"[RERANK] candidates={len(documents)}")

        pairs: List[Tuple[str, str]] = [
            (query, doc.get("content", ""))
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        scores = np.asarray(scores).reshape(-1)

        reranked = []

        for doc, score in zip(documents, scores):
            item = doc.copy()
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def predict(
        self,
        pairs: Sequence[Sequence[str]],
    ):
        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return np.asarray(scores).reshape(-1).tolist()
