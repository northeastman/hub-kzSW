from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .llm import AIService
from .models import MemoryHit


class SemanticMemory:
    """Layer 4 persistent semantic memory backed by FAISS cosine search."""

    def __init__(self, directory: Path, dimension: int, ai: AIService) -> None:
        self.directory = directory
        self.dimension = dimension
        self.ai = ai
        self.index_path = directory / "memory.faiss"
        self.records_path = directory / "memory.json"
        self._lock = threading.RLock()
        self._index = faiss.IndexFlatIP(dimension)
        self._records: list[dict[str, Any]] = []
        self._load()

    @property
    def size(self) -> int:
        return len(self._records)

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        vector = await self.ai.embed([text])
        with self._lock:
            self._index.add(np.ascontiguousarray(vector, dtype=np.float32))
            self._records.append({"text": text, "metadata": metadata or {}})
            return len(self._records) - 1

    async def search(
        self, query: str, *, top_k: int = 5, skill: str | None = None
    ) -> list[MemoryHit]:
        if not self._records or top_k <= 0:
            return []
        vector = await self.ai.embed([query])
        # Fetch extra candidates because filtering happens after FAISS search.
        candidate_count = min(len(self._records), max(top_k * 4, top_k))
        with self._lock:
            scores, indices = self._index.search(vector, candidate_count)
            records = list(self._records)
        hits: list[MemoryHit] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = records[int(index)]
            if skill and record["metadata"].get("skill") != skill:
                continue
            hits.append(
                MemoryHit(
                    text=record["text"],
                    score=float(score),
                    metadata=record["metadata"],
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # FAISS's native FileIOWriter cannot reliably open Unicode paths on
            # Windows. Serialize in memory and let pathlib handle the path.
            serialized_index = faiss.serialize_index(self._index)
            temporary_index = self.index_path.with_suffix(".faiss.tmp")
            temporary_index.write_bytes(serialized_index.tobytes())
            temporary_index.replace(self.index_path)

            temporary_records = self.records_path.with_suffix(".json.tmp")
            temporary_records.write_text(
                json.dumps(self._records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_records.replace(self.records_path)

    def _load(self) -> None:
        if not self.index_path.exists() or not self.records_path.exists():
            return
        try:
            serialized_index = np.frombuffer(
                self.index_path.read_bytes(), dtype=np.uint8
            )
            index = faiss.deserialize_index(
                np.ascontiguousarray(serialized_index)
            )
            records = json.loads(self.records_path.read_text(encoding="utf-8"))
            if index.d != self.dimension or index.ntotal != len(records):
                return
            self._index = index
            self._records = records
        except (OSError, ValueError, json.JSONDecodeError):
            # A partial/corrupt persistence pair must not prevent service startup.
            self._index = faiss.IndexFlatIP(self.dimension)
            self._records = []
