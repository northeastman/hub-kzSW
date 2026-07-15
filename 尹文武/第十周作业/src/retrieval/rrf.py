from typing import List, Dict, Any


class RRFFusion:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        result_lists: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:

        scores = {}
        docs = {}
        sources = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc["doc_id"]

                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    docs[doc_id] = doc.copy()
                    sources[doc_id] = set()

                scores[doc_id] += 1.0 / (
                    self.rrf_k + rank + 1
                )

                sources[doc_id].add(
                    doc.get("retrieval_type", "unknown")
                )

        fused = []

        for doc_id, score in scores.items():
            item = docs[doc_id].copy()
            item["score"] = float(score)
            item["retrieval_type"] = "rrf"
            item["retrieval_sources"] = list(sources[doc_id])
            fused.append(item)

        fused.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return fused
