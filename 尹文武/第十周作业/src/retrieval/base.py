from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        pass


def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []

    for item in results:
        doc_id = item.get("doc_id")

        if doc_id in seen:
            continue

        seen.add(doc_id)
        output.append(item)

    return output
