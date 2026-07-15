from typing import Dict, Any, List

from src.retrieval.pipeline import RetrievalPipeline
from src.llm.generator import RAGGenerator


class RAGPipeline:
    def __init__(
        self,
        retriever: RetrievalPipeline,
        generator: RAGGenerator,
    ):
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str) -> Dict[str, Any]:
        retrieved_docs = self.retriever.retrieve(query)

        answer = self.generator.generate(
            query=query,
            retrieved_docs=retrieved_docs,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": self._format_sources(retrieved_docs),
            "retrieved_docs": retrieved_docs,
        }

    def _format_sources(
        self,
        docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sources = []

        for doc in docs:
            sources.append(
                {
                    "book": doc.get("book"),
                    "chapter": doc.get("chapter"),
                    "start_page": doc.get("start_page"),
                    "end_page": doc.get("end_page"),
                    "score": doc.get("score"),
                    "rerank_score": doc.get("rerank_score"),
                    "retrieval_type": doc.get("retrieval_type"),
                }
            )

        return sources
