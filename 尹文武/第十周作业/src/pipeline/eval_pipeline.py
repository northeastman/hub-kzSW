from typing import Dict, Any, List

from src.pipeline.rag_pipeline import RAGPipeline
from src.llm.evaluator import RAGEvaluator


class EvaluationPipeline:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        evaluator: RAGEvaluator,
    ):
        self.rag_pipeline = rag_pipeline
        self.evaluator = evaluator

    def evaluate_one(self, query: str) -> Dict[str, Any]:
        rag_result = self.rag_pipeline.ask(query)

        eval_result = self.evaluator.evaluate(
            query=query,
            answer=rag_result["answer"],
            retrieved_docs=rag_result["retrieved_docs"],
        )

        return {
            "query": query,
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "evaluation": eval_result,
        }

    def evaluate_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        results = []

        for i, q in enumerate(questions, 1):
            print(f"\n[EVAL] {i}/{len(questions)}: {q}")
            results.append(self.evaluate_one(q))

        return results
