import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.config import RetrievalConfig
from src.retrieval.factory import RetrieverFactory
from src.llm.ollama_client import OllamaClient
from src.llm.generator import RAGGenerator
from src.llm.evaluator import RAGEvaluator
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.eval_pipeline import EvaluationPipeline


FAISS_DIR = ROOT_DIR / "data" / "faiss_index"

EMBEDDING_MODEL = Path(
    r"C:\Users\sean\Downloads\LLM_modules\bge-large-zh-v1.5"
)

RERANKER_MODEL = Path(
    r"C:\Users\sean\Downloads\LLM_modules\bge-reranker-large"
)

QUESTIONS_FILE = ROOT_DIR / "data" / "eval" / "questions.json"
RESULTS_FILE = ROOT_DIR / "data" / "eval" / "results.json"


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    generator_llm = OllamaClient(
        model="qwen2.5:7b",
        temperature=0.2,
    )

    evaluator_llm = OllamaClient(
        model="qwen3:14b",
        temperature=0.0,
    )

    retrieval_config = RetrievalConfig(
        mode="rrf",
        top_k=5,
        dense_k=20,
        bm25_k=20,
        enable_multi_query=False,
        enable_hyde=False,
        enable_self_query=False,
        enable_rerank=True,
        rerank_candidate_k=30,
        rerank_top_k=5,
    )

    retriever = RetrieverFactory.create(
        faiss_dir=FAISS_DIR,
        embedding_model_path=EMBEDDING_MODEL,
        config=retrieval_config,
        llm_client=generator_llm,
        reranker_model_path=RERANKER_MODEL,
    )

    generator = RAGGenerator(
        llm_client=generator_llm,
    )

    rag = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    evaluator = RAGEvaluator(
        llm_client=evaluator_llm,
    )

    eval_pipeline = EvaluationPipeline(
        rag_pipeline=rag,
        evaluator=evaluator,
    )

    results = eval_pipeline.evaluate_batch(questions)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[DONE] saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
