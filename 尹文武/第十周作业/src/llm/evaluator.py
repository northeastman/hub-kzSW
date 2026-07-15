import json
import re
from typing import List, Dict, Any

from src.llm.ollama_client import OllamaClient


class RAGEvaluator:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def build_context(self, retrieved_docs: List[Dict[str, Any]], max_chars: int = 6000) -> str:
        blocks = []

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            block = (
                f"[资料{i}]\n"
                f"书名：{doc.get('book')}\n"
                f"章节：{doc.get('chapter')}\n"
                f"页码：{doc.get('start_page')}-{doc.get('end_page')}\n"
                f"内容：{content}"
            )

            blocks.append(block)

        context = "\n\n".join(blocks)
        return context[:max_chars]

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = self.build_context(retrieved_docs)

        prompt = f"""
你是一个严格的 RAG 系统评估员。请基于【问题】、【检索资料】和【模型回答】进行多维度评分。

评分范围均为 0-10 分，10 分最好。

请只输出 JSON，不要输出额外解释。

评分维度：
1. relevance：回答是否切中问题
2. faithfulness：回答是否忠实于检索资料，是否存在幻觉
3. completeness：回答是否完整覆盖问题
4. citation_quality：是否合理利用来源信息
5. clarity：表达是否清晰、有条理
6. retrieval_quality：检索资料是否支持回答
7. overall：综合评分

输出格式：
{{
  "relevance": 0,
  "faithfulness": 0,
  "completeness": 0,
  "citation_quality": 0,
  "clarity": 0,
  "retrieval_quality": 0,
  "overall": 0,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "suggestion": "改进建议"
}}

【问题】
{query}

【检索资料】
{context}

【模型回答】
{answer}
"""

        raw = self.llm_client.generate(prompt)
        return self._parse_json(raw)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {
                "parse_error": True,
                "raw_output": text,
            }

        try:
            return json.loads(match.group())
        except Exception:
            return {
                "parse_error": True,
                "raw_output": text,
            }
