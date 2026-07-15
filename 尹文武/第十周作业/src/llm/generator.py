from typing import List, Dict, Any

from src.llm.ollama_client import OllamaClient


class RAGGenerator:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def build_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_chars: int = 6000,
    ) -> str:
        blocks = []
        current_len = 0

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "").strip()

            if not content:
                continue

            book = doc.get("book", "未知书籍")
            chapter = doc.get("chapter", "")
            start_page = doc.get("start_page", "")
            end_page = doc.get("end_page", "")

            source = (
                f"[资料{i}] "
                f"书名：《{book}》；"
                f"章节：{chapter}；"
                f"页码：{start_page}-{end_page}\n"
                f"{content}"
            )

            if current_len + len(source) > max_chars:
                break

            blocks.append(source)
            current_len += len(source)

        return "\n\n".join(blocks)

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        context = self.build_context(retrieved_docs)

        return f"""
你是一个严谨的中文知识库问答助手。

请严格根据【参考资料】回答用户问题。

要求：
1. 不要编造资料中没有的信息。
2. 如果资料不足，请明确说明“根据当前资料无法确定”。
3. 回答要有条理。
4. 涉及观点、定义、人物评价时，尽量指出出处。
5. 最后给出“参考来源”。

【用户问题】
{query}

【参考资料】
{context}

【回答】
"""

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        prompt = self.build_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
        )

        return self.llm_client.generate(prompt)
