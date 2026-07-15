import json
import re
from typing import List, Dict, Any


class QueryTransformer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def multi_query(
        self,
        query: str,
        num: int = 3
    ) -> List[str]:

        if not self.llm_client:
            return []

        prompt = f"""
请将下面的问题改写成 {num} 个不同问法。
要求：
1. 保留原意
2. 每行一个问题
3. 不要解释
4. 不要编号

原问题：{query}
"""

        text = self.llm_client.generate(prompt)

        queries = []

        for line in text.splitlines():
            line = line.strip()
            line = re.sub(r"^[\-\d\.\、\s]+", "", line)

            if line:
                queries.append(line)

        return queries[:num]

    def hyde(self, query: str) -> str:
        if not self.llm_client:
            return query

        prompt = f"""
请根据问题生成一段可能出现在中文学术资料中的文档片段。
不要回答问题，不要解释，只生成适合用于语义检索的假设性资料文本。

问题：{query}
"""

        return self.llm_client.generate(prompt).strip()

    def self_query(
        self,
        query: str
    ) -> Dict[str, Any]:

        if not self.llm_client:
            return {}

        prompt = f"""
请从用户问题中提取检索过滤条件，只输出 JSON。

字段：
- book: 书名，没有则为空字符串
- chapter: 章节名，没有则为空字符串
- keyword: 核心关键词，没有则为空字符串

用户问题：{query}

输出示例：
{{"book":"","chapter":"","keyword":"气韵"}}
"""

        text = self.llm_client.generate(prompt)

        try:
            match = re.search(r"\{.*\}", text, re.S)

            if not match:
                return {}

            return json.loads(match.group())

        except Exception:
            return {}
