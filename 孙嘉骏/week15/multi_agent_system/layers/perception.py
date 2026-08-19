# layers/perception.py
from typing import List, Dict, Any
from .memory import Memory
from utils.config import LONG_TERM_MEMORY_TOP_K

class Perception:
    """感知层：负责构建提示词、检索记忆并注入上下文"""
    def __init__(self, memory: Memory):
        self.memory = memory

    def build_initial_messages(self, system_prompt: str, user_query: str, tool_definitions: List[Dict]) -> List[Dict]:
        """构建初始消息列表"""
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # 检索长期记忆
        memories = self.memory.query_long_term(user_query, top_k=LONG_TERM_MEMORY_TOP_K)
        if memories:
            memory_text = "\n".join([f"- {m['content']}" for m in memories])
            messages.append({
                "role": "system",
                "content": f"相关背景记忆：\n{memory_text}"
            })
        # 添加用户问题
        messages.append({"role": "user", "content": user_query})
        return messages

    def append_tool_result(self, messages: List[Dict], tool_call_id: str, result: str):
        """将工具结果附加到消息列表"""
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })