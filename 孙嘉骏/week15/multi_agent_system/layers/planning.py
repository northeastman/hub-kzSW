# layers/planning.py
from typing import List, Dict, Any
from utils.llm_client import llm_client

class Planning:
    """规划层：封装LLM调用，解析输出"""
    def __init__(self, llm=None):
        self.llm = llm or llm_client

    def call_llm(self, messages: List[Dict], tools: List[Dict] = None) -> Any:
        """调用LLM，返回assistant message对象"""
        return self.llm.chat(messages, tools=tools)

    def has_tool_calls(self, response: Any) -> bool:
        return hasattr(response, "tool_calls") and response.tool_calls is not None and len(response.tool_calls) > 0

    def extract_tool_calls(self, response: Any) -> List[Dict]:
        """从response中提取tool_calls为统一格式"""
        if not self.has_tool_calls(response):
            return []
        return [
            {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in response.tool_calls
        ]

    def extract_content(self, response: Any) -> str:
        return response.content if response.content else ""