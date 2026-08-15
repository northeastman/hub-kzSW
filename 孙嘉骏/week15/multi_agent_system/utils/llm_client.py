# utils/llm_client.py
import openai
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.client = openai.OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL
        )
        self.model = model or DEEPSEEK_MODEL

    def chat(self, messages, tools=None, tool_choice="auto"):
        """调用 DeepSeek Chat API，返回 assistant message 对象"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,          # 较低温度保证稳定性
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

# 全局单例
llm_client = LLMClient()