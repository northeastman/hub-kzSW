"""极简 LLM 客户端 —— 供主 agent 与 subagent 共用。

DeepSeek deepseek-chat，OpenAI 兼容接口。线程安全：OpenAI() 客户端可被多线程共享，
因此多个 subagent 并行调用时复用同一个 client 即可。
依赖：pip install openai
"""
import os
import time
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请先设置环境变量 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


def llm_chat(system: str, user: str, *, temperature: float = 0.3,
             max_tokens: int = 1200, retries: int = 3) -> str:
    """单轮对话，返回文本。失败指数退避重试。"""
    last_err = None
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == retries - 1:
                break
            time.sleep(2 ** attempt)
            logger.warning("LLM 重试(%d): %s", attempt + 1, str(e)[:80])
    raise RuntimeError(f"LLM 调用失败: {last_err}")
