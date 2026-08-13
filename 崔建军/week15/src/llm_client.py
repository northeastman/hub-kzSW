"""极简 LLM 客户端（爆款历史人物文章 subagent 项目用）

DeepSeek deepseek-chat，OpenAI 兼容接口。
依赖：pip install openai
"""
import os, time, logging
from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

_client = None

# 进程级 token 用量累计（供前端/日志观测成本）
TOKEN_USAGE = {"prompt": 0, "completion": 0, "calls": 0}


def reset_token_usage():
    TOKEN_USAGE["prompt"] = 0
    TOKEN_USAGE["completion"] = 0
    TOKEN_USAGE["calls"] = 0


def get_token_usage() -> dict:
    return dict(TOKEN_USAGE)


def get_client():
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    """单轮 LLM 对话。stop 用于 ReAct 在 Observation 前截断。"""
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens, stop=stop)
            # 累计 token 用量
            try:
                u = resp.usage
                if u:
                    TOKEN_USAGE["prompt"] += u.prompt_tokens or 0
                    TOKEN_USAGE["completion"] += u.completion_tokens or 0
                    TOKEN_USAGE["calls"] += 1
            except Exception:
                pass
            return resp.choices[0].message.content
        except RateLimitError as e:
            # 429 限流：用更长退避（5/10/15s），避免高峰期反复撞限
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            logger.warning(f"429 限流，等待{wait}s重试: {str(e)[:80]}")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            logger.warning(f"LLM 重试({attempt + 1}): {str(e)[:80]}")
