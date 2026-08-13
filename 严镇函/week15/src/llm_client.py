"""
LLM 客户端封装

封装 OpenAI-compatible API 调用，提供：
1. 统一的 chat_completion 接口
2. 异步支持（asyncio）
3. 错误处理和重试
4. 日志记录

为什么需要封装？
- 隔离底层 API 细节，上层只关心"发消息拿结果"
- 方便切换不同 LLM 提供商（DeepSeek/OpenAI/智谱等）
- 统一处理超时、重试、限流
"""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI, APIError, RateLimitError

from config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM 异步客户端

    使用 AsyncOpenAI 实现非阻塞调用，
    这样多个 SubAgent 可以同时发起请求而不互相等待。
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.task_timeout,
        )
        self.model = settings.llm_model
        logger.info(f"LLMClient 初始化完成，模型: {self.model}")

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_retries: int = 2,
    ) -> str:
        """
        发起一次异步对话请求

        Args:
            system_prompt: 系统提示词（定义角色和行为）
            user_prompt: 用户提示词（具体任务）
            temperature: 随机性（0=确定，1=创意）
            max_retries: 失败重试次数

        Returns:
            LLM 生成的文本

        Raises:
            RuntimeError: 所有重试都失败时抛出
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"请求 LLM (attempt {attempt + 1}): {user_prompt[:100]}...")

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

                content = response.choices[0].message.content
                logger.debug(f"LLM 响应: {content[:100]}...")
                return content

            except RateLimitError:
                wait_time = 2 ** attempt  # 指数退避
                logger.warning(f"触发限流，等待 {wait_time}s 后重试...")
                await asyncio.sleep(wait_time)

            except APIError as e:
                logger.error(f"API 错误: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"LLM API 调用失败: {e}")
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"未知错误: {e}")
                raise RuntimeError(f"LLM 调用异常: {e}")

        raise RuntimeError("LLM 调用失败，已耗尽重试次数")

    async def chat_with_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """
        请求 LLM 输出 JSON 格式（用于 Planner 生成结构化计划）

        temperature 较低（0.3）是为了让输出更稳定、更可预测。
        """
        # 在 system_prompt 中强调 JSON 输出
        json_system_prompt = (
            f"{system_prompt}\n\n"
            "你必须以纯 JSON 格式输出，不要包含任何其他文字或 markdown 代码块标记。"
        )
        return await self.chat_completion(json_system_prompt, user_prompt, temperature)


# 全局客户端实例（单例模式）
llm_client = LLMClient()