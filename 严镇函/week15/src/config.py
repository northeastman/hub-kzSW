"""
配置管理模块
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str = Field(..., description="LLM API Key")
    llm_base_url: str = Field(default="https://api.deepseek.com/v1", description="LLM API Base URL")
    llm_model: str = Field(default="deepseek-chat", description="LLM 模型名称")
    max_concurrent_tasks: int = Field(default=5, description="最大并发子任务数")
    task_timeout: int = Field(default=60, description="单个任务超时时间（秒）")

    class Config:
        env_file = Path(__file__).parent.parent / ".env"
        env_file_encoding = "utf-8"


settings = Settings()