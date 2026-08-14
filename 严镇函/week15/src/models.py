"""
Pydantic 数据模型定义

定义 Multi-Agent 系统中所有数据结构的类型约束。
这是整个系统的"数据契约"，所有模块都基于这些模型交互。
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """子任务执行状态"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    SUCCESS = "success"      # 成功完成
    FAILED = "failed"        # 执行失败


class SubTask(BaseModel):
    """
    子任务模型

    Planner 拆分后生成的最小执行单元。
    每个 SubTask 会被分配给一个 SubAgent 独立执行。
    """
    task_id: str = Field(..., description="子任务唯一标识")
    description: str = Field(..., description="子任务描述")
    dependencies: list[str] = Field(default_factory=list, description="依赖的其他子任务ID")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="执行状态")
    result: Optional[str] = Field(default=None, description="执行结果")
    error: Optional[str] = Field(default=None, description="错误信息")


class SubTaskResult(BaseModel):
    """
    子任务执行结果

    SubAgent 执行完成后返回的结果包装。
    """
    task_id: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0  # 执行耗时（秒）


class Plan(BaseModel):
    """
    执行计划

    Planner 的输出：一组可以并行/串行执行的子任务列表。
    """
    original_query: str = Field(..., description="原始用户问题")
    subtasks: list[SubTask] = Field(..., description="拆分后的子任务列表")
    reasoning: Optional[str] = Field(default=None, description="Planner 的拆分理由")


class AgentRequest(BaseModel):
    """FastAPI 请求模型"""
    query: str = Field(..., description="用户输入的复杂任务")
    context: Optional[dict[str, Any]] = Field(default=None, description="额外上下文")


class AgentResponse(BaseModel):
    """FastAPI 响应模型"""
    success: bool
    query: str
    plan: Optional[Plan] = None
    results: list[SubTaskResult] = Field(default_factory=list)
    final_answer: Optional[str] = None
    total_time: float = 0.0
    error: Optional[str] = None