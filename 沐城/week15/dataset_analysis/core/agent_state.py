"""
Agent 状态管理类
定义 OTAT 循环中的状态数据结构
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentPhase(Enum):
    """OTAT 循环的四个阶段"""
    OBSERVE = "observe"
    THINK = "think"
    ACT = "act"
    CHECK = "check"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentState:
    """
    Agent 状态类

    Attributes:
        phase: 当前所处的 OTAT 循环阶段
        observation: 观察到的环境信息
        thought: 推理思考内容
        action: 要执行的动作/工具名称
        action_input: 工具调用参数
        check_result: 验证结果
        messages: 对话历史
        step_count: 当前步数
        max_steps: 最大步数限制
        is_complete: 是否完成
        error: 错误信息
    """
    phase: AgentPhase = AgentPhase.OBSERVE
    observation: Optional[str] = None
    thought: str = ""
    action: Optional[str] = None
    action_input: Dict[str, Any] = field(default_factory=dict)
    check_result: Optional[bool] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 10
    is_complete: bool = False
    final_answer: Optional[str] = None
    error: Optional[str] = None

    def reset(self):
        """重置状态到初始值"""
        self.phase = AgentPhase.OBSERVE
        self.observation = None
        self.thought = ""
        self.action = None
        self.action_input = {}
        self.check_result = None
        self.step_count = 0
        self.is_complete = False
        self.final_answer = None
        self.error = None

    def add_message(self, role: str, content: str):
        """添加消息到历史记录"""
        self.messages.append({"role": role, "content": content})

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "phase": self.phase.value,
            "observation": self.observation,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "check_result": self.check_check_result,
            "messages": self.messages,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "is_complete": self.is_complete,
            "final_answer": self.final_answer,
            "error": self.error,
        }
