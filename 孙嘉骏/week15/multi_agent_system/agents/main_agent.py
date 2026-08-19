# agents/main_agent.py
from .base_agent import BaseAgent
from layers.memory import Memory
from tools.registry import global_registry
from tools.create_subagent import set_main_agent_context
from utils.config import MAX_MAIN_AGENT_STEPS, MAX_SUBAGENT_DEPTH

class MainAgent(BaseAgent):
    """主Agent，负责总体任务协调、任务分解、子Agent创建"""
    def __init__(
        self,
        memory: Memory,
        allowed_tools: list = None,
        max_steps: int = MAX_MAIN_AGENT_STEPS,
        depth: int = 0,
        name: str = "MainAgent"
    ):
        system_prompt = (
            "你是一个高级任务规划与执行助手。你可以使用工具来解决用户问题。"
            "如果问题较为复杂，可以分解为多个独立子问题，使用 create_subagent 工具并行创建子Agent来处理。"
            "你也可以使用 plan_tasks 先生成任务图，然后根据任务图创建子Agent。"
            "对于子Agent返回的结果，可以使用 verify_result 工具进行验证。"
            "如果遇到不确定的情况或需要用户确认，可以使用 ask_human 工具。"
            "请按照 ReAct 格式输出：先思考（Thought），然后选择动作（Action），并给出动作输入（Action Input）。"
            "如果任务已完成，输出最终答案（Final Answer）。"
        )
        # 主Agent默认允许所有工具
        if allowed_tools is None:
            allowed_tools = list(global_registry._tools.keys())  # 注册的所有工具
        super().__init__(
            memory=memory,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            max_steps=max_steps,
            depth=depth,
            name=name
        )
        # 设置create_subagent上下文
        set_main_agent_context(
            memory=memory,
            depth=depth,
            max_depth=MAX_SUBAGENT_DEPTH,
            allowed_tools=self.allowed_tools
        )