# agents/base_agent.py
from typing import List, Dict, Any, Optional
from layers.memory import Memory
from layers.perception import Perception
from layers.execution import Execution
from layers.planning import Planning
from tools.registry import global_registry

class BaseAgent:
    """所有Agent的基类，实现ReAct循环"""
    def __init__(
        self,
        memory: Memory,
        system_prompt: str,
        allowed_tools: List[str] = None,
        max_steps: int = 5,
        depth: int = 0,
        name: str = "Agent"
    ):
        self.memory = memory
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools if allowed_tools is not None else []  # 空列表表示仅基础工具
        self.max_steps = max_steps
        self.depth = depth
        self.name = name
        self.perception = Perception(memory)
        self.execution = Execution()
        self.planning = Planning()
        self.messages: List[Dict] = []

    def _get_tool_definitions(self) -> List[Dict]:
        """获取本Agent可用的工具定义"""
        return global_registry.get_tool_definitions(self.allowed_tools)

    def run(self, user_input: str) -> str:
        """运行ReAct循环，返回最终答案"""
        # 构建初始消息
        self.messages = self.perception.build_initial_messages(
            system_prompt=self.system_prompt,
            user_query=user_input,
            tool_definitions=self._get_tool_definitions()
        )
        # 循环
        for step in range(self.max_steps):
            # 调用LLM
            response = self.planning.call_llm(self.messages, tools=self._get_tool_definitions())
            
            # 将assistant消息加入历史
            assistant_msg = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.tool_calls
                ]
            self.messages.append(assistant_msg)

            # 检查是否有工具调用
            if response.tool_calls:
                tool_calls = self.planning.extract_tool_calls(response)
                # 执行工具（并行）
                results = self.execution.execute_tool_calls(tool_calls, self.allowed_tools)
                # 将结果加入消息
                for res in results:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": res["tool_call_id"],
                        "content": res["content"]
                    })
                # 继续循环
                continue
            else:
                # 没有工具调用，返回最终答案
                return response.content or "任务已完成（无内容）"
        # 达到最大步数
        return "任务未完成，达到最大步数限制。"