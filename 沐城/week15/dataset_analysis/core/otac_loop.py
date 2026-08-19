"""
OTAT 循环引擎核心实现
Observe -> Think -> Act -> Check 验证驱动的自主循环
"""
import re
import json
import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .agent_state import AgentState, AgentPhase
from .tool_registry import TOOLS_REGISTRY

logger = logging.getLogger(__name__)


# ReAct 风格输出正则解析模式
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nCheck:|\nFinal Answer:|$)", re.DOTALL)
ACTION_PATTERN = re.compile(r"Action:\s*(\w+)")
ACTION_INPUT_PATTERN = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
CHECK_PATTERN = re.compile(r"Check:\s*(.+?)(?=\nThought:|\nAction:|\nFinal Answer:|$)", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


class OTATLoop:
    """
    OTAT 循环引擎

    执行流程:
    1. Observe: 观察当前状态和环境信息
    2. Think: LLM 推理决定下一步行动
    3. Act: 执行工具调用
    4. Check: 验证结果有效性

    循环直到得出最终答案或达到最大步数
    """

    def __init__(
        self,
        llm_client: Any,
        system_prompt: str,
        tools_registry: Optional[Any] = None,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        """
        初始化 OTAT 循环引擎

        Args:
            llm_client: LLM 客户端实例
            system_prompt: 系统提示词，定义行为规范
            tools_registry: 工具注册表，默认为全局 TOOLS_REGISTRY
            max_steps: 最大循环步数
            verbose: 是否输出详细日志
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.tools = tools_registry or TOOLS_REGISTRY
        self.max_steps = max_steps
        self.verbose = verbose

    def _parse_llm_output(self, text: str) -> Dict[str, Any]:
        """
        解析 LLM 输出文本，提取结构化信息

        Returns:
            包含 thought, action, action_input, check, final_answer 的字典
        """
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "check": None,
            "final_answer": None,
            "raw": text,
        }

        # 检查是否有 Final Answer
        final_match = FINAL_ANSWER_PATTERN.search(text)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # 提取 Thought
        thought_match = THOUGHT_PATTERN.search(text)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取 Action
        action_match = ACTION_PATTERN.search(text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # 提取 Action Input
        input_match = ACTION_INPUT_PATTERN.search(text)
        if input_match:
            try:
                result["action_input"] = json.loads(input_match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"Action Input JSON 解析失败：{e}")
                result["action_input"] = {}

        # 提取 Check
        check_match = CHECK_PATTERN.search(text)
        if check_match:
            result["check"] = check_match.group(1).strip()

        return result

    def _observe(self, state: AgentState, task: str) -> str:
        """
        Observe 阶段：观察环境和状态

        Args:
            state: 当前 Agent 状态
            task: 当前任务描述

        Returns:
            观察结果字符串
        """
        state.phase = AgentPhase.OBSERVE

        # 构建观察信息
        observation_parts = []

        # 添加任务信息
        observation_parts.append(f"当前任务：{task}")

        # 添加历史对话摘要
        if state.messages:
            recent_messages = state.messages[-4:]  # 只看最近 4 条
            for msg in recent_messages:
                observation_parts.append(f"[{msg['role']}]: {msg['content'][:200]}")

        # 添加上一步结果
        if state.observation:
            observation_parts.append(f"上一步观察：{state.observation[:200]}")

        if state.check_result is not None:
            observation_parts.append(f"上一步验证：{'通过' if state.check_result else '失败'}")

        observation = "\n".join(observation_parts)
        state.observation = observation

        if self.verbose:
            logger.info(f"[Observe] {observation[:500]}")

        return observation

    def _think(self, state: AgentState) -> str:
        """
        Think 阶段：LLM 推理决策

        Args:
            state: 当前 Agent 状态

        Returns:
            LLM 输出的原始文本
        """
        state.phase = AgentPhase.THINK

        # 构建消息历史
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # 添加对话历史
        messages.extend(state.messages)

        # 添加当前观察
        messages.append(
            {"role": "user", "content": f"当前状态：\n{state.observation}\n\n请分析下一步行动："}
        )

        if self.verbose:
            logger.info(f"[Think] 发送消息到 LLM, 历史消息数：{len(messages)}")

        # 调用 LLM (适配现有 CustomModel 接口)
        try:
            # 尝试使用 invoke 方法 (CustomModel 接口)
            if hasattr(self.llm_client, "invoke"):
                response = self.llm_client.invoke(
                    prompt="你是一个数据分析助手，使用 OTAT 循环进行分析。",
                    content=messages[-1]["content"]
                )
                llm_output = response.content if response else ""
            else:
                # 尝试标准 chat 接口
                response = self.llm_client.chat(messages)
                llm_output = response.content if hasattr(response, "content") else str(response)

            state.thought = llm_output
            state.add_message("assistant", llm_output)

            if self.verbose:
                logger.info(f"[Think] LLM 输出：{llm_output[:500]}")

            return llm_output

        except Exception as e:
            logger.error(f"[Think] LLM 调用失败：{e}")
            state.error = f"LLM 调用失败：{str(e)}"
            state.phase = AgentPhase.ERROR
            return ""

    def _act(self, state: AgentState, parsed: Dict[str, Any]) -> str:
        """
        Act 阶段：：执行工具调用

        Args:
            state: 当前 Agent 状态
            parsed: 解析后的 LLM 输出

        Returns:
            工具执行结果
        """
        state.phase = AgentPhase.ACT

        if not parsed.get("action"):
            # 没有 Action，可能是 Final Answer 或无法解析
            if parsed.get("final_answer"):
                state.final_answer = parsed["final_answer"]
                state.is_complete = True
                return parsed["final_answer"]
            else:
                state.observation = "无法解析出有效行动，请重新思考"
                return state.observation

        tool_name = parsed["action"]
        tool_input = parsed.get("action_input", {})

        # 查找并调用工具
        tool = self.tools.get(tool_name)
        if not tool:
            available_tools = self.tools.list_tools()
            observation = f"未知工具 '{tool_name}'，可用工具：{available_tools}"
            logger.warning(f"[Act] {observation}")
        else:
            try:
                observation = tool(**tool_input)
                logger.info(f"[Act] 工具 {tool_name} 执行成功")
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
                logger.error(f"[Act] 工具 {tool_name} 执行失败：{e}")

        state.observation = str(observation)
        state.add_message("user", f"Observation: {observation}")

        if self.verbose:
            logger.info(f"[Act] 观察结果：{observation[:500]}")

        return observation

    def _check(self, state: AgentState, parsed: Dict[str, Any]) -> bool:
        """
        Check 阶段：验证结果有效性

        Args:
            state: 当前 Agent 状态
            parsed: 解析后的 LLM 输出

        Returns:
            验证是否通过
        """
        state.phase = AgentPhase.CHECK

        # 检查是否有显式的 Check 内容
        check_content = parsed.get("check", "")

        # 验证逻辑
        check_passed = True
        check_reason = ""

        # 1. 检查观察结果是否有效
        if state.observation and "错误" in state.observation:
            check_passed = False
            check_reason = "观察结果包含错误信息"

        # 2. 检查是否有 Final Answer
        if state.final_answer:
            check_passed = True
            check_reason = "已获得最终答案"

        # 3. 检查显式 Check 内容
        if check_content:
            if "失败" in check_content or "错误" in check_content:
                check_passed = False
                check_reason = check_content

        state.check_result = check_passed
        state.add_message("user", f"Check: {'通过' if check_passed else '失败'} - {check_reason}")

        if self.verbose:
            logger.info(f"[Check] 验证结果：{'通过' if check_passed else '失败'} - {check_reason}")

        return check_passed

    def run(self, task: str, initial_state: Optional[AgentState] = None) -> Generator[Dict[str, Any], None, None]:
        """
        运行 OTAT 循环

        Args:
            task: 任务描述
            initial_state: 初始状态，None 则创建新状态

        Yields:
            每一步的状态字典
        """
        state = initial_state or AgentState(max_steps=self.max_steps)
        state.reset()

        logger.info(f"[OTAT] 开始执行任务：{task}")

        while not state.is_complete and state.step_count < state.max_steps:
            state.step_count += 1

            if self.verbose:
                logger.info(f"[OTAT] ========== 第 {state.step_count} 步 ==========")

            # 1. Observe
            self._observe(state, task)
            yield {
                "step": state.step_count,
                "phase": state.phase.value,
                "observation": state.observation,
            }

            # 2. Think
            llm_output = self._think(state)
            if state.phase == AgentPhase.ERROR:
                yield {
                    "step": state.step_count,
                    "phase": state.phase.value,
                    "error": state.error,
                }
                break

            parsed = self._parse_llm_output(llm_output)
            yield {
                "step": state.step_count,
                "phase": state.phase.value,
                "thought": parsed.get("thought"),
            }

            # 3. Act
            if parsed.get("action"):
                self._act(state, parsed)
                yield {
                    "step": state.step_count,
                    "phase": state.phase.value,
                    "action": parsed.get("action"),
                    "action_input": parsed.get("action_input"),
                    "observation": state.observation,
                }

                if state.is_complete:
                    break

            # 4. Check
            check_passed = self._check(state, parsed)
            yield {
                "step": state.step_count,
                "phase": state.phase.value,
                "check_result": check_passed,
            }

            # 如果验证失败且没有 Final Answer，继续循环
            if not check_passed and not state.final_answer:
                # 可以选择重试或继续
                pass

        # 循环结束
        if not state.is_complete:
            if state.step_count >= state.max_steps:
                state.final_answer = f"已达最大步数 {state.max_steps}，未能得出最终答案"
            else:
                state.is_complete = True
                if not state.final_answer:
                    state.final_answer = "循环终止"

        yield {
            "step": state.step_count,
            "phase": AgentPhase.DONE.value,
            "is_complete": state.is_complete,
            "final_answer": state.final_answer,
            "state": state,
        }

        logger.info(f"[OTAT] 任务执行完成，最终答案：{state.final_answer}")
