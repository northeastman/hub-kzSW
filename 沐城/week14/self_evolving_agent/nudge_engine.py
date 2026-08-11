"""Nudge Engine - 在线经验捕捉，参考 miniHermes 实现"""
import threading
import logging
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime

from .models import Trajectory, Skill, SkillSource

logger = logging.getLogger(__name__)


class NudgeEngine:
    """
    Nudge 引擎 - 在任务完成后触发反思，沉淀经验为 Skill

    参考 Hermes Agent 的 Nudge 机制：
    - Memory Nudge: 每 10 轮用户对话触发
    - Skill Nudge: 每 10 次 LLM 调用触发
    """

    def __init__(
            self,
            memory_nudge_interval: int = 10,
            skill_nudge_interval: int = 10,
            llm_provider: Optional[Callable] = None,
            skill_saver: Optional[Callable[[Skill], None]] = None,
    ):
        self.memory_nudge_interval = memory_nudge_interval
        self.skill_nudge_interval = skill_nudge_interval
        self.llm_provider = llm_provider  # 用于生成 Skill 的 LLM 调用
        self.skill_saver = skill_saver

        # 计数器
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        self._is_running = False

    def should_nudge_memory(self) -> bool:
        """检查是否应该触发 Memory Nudge"""
        return self._turns_since_memory >= self.memory_nudge_interval

    def should_nudge_skill(self) -> bool:
        """检查是否应该触发 Skill Nudge"""
        return self._iters_since_skill >= self.skill_nudge_interval

    def record_turn(self):
        """记录一轮对话"""
        self._turns_since_memory += 1

    def record_llm_call(self):
        """记录一次 LLM 调用"""
        self._iters_since_skill += 1

    def reset_memory_counter(self):
        """重置 Memory 计数器（反弹保护）"""
        self._turns_since_memory = 0

    def reset_skill_counter(self):
        """重置 Skill 计数器（反弹保护）"""
        self._iters_since_skill = 0

    def try_nudge(self, trajectory: Trajectory) -> Optional[Skill]:
        """
        尝试触发 Nudge

        Args:
            trajectory: 执行轨迹

        Returns:
            生成的 Skill，如果没有触发则返回 None
        """
        # Skill Nudge: 每 N 次 LLM 调用触发
        if self.should_nudge_skill():
            logger.info(f"触发 Skill Nudge (iters_since_skill={self._iters_since_skill})")
            skill = self._generate_skill_from_trajectory(trajectory)
            self.reset_skill_counter()
            return skill

        return None

    def _generate_skill_from_trajectory(self, trajectory: Trajectory) -> Skill:
        """
        从执行轨迹生成 Skill

        参考 Hermes 的 Skill Nudge Prompt:
        "分析对话，识别可复用的操作模式"
        """
        # 构建 Nudge Prompt
        prompt = self._build_nudge_prompt(trajectory)

        # 调用 LLM 生成 Skill
        if self.llm_provider:
            response = self.llm_provider(prompt)
            skill_content = self._parse_skill_response(response)
        else:
            # 无 LLM 时使用模板生成
            skill_content = self._generate_template_skill(trajectory)

        skill = Skill(
            name=f"auto-{trajectory.task_description[:30].replace(' ', '-').lower()}",
            description=f"从任务自动生成的 Skill: {trajectory.task_description[:50]}",
            content=skill_content,
            source=SkillSource.NUDGE,
        )

        if self.skill_saver:
            self.skill_saver(skill)

        return skill

    def _build_nudge_prompt(self, trajectory: Trajectory) -> str:
        """构建 Nudge Prompt"""
        return f"""
你是一个 Skill 提炼专家。请分析以下执行轨迹，提炼出可复用的 Skill。

## 任务描述
{trajectory.task_description}

## 执行过程
共 {trajectory.tool_call_count} 次工具调用
{'✅ 成功' if trajectory.success else '❌ 失败'}
{'⚠️ 有错误但已修复' if trajectory.error_fixed else ''}

## 对话摘要
{self._summarize_messages(trajectory.messages)}

## 工具调用记录
{self._summarize_tool_calls(trajectory.tool_calls)}

## 输出格式
请生成 SKILL.md 文件，包含:
1. name: 技能名称
2. description: 简短描述
3. 核心步骤 (3-5 步)
4. 注意事项 (踩坑经验)

只输出 SKILL.md 内容，不要其他解释。
"""

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """摘要对话"""
        if not messages:
            return "无"
        # 只取关键信息
        summary = []
        for msg in messages[-5:]:  # 只看最后5条
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            summary.append(f"[{role}]: {content}...")
        return "\n".join(summary)

    def _summarize_tool_calls(self, tool_calls: List[Dict]) -> str:
        """摘要工具调用"""
        if not tool_calls:
            return "无"
        summary = []
        for tc in tool_calls[-5:]:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            summary.append(f"- {name}({list(args.keys())})")
        return "\n".join(summary)

    def _parse_skill_response(self, response: str) -> str:
        """解析 LLM 响应，提取 Skill 内容"""
        # 提取 SKILL.md 内容
        if "---" in response:
            # 有 frontmatter，直接返回
            return response
        # 否则包装成 SKILL.md
        return f"""# 自动生成的 Skill

{response}
"""

    def _generate_template_skill(self, trajectory: Trajectory) -> str:
        """无 LLM 时使用模板生成"""
        steps = []
        for i, tc in enumerate(trajectory.tool_calls[:5], 1):
            steps.append(f"{i}. 调用 {tc.get('name', 'unknown')}")

        return f"""# {trajectory.task_description[:50]}

## 描述
此 Skill 从以下任务自动生成: {trajectory.task_description}

## 步骤
{chr(10).join(steps)}

## 注意事项
- 此 Skill 由 Nudge 自动生成，建议人工审核后使用
- 原任务 {'成功' if trajectory.success else '需要改进'}
"""

    def spawn_nudge_async(self, trajectory: Trajectory, callback=None):
        """
        异步触发 Nudge（在独立线程运行，不阻塞用户）
        """

        def _run():
            skill = self.try_nudge(trajectory)
            if callback and skill:
                callback(skill)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
