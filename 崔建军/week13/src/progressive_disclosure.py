"""
渐进式披露机制 — Progressive Disclosure

核心思想：
  1. Phase 1: 只注入 skill 的 name+description（用于路由匹配）
  2. Phase 2: 匹配成功后，注入完整 SKILL.md（用于执行）
  3. Phase 3: 按需读取详细文档（如有）

这是 Context Window 管理策略，避免一次性注入过多信息。
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DisclosureLevel:
    """披露层级"""
    level: int
    name: str
    content: str = ""


@dataclass
class DisclosureContext:
    """披露上下文"""
    current_level: int = 0
    max_level: int = 3
    conversation_history: List[str] = field(default_factory=list)


class ProgressiveDisclosure:
    """渐进式披露管理器"""
    
    def __init__(self):
        self.context = DisclosureContext()
    
    def get_current_level(self) -> int:
        """获取当前披露层级"""
        return self.context.current_level
    
    def should_disclose(self, level: int) -> bool:
        """判断是否应该披露指定层级"""
        return level <= self.context.current_level
    
    def auto_advance(self) -> int:
        """根据对话历史自动推进层级"""
        history = self.context.conversation_history
        
        # 如果有连续对话，推进层级
        if len(history) >= 3:
            # 检测深入提问词
            deep_keywords = ["详细", "更多", "深入", "解释", "如何", "为什么"]
            # 检查最近3条历史中是否包含深入提问词
            for h in history[-3:]:
                if any(kw in h for kw in deep_keywords):
                    self.context.current_level = min(self.context.current_level + 1, self.context.max_level)
                    break
        
        return self.context.current_level
    
    def update_context(self, user_input: str, response: str):
        """更新披露上下文"""
        self.context.conversation_history.append(user_input)
        self.context.conversation_history.append(response)
        
        # 保留最近 10 条
        if len(self.context.conversation_history) > 20:
            self.context.conversation_history = self.context.conversation_history[-20:]
    
    def format_disclosure(self, skill_name: str, levels: List[DisclosureLevel]) -> str:
        """格式化披露内容"""
        result = []
        for level in levels:
            if self.should_disclose(level.level):
                result.append(f"{'='*60}")
                result.append(f"【Level {level.level}】{level.name}")
                result.append(f"{'='*60}")
                result.append(level.content)
                result.append("")
        return "\n".join(result)


def generate_skill_disclosure(skill, max_level: int = 2) -> List[DisclosureLevel]:
    """为 skill 生成披露层级（按需生成，支持渐进式加载）
    
    Args:
        skill: Skill 对象
        max_level: 最大生成层级
            - 0: 仅基础信息（name + description + keywords），用于路由匹配
            - 1: + 执行流程（actions），用于执行规划
            - 2: + 完整内容（raw_content），用于详细执行和展示
    """
    levels = []
    
    if max_level >= 0:
        level0 = DisclosureLevel(
            level=0,
            name="基础信息",
            content=f"""名称：{skill.name}
版本：{skill.version}
描述：{skill.description}

触发关键词：{', '.join(skill.keywords[:5])}{'...' if len(skill.keywords) > 5 else ''}"""
        )
        levels.append(level0)
    
    if max_level >= 1 and skill.actions:
        actions_text = "\n".join([f"  {a.step}. {a.title}" for a in skill.actions])
        level1 = DisclosureLevel(
            level=1,
            name="执行流程",
            content=f"""执行步骤：
{actions_text}"""
        )
        levels.append(level1)
    
    if max_level >= 2 and skill.raw_content:
        level2 = DisclosureLevel(
            level=2,
            name="完整内容",
            content=skill.raw_content
        )
        levels.append(level2)
    
    return levels
