"""数据模型定义"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class SkillStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"


class SkillSource(str, Enum):
    MANUAL = "manual"
    NUDGE = "nudge"
    GEPA = "gepa"


@dataclass
class Skill:
    """Skill 数据模型"""
    name: str
    description: str
    content: str  # SKILL.md 完整内容
    source: SkillSource = SkillSource.MANUAL
    status: SkillStatus = SkillStatus.ACTIVE
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    success_count: int = 0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_name: Optional[str] = None  # 如果是优化变体，指向父 Skill

    def to_markdown(self) -> str:
        """导出为 SKILL.md 格式"""
        return f"""---
name: {self.name}
description: {self.description}
version: {self.version}
source: {self.source}
---

{self.content}
"""

    @classmethod
    def from_markdown(cls, content: str) -> "Skill":
        """从 SKILL.md 解析"""
        # 简化实现：提取 frontmatter
        lines = content.split("\n")
        name, description = "unknown", ""
        in_frontmatter = False
        body_lines = []

        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip()
            else:
                body_lines.append(line)

        return cls(
            name=name,
            description=description,
            content="\n".join(body_lines).strip()
        )


@dataclass
class Trajectory:
    """执行轨迹 - Nudge 的数据来源"""
    task_description: str
    messages: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    tool_call_count: int = 0
    has_error: bool = False
    error_fixed: bool = False
    has_user_correction: bool = False
    final_result: str = ""
    success: bool = False
    token_usage: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_conversation(cls, conversation: List[Dict[str, str]],
                          tool_call_log: List[Dict]) -> "Trajectory":
        """从对话历史构建轨迹"""
        tool_calls = [t for t in tool_call_log if t.get("type") == "tool_call"]
        has_error = any("error" in str(msg).lower() for msg in conversation)
        error_fixed = has_error and "fixed" in str(conversation[-1]).lower()

        return cls(
            task_description=conversation[0].get("content", "")[:200] if conversation else "",
            messages=conversation,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            has_error=has_error,
            error_fixed=error_fixed,
            has_user_correction=False,  # 需要外部标记
        )


@dataclass
class OptimizationResult:
    """GEPA 优化结果"""
    original_skill: Skill
    optimized_skill: Skill
    improvement_score: float
    metrics: Dict[str, Any]
    generation: int
    timestamp: datetime = field(default_factory=datetime.now)
