"""
Skill 加载器 — 解析 SKILL.md 文件

核心功能：
  1. 从 YAML frontmatter 解析 skill 元数据
  2. 解析触发条件（关键词）
  3. 解析执行流程步骤
  4. 支持从目录加载多个 skill
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SkillAction:
    """单个执行步骤"""
    step: int
    title: str
    description: str = ""
    command: Optional[str] = None


@dataclass
class Skill:
    """完整的 Skill 定义（支持懒加载）"""
    name: str
    description: str
    version: str = "1.0.0"
    base_dir: Path = None
    keywords: List[str] = field(default_factory=list)
    actions: List[SkillAction] = field(default_factory=list)
    raw_content: str = ""
    content_loaded: bool = False  # 标记完整内容是否已加载
    
    def match(self, user_input: str) -> bool:
        """检查用户输入是否匹配此 skill"""
        lower_input = user_input.lower()
        for keyword in self.keywords:
            if keyword.lower() in lower_input:
                return True
        return False


class SkillLoader:
    """Skill 加载器"""
    
    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir) if skills_dir else Path(__file__).parent / "skills"
        self.skills: Dict[str, Skill] = {}
    
    def load_all_skills(self) -> List[Skill]:
        """加载所有 skill"""
        self.skills = {}
        if not self.skills_dir.exists():
            return []
        
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill = self._load_skill(skill_dir)
                if skill:
                    self.skills[skill.name] = skill
        return list(self.skills.values())
    
    def _load_skill(self, skill_dir: Path) -> Optional[Skill]:
        """加载单个 skill（仅加载元数据，实现懒加载）
        
        渐进式加载策略：
        - 首次加载：仅解析 frontmatter（name, description, version）和 keywords
        - 执行时：按需加载完整内容和执行流程
        """
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            return None
        
        content = skill_md_path.read_text(encoding="utf-8")
        
        # 解析 YAML frontmatter
        meta = self._parse_frontmatter(content)
        if not meta.get("name"):
            return None
        
        # 首次只加载元数据和关键词（用于匹配阶段）
        skill = Skill(
            name=meta["name"],
            description=meta.get("description", ""),
            version=meta.get("version", "1.0.0"),
            base_dir=skill_dir,
            content_loaded=False  # 标记完整内容未加载
        )
        
        # 优先从 frontmatter 读取关键词，降级从触发场景提取
        skill.keywords = self._parse_keywords(content, meta)
        
        # 暂不解析执行流程，等到执行时再加载
        
        return skill
    
    def _parse_frontmatter(self, content: str) -> Dict[str, str]:
        """解析 YAML frontmatter"""
        frontmatter_pattern = r"^---\n(.*?)\n---"
        match = re.search(frontmatter_pattern, content, re.DOTALL)
        if not match:
            return {}
        
        meta = {}
        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta
    
    def _parse_keywords(self, content: str, meta: Dict[str, str]) -> List[str]:
        """解析触发关键词（优先从 frontmatter 读取，降级从触发场景提取）"""
        keywords = []
        
        # 优先从 frontmatter 读取关键词（支持逗号分隔或 YAML 列表）
        if "keywords" in meta:
            kw_str = meta["keywords"]
            # 支持逗号分隔格式：keywords: "闪卡, flash card, 单词卡"
            if "," in kw_str:
                keywords = [k.strip() for k in kw_str.split(",")]
            # 支持 YAML 列表格式：keywords: ["闪卡", "flash card"]
            elif "[" in kw_str and "]" in kw_str:
                try:
                    import yaml
                    keywords = yaml.safe_load(kw_str)
                    if isinstance(keywords, list):
                        keywords = [str(k).strip() for k in keywords]
                    else:
                        keywords = []
                except ImportError:
                    pass
            return list(set(k for k in keywords if k))
        
        # 降级：从触发场景部分提取
        trigger_section_pattern = r"##\s*(触发场景|Trigger|触发条件)\s*\n([\s\S]*?)(?=\n##\s|$)"
        match = re.search(trigger_section_pattern, content)
        
        if match:
            trigger_text = match.group(2)
            
            # 提取引号中的词
            keywords.extend(re.findall(r'"([^"]+)"', trigger_text))
            
            # 提取示例中的动词
            examples = re.findall(r"-.*?([a-zA-Z\u4e00-\u9fff]+).*?", trigger_text)
            keywords.extend(examples)
        
        return list(set(keywords))
    
    def _parse_actions(self, content: str) -> List[SkillAction]:
        """解析执行流程步骤"""
        actions = []
        action_section_pattern = r"##\s*(执行流程|执行步骤|流程|Steps)\s*\n([\s\S]*?)(?=\n##\s|$)"
        match = re.search(action_section_pattern, content)
        
        if match:
            steps = re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.\s|$)", match.group(2), re.DOTALL)
            for step_num, step_content in steps:
                lines = step_content.strip().split("\n", 1)
                title = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else ""
                
                # 提取命令
                cmd_match = re.search(r"`([^`]+)`", step_content)
                command = cmd_match.group(1) if cmd_match else None
                
                actions.append(SkillAction(
                    step=int(step_num), title=title, description=description, command=command
                ))
        
        return actions
    
    def find_matching_skill(self, user_input: str) -> Optional[Skill]:
        """查找匹配用户输入的 skill"""
        for skill in self.skills.values():
            if skill.match(user_input):
                return skill
        return None
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """根据名称获取 skill"""
        return self.skills.get(name)
    
    def load_full_content(self, skill: Skill) -> bool:
        """按需加载 skill 的完整内容和执行流程
        
        渐进式加载第二阶段：匹配成功后，加载完整 SKILL.md 用于执行
        """
        if skill.content_loaded:
            return True  # 已经加载过，无需重复加载
        
        skill_md_path = skill.base_dir / "SKILL.md"
        if not skill_md_path.exists():
            return False
        
        content = skill_md_path.read_text(encoding="utf-8")
        skill.raw_content = content
        
        # 解析执行流程
        skill.actions = self._parse_actions(content)
        
        # 标记已加载
        skill.content_loaded = True
        
        return True
