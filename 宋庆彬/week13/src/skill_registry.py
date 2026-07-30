"""Skill 的发现、索引和按需加载。

启动阶段只读取 YAML Frontmatter。完整 SKILL.md 以及 references/ 下的资源，
分别在 load_skill 和 read_skill_resource 被调用时才读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SkillError(RuntimeError):
    """Skill 定义或加载失败。"""


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    version: str
    triggers: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    path: Path

    @property
    def directory(self) -> Path:
        return self.path.parent


class SkillRegistry:
    """扫描 skills/*/SKILL.md，并维护轻量元数据索引。"""

    def __init__(
        self,
        skills_root: Path,
        *,
        max_skill_chars: int = 120_000,
        max_resource_chars: int = 80_000,
    ) -> None:
        self.skills_root = skills_root.resolve()
        self.max_skill_chars = max_skill_chars
        self.max_resource_chars = max_resource_chars
        self._skills: dict[str, SkillMeta] = {}
        self.warnings: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self._skills.clear()
        self.warnings.clear()

        if not self.skills_root.exists():
            raise SkillError(f"Skills 目录不存在：{self.skills_root}")

        for skill_file in sorted(self.skills_root.glob("*/SKILL.md")):
            try:
                meta = self._parse_frontmatter(skill_file)
                if meta.name in self._skills:
                    original = self._skills[meta.name].path
                    raise SkillError(
                        f"Skill 名称重复 {meta.name!r}：{original} 与 {skill_file}"
                    )
                self._skills[meta.name] = meta
            except (OSError, SkillError, yaml.YAMLError) as exc:
                self.warnings.append(f"{skill_file}: {exc}")

    def list_skills(self) -> list[SkillMeta]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillMeta:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills)) or "无"
            raise SkillError(f"未知 Skill {name!r}；当前可用：{available}") from exc

    def catalog(self) -> str:
        """返回适合常驻 System Prompt 的紧凑索引。"""
        if not self._skills:
            return "（当前没有可用 Skill）"

        lines: list[str] = []
        for meta in self._skills.values():
            description = " ".join(meta.description.split())
            if len(description) > 240:
                description = description[:237] + "..."
            trigger_text = ""
            if meta.triggers:
                trigger_text = " 触发：" + " / ".join(meta.triggers)
            lines.append(f"- [{meta.name}] {description}{trigger_text}")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str:
        """第二层：按需读取完整 SKILL.md。"""
        meta = self.get(name)
        content = meta.path.read_text(encoding="utf-8")
        if len(content) > self.max_skill_chars:
            raise SkillError(
                f"Skill {name!r} 过大：{len(content)} 字符，"
                f"上限 {self.max_skill_chars}"
            )
        return content

    def read_resource(self, name: str, relative_path: str) -> str:
        """第三层：按需读取当前 Skill 目录内的单个资源。"""
        meta = self.get(name)
        resource = self.resolve_inside(meta.directory, relative_path)
        if not resource.is_file():
            raise SkillError(f"Skill 资源不存在或不是文件：{relative_path}")
        content = resource.read_text(encoding="utf-8")
        if len(content) > self.max_resource_chars:
            raise SkillError(
                f"Skill 资源过大：{len(content)} 字符，"
                f"上限 {self.max_resource_chars}"
            )
        return content

    @staticmethod
    def resolve_inside(root: Path, relative_path: str) -> Path:
        """解析受限相对路径，拒绝绝对路径和目录穿越。"""
        raw = Path(relative_path)
        if raw.is_absolute():
            raise SkillError("只允许使用相对路径")

        root = root.resolve()
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SkillError(f"路径越界：{relative_path}") from exc
        return candidate

    @staticmethod
    def _parse_frontmatter(path: Path) -> SkillMeta:
        """只读文件头，遇到第二个 --- 即停止，不加载 Skill 正文。"""
        frontmatter_lines: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if first.strip() != "---":
                raise SkillError("SKILL.md 必须以 YAML Frontmatter（---）开头")

            for line_number, line in enumerate(handle, start=2):
                if line.strip() == "---":
                    break
                frontmatter_lines.append(line)
                if line_number > 300:
                    raise SkillError("Frontmatter 超过 300 行")
            else:
                raise SkillError("Frontmatter 缺少结束标记 ---")

        raw: Any = yaml.safe_load("".join(frontmatter_lines)) or {}
        if not isinstance(raw, dict):
            raise SkillError("Frontmatter 必须是对象")

        name = str(raw.get("name", "")).strip()
        description = str(raw.get("description", "")).strip()
        if not name:
            raise SkillError("Frontmatter 缺少 name")
        if not description:
            raise SkillError("Frontmatter 缺少 description")

        triggers = SkillRegistry._as_string_tuple(raw.get("triggers", ()))
        allowed_tools = SkillRegistry._as_string_tuple(
            raw.get("allowed_tools", ())
        )
        return SkillMeta(
            name=name,
            description=description,
            version=str(raw.get("version", "1.0.0")).strip(),
            triggers=triggers,
            allowed_tools=allowed_tools,
            path=path.resolve(),
        )

    @staticmethod
    def _as_string_tuple(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value.strip(),) if value.strip() else ()
        if not isinstance(value, (list, tuple)):
            raise SkillError("triggers/allowed_tools 必须是字符串或列表")
        return tuple(str(item).strip() for item in value if str(item).strip())
