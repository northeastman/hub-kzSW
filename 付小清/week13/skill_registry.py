"""
Stage 0 — Skill 索引（仅元数据）

扫描 skills 目录，只读取每个 SKILL.md 的 YAML frontmatter（name + description），
不加载正文。对应 Cursor 在 system prompt 里注入的 skill 摘要列表。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillMeta:
    """Skill 索引条目 — 渐进加载的第一层"""

    name: str
    description: str
    skill_dir: Path
    skill_md: Path
    version: str = ""
    disable_model_invocation: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def index_chars(self) -> int:
        return len(self.name) + len(self.description)

    def to_index_line(self) -> str:
        return f"- **{self.name}**: {self.description[:120]}{'...' if len(self.description) > 120 else ''}"


def _parse_simple_yaml(block: str) -> dict[str, str]:
    """轻量 YAML 解析，足够处理 SKILL.md frontmatter。"""
    data: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal key, buf
        if key is not None:
            data[key] = "\n".join(buf).strip()
        key = None
        buf = []

    for line in block.splitlines():
        if line.startswith("  ") and key is not None:
            buf.append(line.strip())
            continue
        if ":" in line and not line.startswith(" "):
            flush()
            k, v = line.split(":", 1)
            key = k.strip()
            v = v.strip()
            if v in (">-", ">", "|", "|-", "|-"):
                buf = []
            elif v:
                data[key] = v.strip('"').strip("'")
                key = None
            else:
                buf = []
        elif key is not None:
            buf.append(line.strip())
    flush()
    return data


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    return _parse_simple_yaml(m.group(1)), text[m.end() :]


def load_skill_meta(skill_dir: Path) -> SkillMeta | None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None
    raw = skill_md.read_text(encoding="utf-8")
    meta, _ = parse_frontmatter(raw)
    name = meta.get("name", skill_dir.name)
    desc = meta.get("description", "")
    return SkillMeta(
        name=name,
        description=desc,
        skill_dir=skill_dir.resolve(),
        skill_md=skill_md.resolve(),
        version=meta.get("version", ""),
        disable_model_invocation=str(meta.get("disable-model-invocation", "")).lower() == "true",
        extra={k: v for k, v in meta.items() if k not in ("name", "description", "version", "disable-model-invocation")},
    )


def scan_skills_dir(skills_root: Path) -> list[SkillMeta]:
    """递归扫描：每个含 SKILL.md 的子目录视为一个 skill。"""
    skills_root = skills_root.resolve()
    found: list[SkillMeta] = []

    if (skills_root / "SKILL.md").is_file():
        m = load_skill_meta(skills_root)
        if m:
            found.append(m)
        return found

    for child in sorted(skills_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            skill_md = child / "SKILL.md"
            if skill_md.is_file():
                m = load_skill_meta(child)
                if m:
                    found.append(m)
    return found


def build_index_prompt(skills: list[SkillMeta]) -> str:
    """构造仅含元数据的 skill 列表（模拟 Agent 初始可见上下文）。"""
    lines = ["<available_skills>", "The following skills provide specialized capabilities.", ""]
    for s in skills:
        lines.append(f"<skill>")
        lines.append(f"  <name>{s.name}</name>")
        lines.append(f"  <description>{s.description}</description>")
        lines.append(f"</skill>")
        lines.append("")
    lines.append("</available_skills>")
    return "\n".join(lines)
