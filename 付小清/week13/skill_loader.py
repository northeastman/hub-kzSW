"""
Stage 2–3 — Skill 正文与资源渐进加载

- Stage 2 BODY: 按需读取 SKILL.md 正文（frontmatter 已在索引阶段解析）
- Stage 3 RESOURCE: 按 skill 指令或用户意图，按需加载 references/、scripts/ 等附属文件

对应 Cursor create-skill 中的 Progressive Disclosure：
  SKILL.md 放 essentials，reference.md / scripts 仅在需要时 Read。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from skill_registry import SkillMeta, parse_frontmatter


class LoadStage(str, Enum):
    INDEX = "index"          # 仅元数据（registry 负责）
    BODY = "body"            # SKILL.md 正文
    RESOURCE = "resource"    # 附属文件


@dataclass
class LoadEvent:
    stage: LoadStage
    path: Path
    chars: int
    reason: str = ""


@dataclass
class LoadedSkill:
    meta: SkillMeta
    body: str = ""
    resources: dict[str, str] = field(default_factory=dict)
    events: list[LoadEvent] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return self.meta.index_chars + len(self.body) + sum(len(v) for v in self.resources.values())

    def loaded_paths(self) -> list[str]:
        paths = [str(self.meta.skill_md)] if self.body else []
        paths.extend(sorted(self.resources.keys()))
        return paths


# Markdown 链接与常见路径引用
LINK_RE = re.compile(r"\]\(([^)]+\.(?:md|py|ts|json|sh))\)")
PATH_RE = re.compile(
    r"(?:references|scripts)/[\w./\-]+\.(?:md|py|ts|json|sh)|"
    r"\{baseDir\}/(?:references|scripts)/[\w./\-]+\.(?:md|py|ts|json|sh)"
)

# baoyu-diagram：图表类型 → reference 文件
DIAGRAM_TYPE_MAP = {
    "架构": "references/architecture.md",
    "architecture": "references/architecture.md",
    "流程": "references/flowchart.md",
    "flowchart": "references/flowchart.md",
    "时序": "references/sequence.md",
    "sequence": "references/sequence.md",
    "结构": "references/structural.md",
    "structural": "references/structural.md",
    "类图": "references/structural.md",
    "er": "references/structural.md",
}


def extract_linked_resources(body: str) -> list[str]:
    """从 SKILL.md 正文中提取可能需要的资源路径（尚未读取）。"""
    hints: set[str] = set()
    for m in LINK_RE.finditer(body):
        hints.add(m.group(1).split("#")[0])
    for m in PATH_RE.finditer(body):
        p = m.group(0).replace("{baseDir}/", "")
        hints.add(p)
    return sorted(hints)


def infer_diagram_reference(user_query: str) -> str | None:
    q = user_query.lower()
    for keyword, ref in DIAGRAM_TYPE_MAP.items():
        if keyword in q or keyword in user_query:
            return ref
    return None


def load_body(meta: SkillMeta) -> tuple[str, LoadEvent]:
    raw = meta.skill_md.read_text(encoding="utf-8")
    _, body = parse_frontmatter(raw)
    ev = LoadEvent(LoadStage.BODY, meta.skill_md, len(body), "匹配成功后加载 SKILL.md 正文")
    return body, ev


def load_resource(meta: SkillMeta, rel_path: str, reason: str = "") -> tuple[str | None, LoadEvent | None]:
    rel = rel_path.replace("{baseDir}/", "").lstrip("/")
    full = (meta.skill_dir / rel).resolve()
    try:
        full.relative_to(meta.skill_dir.resolve())
    except ValueError:
        return None, None
    if not full.is_file():
        return None, None

    text = full.read_text(encoding="utf-8")
    ev = LoadEvent(LoadStage.RESOURCE, full, len(text), reason or f"按需加载 {rel}")
    return text, ev


class ProgressiveSkillLoader:
    """按需叠加加载 skill 内容，并记录每一层的加载事件。"""

    def __init__(self, meta: SkillMeta):
        self.meta = meta
        self.loaded = LoadedSkill(meta=meta)

    def ensure_body(self) -> str:
        if self.loaded.body:
            return self.loaded.body
        body, ev = load_body(self.meta)
        self.loaded.body = body
        self.loaded.events.append(ev)
        return body

    def load_resources_for_query(self, user_query: str, explicit: list[str] | None = None) -> list[str]:
        """根据用户 query 与 skill 正文中的链接，渐进加载附属资源。"""
        body = self.ensure_body()
        to_load: list[tuple[str, str]] = []

        if explicit:
            for p in explicit:
                to_load.append((p, "调用方指定"))

        if self.meta.name == "baoyu-diagram":
            ref = infer_diagram_reference(user_query)
            if ref:
                to_load.append((ref, f"用户意图含图表类型关键词 → {ref}"))

        for hint in extract_linked_resources(body):
            # 索引阶段不预加载大 reference；仅当 query 命中或 explicit 时再读
            if hint.startswith("references/") and not explicit:
                if infer_diagram_reference(user_query) == hint:
                    to_load.append((hint, "正文链接 + 意图匹配"))
            elif hint.startswith("scripts/"):
                to_load.append((hint, "执行阶段需要脚本路径"))

        loaded_keys: list[str] = []
        seen: set[str] = set()
        for rel, reason in to_load:
            if rel in seen or rel in self.loaded.resources:
                continue
            seen.add(rel)
            text, ev = load_resource(self.meta, rel, reason)
            if text is not None and ev is not None:
                self.loaded.resources[rel] = text
                self.loaded.events.append(ev)
                loaded_keys.append(rel)
        return loaded_keys

    def get_script_path(self, rel: str) -> Path | None:
        rel = rel.replace("{baseDir}/", "").lstrip("/")
        p = self.meta.skill_dir / rel
        return p if p.is_file() else None
