"""
Layer 5 能力层：扫描并加载 memory/skills/*/SKILL.md

教学重点：
  1. Markdown 作为"能力配置语言"：与 SOUL/USER/MEMORY 同源，人类可读可编辑
  2. 渐进式披露：list_skills() 只给 name+description（轻量，注入 system prompt），
     load_skill() 才读完整正文（命中后才付出 token 成本）
  3. frontmatter 用行级解析，不引入 yaml 依赖（对齐 heartbeat_parser 风格）

使用方式：
  from src.skill_loader import SkillLoader
  loader = SkillLoader()
  for s in loader.list_skills():
      print(s["name"], s["description"])
  body = loader.load_skill("weekly-report")
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent.parent / "memory" / "skills"


class SkillLoader:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir

    def _parse_skill_file(self, skill_md: Path) -> dict | None:
        """解析单个 SKILL.md，返回 {name, description, run, body}；格式错返回 None。"""
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"读取 {skill_md} 失败：{e}")
            return None

        if not text.lstrip().startswith("---"):
            logger.warning(f"{skill_md} 缺少 frontmatter，跳过")
            return None

        stripped = text.lstrip()
        rest = stripped[len("---"):]
        end = rest.find("\n---")
        if end == -1:
            logger.warning(f"{skill_md} frontmatter 未闭合，跳过")
            return None
        fm_block = rest[:end]
        body = rest[end + len("\n---"):].lstrip("\n")

        meta = {"name": None, "description": "", "run": None}
        for line in fm_block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

        if not meta.get("name"):
            logger.warning(f"{skill_md} frontmatter 缺 name，跳过")
            return None
        meta["run"] = meta.get("run") or None
        meta["body"] = body
        return meta

    def _iter_skill_metas(self):
        """遍历所有合法 skill 的 meta（含 body）。"""
        if not self.skills_dir.exists():
            return
        for sub in sorted(self.skills_dir.iterdir()):
            if not sub.is_dir():
                continue
            skill_md = sub / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = self._parse_skill_file(skill_md)
            if meta:
                meta["dir"] = sub
                yield meta

    def list_skills(self) -> list[dict]:
        """返回所有 skill 的 name+description+has_script（轻量，用于注入 system prompt）。"""
        result = []
        for meta in self._iter_skill_metas():
            result.append({
                "name": meta["name"],
                "description": meta["description"],
                "has_script": meta["run"] is not None,
            })
        return result

    def get_skill_meta(self, name: str) -> dict | None:
        """返回指定 skill 的完整 meta：{name, description, run, dir}；不存在返回 None。"""
        for meta in self._iter_skill_metas():
            if meta["name"] == name:
                return {"name": meta["name"], "description": meta["description"],
                        "run": meta["run"], "dir": meta["dir"]}
        return None

    def load_skill(self, name: str) -> str | None:
        """返回指定 skill 的正文（已剥离 frontmatter）；不存在返回 None。"""
        for meta in self._iter_skill_metas():
            if meta["name"] == name:
                return meta["body"]
        return None
