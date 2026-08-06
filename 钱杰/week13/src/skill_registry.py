"""
skill_registry.py — 4 级渐进式 Skill 加载注册表

设计核心：Skill 不在启动时全量加载，而是按需逐级展开，每级只读取必要内容。

  Level 0 (SCAN)   : 仅扫描目录名          —— 零文件 IO，启动毫秒级
  Level 1 (META)   : 解析 YAML frontmatter —— 只读前几十行，拿 name + description
  Level 2 (FULL)   : 完整 SKILL.md 正文    —— 仅在 skill 被命中时加载
  Level 3 (ASSETS) : 脚本 / references    —— 真正执行时才列举

每级缓存累积，升级时只补差额。一次会话中绝大多数 skill 永远停在 Level 0/1。
"""

from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── 加载级别常量 ────────────────────────────────────────────────────────────
LEVEL_SCAN   = 0  # 仅目录名
LEVEL_META   = 1  # + YAML frontmatter
LEVEL_FULL   = 2  # + SKILL.md 完整正文
LEVEL_ASSETS = 3  # + scripts/references/data 文件清单

_LEVEL_NAMES = {0: "SCAN", 1: "META", 2: "FULL", 3: "ASSETS"}


# ── YAML frontmatter 极简解析（避免引入 pyyaml 依赖）────────────────────────
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL
)
_KV_RE = re.compile(r"^([a-zA-Z_][\w\-]*)\s*:\s*(.*)$")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """返回 (frontmatter_dict, body)。frontmatter 不存在时 dict 为空。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_fm, body = m.group(1), m.group(2)
    fm: dict = {}
    current_key: Optional[str] = None
    for line in raw_fm.splitlines():
        kv = _KV_RE.match(line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            # 处理折叠块标量 >- / >
            if val in (">-", ">", "|-", "|"):
                current_key = key
                fm[key] = ""
                continue
            if val == "":
                current_key = key
                fm[key] = ""
                continue
            fm[key] = val.strip('"').strip("'")
            current_key = None
        elif current_key and line.startswith(("  ", "\t")):
            # 多行值（折叠块标量缩进续行）
            fm[current_key] = (fm[current_key] + " " + line.strip()).strip()
    return fm, body


@dataclass
class SkillEntry:
    """单个 skill 的累积视图，level 表示当前已加载到哪一级。"""
    name: str                       # 目录名（也是 skill 的唯一标识）
    path: Path                      # skill 根目录
    level: int = LEVEL_SCAN         # 当前加载级别
    # Level 1
    display_name: str = ""          # frontmatter.name（缺省回退到目录名）
    description: str = ""           # frontmatter.description
    version: str = ""
    # Level 2
    body: str = ""                  # SKILL.md 正文（去 frontmatter）
    # Level 3
    scripts: list[Path] = field(default_factory=list)
    references: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)

    def to_public_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["level_name"] = _LEVEL_NAMES.get(self.level, "?")
        d["scripts"] = [str(p) for p in self.scripts]
        d["references"] = [str(p) for p in self.references]
        d["data_files"] = [str(p) for p in self.data_files]
        return d


class SkillRegistry:
    """4 级渐进式加载注册表。"""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillEntry] = {}

    # ── Level 0: 扫描 ──────────────────────────────────────────────────
    def scan(self) -> list[str]:
        """扫描 skills_dir 下所有含 SKILL.md 的子目录，返回 skill 名列表。"""
        if not self.skills_dir.exists():
            logger.warning(f"skills 目录不存在：{self.skills_dir}")
            return []
        names = []
        for sub in sorted(self.skills_dir.iterdir()):
            if sub.is_dir() and (sub / "SKILL.md").exists():
                self._skills.setdefault(
                    sub.name, SkillEntry(name=sub.name, path=sub, level=LEVEL_SCAN)
                )
                names.append(sub.name)
        logger.info(f"[Registry] Level 0 SCAN：发现 {len(names)} 个 skill —— {names}")
        return names

    # ── Level 1: 元数据 ────────────────────────────────────────────────
    def load_metadata(self, name: str) -> Optional[SkillEntry]:
        """解析 frontmatter，只读前 2KB 即可。"""
        entry = self._skills.get(name)
        if entry is None:
            return None
        if entry.level >= LEVEL_META:
            return entry
        skill_md = entry.path / "SKILL.md"
        try:
            # 只读前 2KB 足够拿到 frontmatter
            with skill_md.open("r", encoding="utf-8") as f:
                head = f.read(2048)
            fm, _ = _parse_frontmatter(head)
            entry.display_name = fm.get("name", name)
            entry.description = fm.get("description", "")
            entry.version = fm.get("version", "")
            entry.level = LEVEL_META
            logger.info(
                f"[Registry] Level 1 META：{name} —— "
                f"name={entry.display_name}, desc={entry.description[:60]}..."
            )
        except Exception as e:
            logger.error(f"[Registry] 加载 {name} 元数据失败：{e}")
        return entry

    def load_all_metadata(self) -> list[SkillEntry]:
        """批量加载所有 skill 的 Level 1（用于意图匹配前预热）。"""
        for name in list(self._skills.keys()):
            self.load_metadata(name)
        return list(self._skills.values())

    # ── Level 2: 完整正文 ──────────────────────────────────────────────
    def load_full(self, name: str) -> Optional[SkillEntry]:
        """读取完整 SKILL.md，仅在 skill 被命中时调用。"""
        entry = self._skills.get(name)
        if entry is None:
            return None
        if entry.level < LEVEL_META:
            self.load_metadata(name)
        if entry.level >= LEVEL_FULL:
            return entry
        skill_md = entry.path / "SKILL.md"
        try:
            text = skill_md.read_text(encoding="utf-8")
            _, body = _parse_frontmatter(text)
            entry.body = body
            entry.level = LEVEL_FULL
            logger.info(
                f"[Registry] Level 2 FULL：{name} —— body {len(entry.body)} 字符"
            )
        except Exception as e:
            logger.error(f"[Registry] 加载 {name} 完整内容失败：{e}")
        return entry

    # ── Level 3: 资产清单 ──────────────────────────────────────────────
    def load_assets(self, name: str) -> Optional[SkillEntry]:
        """列举 scripts / references / data 目录，真正执行时才调。"""
        entry = self._skills.get(name)
        if entry is None:
            return None
        if entry.level < LEVEL_FULL:
            self.load_full(name)
        if entry.level >= LEVEL_ASSETS:
            return entry
        for sub in ["scripts", "references", "data"]:
            d = entry.path / sub
            if not d.exists():
                continue
            files = [p for p in d.rglob("*") if p.is_file()]
            if sub == "scripts":
                entry.scripts = files
            elif sub == "references":
                entry.references = files
            else:
                entry.data_files = files
        entry.level = LEVEL_ASSETS
        logger.info(
            f"[Registry] Level 3 ASSETS：{name} —— "
            f"scripts={len(entry.scripts)}, refs={len(entry.references)}, data={len(entry.data_files)}"
        )
        return entry

    # ── 查询接口 ───────────────────────────────────────────────────────
    def get(self, name: str) -> Optional[SkillEntry]:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillEntry]:
        return list(self._skills.values())

    def stats(self) -> dict:
        """返回各级别已加载的 skill 数量，用于 UI 展示。"""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for e in self._skills.values():
            counts[e.level] = counts.get(e.level, 0) + 1
        return {
            "total":      len(self._skills),
            "by_level":   {_LEVEL_NAMES[k]: v for k, v in counts.items()},
            "level_names": _LEVEL_NAMES,
        }


# ── CLI 自测 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reg = SkillRegistry(Path(__file__).parent.parent / "skills")
    print("\n=== Level 0 SCAN ===")
    print(reg.scan())
    print("\n=== Level 1 META (all) ===")
    for e in reg.load_all_metadata():
        print(f"  {e.name:20s}  {e.display_name:20s}  {e.description[:50]}")
    print("\n=== Stats ===")
    print(json.dumps(reg.stats(), indent=2, ensure_ascii=False))
