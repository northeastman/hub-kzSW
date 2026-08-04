import yaml
from pathlib import Path
from typing import Optional

class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._cache = {}
        self._scan()

    def _scan(self):
        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir():
                continue
            md_file = skill_path / "SKILL.md"
            if not md_file.exists():
                continue
            content = md_file.read_text(encoding='utf-8', errors='replace')
            meta, body = self._parse_frontmatter(content)
            name = meta.get('name', skill_path.name)
            self._cache[name] = {
                'meta': meta,
                'body': body,
                'path': skill_path
            }

    def _parse_frontmatter(self, content: str):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
        else:
            meta = {}
            body = content.strip()
        return meta, body

    def get_descriptions(self) -> str:
        if not self._cache:
            return "No skills available."
        lines = ["## Available Skills"]
        lines.append("You can load a skill by calling `load_skill(skill_name)`.")
        lines.append("")
        for name, skill in self._cache.items():
            desc = skill['meta'].get('description', 'No description')
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    def load_skill_body(self, name: str) -> Optional[str]:
        skill = self._cache.get(name)
        if not skill:
            return None
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"

    def list_skill_names(self) -> list:
        return list(self._cache.keys())

    def get_skill_path(self, name: str) -> Optional[Path]:
        skill = self._cache.get(name)
        return skill['path'] if skill else None