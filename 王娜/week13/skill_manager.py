"""
Skill Manager — 渐进式三层加载：扫描 → 匹配 → 执行

教学重点：
  1. 启动时扫描 skills/*/SKILL.md，提取 name + description（摘要层）
  2. LLM 判断用户消息是否匹配某个 skill（精准匹配层）
  3. 执行 skill 流水线：LLM 生成数据 → 保存 → 运行脚本 → 收集输出（执行层）

使用方式：
  from src.skill_manager import SkillManager
  mgr = SkillManager()
  skill = mgr.match("给我做张 crazy 的闪卡")  # → {name: "flash-card", ...}
  result = mgr.execute_skill(skill["name"], user_input)
  print(result.output_files)  # → ["crazy.html"]
"""

import os
import sys
import json
import yaml
import subprocess
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_config import get_chat_client

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent.parent / "skills"
WORK_DIR = Path.cwd()  # HTML 输出到当前工作目录


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    skill_name: str
    data_file: str | None = None             # 生成的 JSON / 数据文件路径
    output_files: list[str] = field(default_factory=list)  # 脚本输出的文件
    script_stdout: str = ""                   # 脚本 stdout
    script_stderr: str = ""                   # 脚本 stderr
    final_message: str = ""                   # LLM 生成的友好回复
    steps: list[dict] = field(default_factory=list)       # 每步状态（SSE 广播用）
    error: str | None = None                  # 如果某步失败，记录错误


# ── SkillManager ──────────────────────────────────────────────────────────────

class SkillManager:
    """
    Skill 注册与执行管理器。

    三层加载：
      Level 1: get_summary() → 所有 skill 的 name + description（始终注入 System Prompt）
      Level 2: match() → LLM 精准判断用户消息是否命中某个 skill
      Level 3: execute_skill() → 按 SKILL.md 流程生成数据、跑脚本、收集输出
    """

    def __init__(self, skills_dir: Path = SKILLS_DIR, work_dir: Path = WORK_DIR):
        self.skills_dir = skills_dir
        self.work_dir = work_dir
        self._skills: list[dict] = []
        self._scan()

    # ── 扫描 ──────────────────────────────────────────────────────────────────

    def _scan(self):
        """启动时扫描 skills/*/SKILL.md，解析 YAML frontmatter 和正文。"""
        self._skills = []
        if not self.skills_dir.exists():
            logger.warning(f"Skills 目录不存在：{self.skills_dir}")
            return

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                frontmatter, body = self._parse_skill_md(skill_md)
                name = frontmatter.get("name", skill_dir.name)
                description = frontmatter.get("description", "")
                self._skills.append({
                    "name": name,
                    "description": description,
                    "full_content": body,
                    "base_dir": str(skill_dir),
                    "skill_md_path": str(skill_md),
                    "scripts_dir": str(skill_dir / "scripts") if (skill_dir / "scripts").exists() else None,
                    "data_dir": str(skill_dir / "data") if (skill_dir / "data").exists() else None,
                })
                logger.info(f"[SkillManager] 已注册：{name}")
            except Exception as e:
                logger.warning(f"[SkillManager] 解析 {skill_md} 失败：{e}")

    def _parse_skill_md(self, path: Path) -> tuple[dict, str]:
        """解析 SKILL.md：提取 YAML frontmatter（--- ... ---）和正文。"""
        text = path.read_text(encoding="utf-8")
        # 匹配 frontmatter
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if m:
            fm_text = m.group(1)
            body = m.group(2).strip()
            frontmatter = yaml.safe_load(fm_text) or {}
            return frontmatter, body
        # 无 frontmatter 时，全文作为 body
        return {}, text.strip()

    # ── Level 1: 摘要 ────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        """返回所有 skill 的摘要文本，用于注入 System Prompt（~100 tokens/skill）。"""
        if not self._skills:
            return ""
        lines = ["可用技能列表（当用户请求匹配时请调用对应流程）："]
        for s in self._skills:
            desc = s["description"].replace("\n", " ").strip()
            lines.append(f"- **{s['name']}**：{desc[:200]}")
        return "\n".join(lines)

    def get_skills(self) -> list[dict]:
        """返回已注册的 skill 列表（不含全文，仅元数据）。"""
        return [{"name": s["name"], "description": s["description"]} for s in self._skills]

    def get_skill_info(self, name: str) -> dict | None:
        """获取某个 skill 的完整信息。"""
        for s in self._skills:
            if s["name"] == name:
                return s
        return None

    # ── Level 2: 匹配 ─────────────────────────────────────────────────────────

    def match(self, user_input: str) -> dict | None:
        """
        用 LLM 判断用户消息是否匹配某个 skill。

        返回匹配的 skill 信息（含 full_content），或 None。
        """
        if not self._skills:
            return None

        # 构建轻量匹配 prompt
        skill_list = "\n".join(
            f"- {s['name']}: {s['description'][:120]}"
            for s in self._skills
        )
        prompt = f"""你是一个技能匹配器。根据用户消息判断它是否需要使用以下某个技能。

技能列表：
{skill_list}

用户消息："{user_input}"

请返回纯 JSON（不要带 ```json 标记）：
{{"matched": true/false, "skill": "技能名称或null", "reason": "简短理由"}}

注意：
- 只有用户消息明确请求技能对应的功能时才匹配
- 一般聊天、问候、记忆查询等不匹配任何技能
- 如果没有匹配，skill 字段为 null"""

        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # 确定性匹配
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            # 去除可能的代码块包裹
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            result = json.loads(text)

            if result.get("matched") and result.get("skill"):
                skill_name = result["skill"]
                info = self.get_skill_info(skill_name)
                if info:
                    logger.info(f"[SkillManager] 匹配成功：{skill_name}（理由：{result.get('reason', '')}）")
                    return info
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[SkillManager] 匹配 LLM 调用失败：{e}")

        return None

    # ── Level 3: 执行 ─────────────────────────────────────────────────────────

    def execute_skill(self, skill_name: str, user_input: str) -> SkillResult:
        """
        执行完整的 skill 流水线：
          1. 注入 SKILL.md 让 LLM 按流程生成数据
          2. 解析 LLM 输出，保存数据文件
          3. 运行配套脚本
          4. 收集输出文件
          5. 返回 SkillResult
        """
        info = self.get_skill_info(skill_name)
        if not info:
            return SkillResult(skill_name=skill_name, error=f"Skill '{skill_name}' 未注册")

        result = SkillResult(skill_name=skill_name)

        try:
            # Step 1: LLM 按 SKILL.md 流程执行（第一轮：生成数据）
            step1 = self._llm_generate_data(skill_name, info, user_input, result)
            result.steps.append(step1)
            if step1.get("error"):
                result.error = step1["error"]
                return result

            # Step 2: 保存数据文件
            step2 = self._save_data_file(skill_name, info, result)
            result.steps.append(step2)

            # Step 3: 运行脚本
            step3 = self._run_script(skill_name, info, result)
            result.steps.append(step3)

            # Step 4: LLM 总结（第二轮：生成友好回复）
            step4 = self._llm_summarize(skill_name, info, user_input, result)
            result.steps.append(step4)
            if step4.get("message"):
                result.final_message = step4["message"]

        except Exception as e:
            logger.error(f"[SkillManager] 执行 {skill_name} 失败：{e}")
            result.error = str(e)

        return result

    def _llm_generate_data(
        self, skill_name: str, info: dict, user_input: str, result: SkillResult
    ) -> dict:
        """
        第一轮 LLM 调用：让 LLM 按 SKILL.md 流程生成数据（如 JSON）。
        根据 skill 类型返回不同的指令。
        """
        step = {"step": "llm_generate_data"}

        try:
            if skill_name == "flash-card":
                return self._flashcard_generate(info, user_input, result)
            elif skill_name == "baoyu-diagram":
                return self._diagram_generate(info, user_input, result)
            else:
                # 通用流程：注入 SKILL.md 全文，让 LLM 自行理解并输出
                sys_prompt = f"你负责执行以下技能：\n\n{info['full_content']}"
                client, model = get_chat_client()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_input},
                    ],
                    temperature=0.7,
                )
                step["raw_output"] = resp.choices[0].message.content
                step["status"] = "ok"
        except Exception as e:
            step["error"] = str(e)
            step["status"] = "error"

        return step

    def _flashcard_generate(self, info: dict, user_input: str, result: SkillResult) -> dict:
        """flash-card skill：让 LLM 生成 JSON 数据。"""
        step = {"step": "llm_generate_data", "skill": "flash-card"}

        # 提取单词
        extract_prompt = f"""从用户话语中提取目标英语单词（仅返回小写单词，不要其他内容）。
用户："{user_input}"
单词："""

        client, model = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        word = resp.choices[0].message.content.strip().lower()
        word = re.sub(r"[^a-z]", "", word)  # 只保留字母
        step["word"] = word

        if not word or len(word) < 2:
            step["error"] = f"无法从用户消息中提取有效单词：'{user_input}'"
            return step

        # 生成 JSON 数据
        data_prompt = f"""为英语单词 "{word}" 生成学习数据，返回纯 JSON（不要 ```json 标记），格式：
{{
  "word": "{word}",
  "phonetic": "/音标/",
  "pos": "词性如 adj./n./v.",
  "definition": "中文释义",
  "examples": [
    {{"en": "英文例句1", "zh": "中文翻译1"}},
    {{"en": "英文例句2", "zh": "中文翻译2"}},
    {{"en": "英文例句3", "zh": "中文翻译3"}}
  ],
  "synonyms": ["近义词1", "近义词2", "近义词3", "近义词4", "近义词5"]
}}

要求：
- 例句共恰好3条，每条含en和zh
- 例句地道、长度适中、能体现该词典型用法
- 近义词4-6个，贴近核心含义
- 只返回JSON，不要解释"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": data_prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
            result.data_json = data
            result.word = word
            step["status"] = "ok"
            step["data"] = {k: v for k, v in data.items() if k != "examples"}  # 摘要不含完整例句
            step["examples_count"] = len(data.get("examples", []))
        except json.JSONDecodeError:
            # 尝试从文本中提取第一个 { ... }
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    result.data_json = data
                    result.word = word
                    step["status"] = "ok"
                    step["data"] = {k: v for k, v in data.items() if k != "examples"}
                    step["examples_count"] = len(data.get("examples", []))
                except json.JSONDecodeError:
                    step["error"] = f"LLM 输出的 JSON 无法解析：{text[:200]}"
            else:
                step["error"] = f"LLM 输出中未找到 JSON：{text[:200]}"

        return step

    def _diagram_generate(self, info: dict, user_input: str, result: SkillResult) -> dict:
        """baoyu-diagram skill：让 LLM 按 SKILL.md 生成 SVG。"""
        step = {"step": "llm_generate_data", "skill": "baoyu-diagram"}

        # 注入完整 SKILL.md 作为 system prompt
        sys_prompt = info["full_content"]
        user_msg = f"用户请求：{user_input}\n\n请按照上述 SKILL.md 流程生成 SVG 图表。"

        client, model = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
        text = resp.choices[0].message.content.strip()

        # 提取 SVG 代码
        svg_match = re.search(r"```(?:svg|xml)?\s*\n(.*?)\n```", text, re.DOTALL)
        if svg_match:
            result.data_svg = svg_match.group(1)
        elif "<svg" in text and "</svg>" in text:
            m = re.search(r"(<svg.*?</svg>)", text, re.DOTALL | re.IGNORECASE)
            if m:
                result.data_svg = m.group(1)
        else:
            # 尝试从当前目录或输出中找到 SVG
            result.data_svg = None

        result.data_raw = text
        step["status"] = "ok"
        step["svg_extracted"] = result.data_svg is not None
        return step

    def _save_data_file(self, skill_name: str, info: dict, result: SkillResult) -> dict:
        """Step 2: 保存数据文件到 skill 的 data/ 目录。"""
        step = {"step": "save_data_file"}

        try:
            data_dir = Path(info["data_dir"]) if info["data_dir"] else Path(info["base_dir"]) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            if skill_name == "flash-card" and hasattr(result, "data_json"):
                data = result.data_json
                word = result.word if hasattr(result, "word") else data.get("word", "unknown")
                filepath = data_dir / f"{word}.json"
                filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                result.data_file = str(filepath)
                step["file"] = str(filepath)
                step["status"] = "ok"
            elif skill_name == "baoyu-diagram" and hasattr(result, "data_svg") and result.data_svg:
                # 保存 SVG 到工作目录
                import re as re2
                # 从用户输入生成 slug
                slug = re2.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", 
                               re2.sub(r"[^\w\s-]", "", result.data_raw[:50] or "diagram")).strip("-").lower()[:40]
                if not slug:
                    slug = "diagram"
                svg_path = self.work_dir / f"{slug}.svg"
                svg_path.write_text(result.data_svg, encoding="utf-8")
                result.data_file = str(svg_path)
                result.output_files.append(str(svg_path))
                step["file"] = str(svg_path)
                step["status"] = "ok"
            else:
                step["status"] = "skipped"
                step["reason"] = "无数据文件需要保存"
        except Exception as e:
            step["error"] = str(e)
            step["status"] = "error"

        return step

    def _run_script(self, skill_name: str, info: dict, result: SkillResult) -> dict:
        """Step 3: 运行 skill 配套脚本。"""
        step = {"step": "run_script"}

        try:
            if skill_name == "flash-card" and result.data_file:
                scripts_dir = Path(info["scripts_dir"]) if info["scripts_dir"] else None
                if not scripts_dir:
                    step["status"] = "skipped"
                    step["reason"] = "无 scripts 目录"
                    return step

                script = scripts_dir / "make_flashcard.py"
                if not script.exists():
                    step["status"] = "skipped"
                    step["reason"] = f"脚本不存在：{script}"
                    return step

                # 运行脚本
                html_output = self.work_dir / f"{result.word}.html"
                cmd = [
                    sys.executable, str(script),
                    result.data_file,
                    "-o", str(html_output),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                      cwd=str(self.work_dir))
                result.script_stdout = proc.stdout
                result.script_stderr = proc.stderr

                if proc.returncode == 0:
                    result.output_files.append(str(html_output))
                    step["status"] = "ok"
                    step["output_file"] = str(html_output)
                    step["stdout"] = proc.stdout[:500]
                else:
                    step["status"] = "error"
                    step["error"] = f"脚本退出码 {proc.returncode}：{proc.stderr[:300]}"

                # 自动打开浏览器
                if proc.returncode == 0:
                    try:
                        if sys.platform == "win32":
                            os.startfile(str(html_output))
                        elif sys.platform == "darwin":
                            subprocess.run(["open", str(html_output)])
                        else:
                            subprocess.run(["xdg-open", str(html_output)])
                        step["browser_opened"] = True
                    except Exception:
                        step["browser_opened"] = False

            elif skill_name == "baoyu-diagram":
                # baoyu-diagram 的脚本用 Bun 运行（生成 @2x PNG）
                if result.data_file and result.data_file.endswith(".svg"):
                    scripts_dir = Path(info["scripts_dir"]) if info["scripts_dir"] else None
                    if scripts_dir and (scripts_dir / "main.ts").exists():
                        # 尝试用 bun / npx bun 转换
                        bun_cmd = self._resolve_bun()
                        if bun_cmd:
                            png_path = str(result.data_file).replace(".svg", "@2x.png")
                            proc = subprocess.run(
                                [*bun_cmd, str(scripts_dir / "main.ts"), result.data_file, "-o", png_path],
                                capture_output=True, text=True, timeout=60,
                                cwd=str(self.work_dir),
                            )
                            if proc.returncode == 0 and Path(png_path).exists():
                                result.output_files.append(png_path)
                                step["png_generated"] = True
                                step["png_file"] = png_path
                            else:
                                step["png_skipped"] = True
                                step["png_error"] = proc.stderr[:300] if proc.stderr else ""
                    step["status"] = "ok"
                else:
                    step["status"] = "skipped"
                    step["reason"] = "无 SVG 文件"
            else:
                step["status"] = "skipped"
                step["reason"] = f"Skill '{skill_name}' 无既定脚本"
        except subprocess.TimeoutExpired:
            step["error"] = "脚本执行超时（30秒）"
            step["status"] = "error"
        except Exception as e:
            step["error"] = str(e)
            step["status"] = "error"

        return step

    def _resolve_bun(self) -> list[str] | None:
        """解析 bun 运行时。"""
        # 先检查 bun 是否可用
        try:
            subprocess.run(["bun", "--version"], capture_output=True, timeout=5)
            return ["bun"]
        except Exception:
            pass
        # 再尝试 npx
        try:
            subprocess.run(["npx", "-y", "bun", "--version"], capture_output=True, timeout=10)
            return ["npx", "-y", "bun"]
        except Exception:
            return None

    def _llm_summarize(
        self, skill_name: str, info: dict, user_input: str, result: SkillResult
    ) -> dict:
        """Step 4: 第二轮 LLM 调用，生成友好回复。"""
        step = {"step": "llm_summarize"}

        try:
            if skill_name == "flash-card" and result.output_files:
                file_list = ", ".join(result.output_files)
                summary_prompt = f"""用户请求："{user_input}"
执行结果：已为单词 "{result.word}" 生成闪卡
- 数据文件：{result.data_file}
- HTML 文件：{file_list}

请用友好的语气告诉用户已完成，并简要介绍生成的卡片内容（释义、例句数量等）。
不要使用"好的"、"明白了"开头，直接回复。"""
            elif skill_name == "baoyu-diagram" and result.output_files:
                file_list = ", ".join(result.output_files)
                summary_prompt = f"""用户请求："{user_input}"
执行结果：已生成 SVG 图表
- 文件：{file_list}

请用友好的语气告诉用户图表已生成，简要描述图表类型和内容。
不要使用"好的"、"明白了"开头，直接回复。"""
            else:
                summary_prompt = f"""用户请求："{user_input}"
执行结果：skill "{skill_name}" 执行完毕，输出文件：{result.output_files if result.output_files else '无'}

请用友好的语气告诉用户执行结果。不要使用"好的"、"明白了"开头，直接回复。"""

            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            step["message"] = resp.choices[0].message.content.strip()
            step["status"] = "ok"
        except Exception as e:
            step["error"] = str(e)
            step["message"] = f"已完成 {skill_name} 任务。" + (
                f" 生成文件：{', '.join(result.output_files)}" if result.output_files else ""
            )

        return step


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_skill_manager: SkillManager | None = None


def get_skill_manager() -> SkillManager:
    """获取全局 SkillManager 单例（懒加载）。"""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager
