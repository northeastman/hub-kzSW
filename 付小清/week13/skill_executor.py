"""
Stage 4 — Skill 执行

在 skill 正文与必要资源已加载后：
  - flash-card: 提取单词 → 生成/复用 JSON → 调用 scripts/make_flashcard.py
  - baoyu-diagram: 加载类型 reference →（有 LLM 时）生成 SVG；无 LLM 时输出加载报告
  - 通用: LLM 按 skill 指令执行，可调用 run_script / write_file 工具
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from skill_loader import LoadedSkill, ProgressiveSkillLoader
from skill_registry import SkillMeta


@dataclass
class ExecStep:
    action: str
    detail: str
    ok: bool = True


@dataclass
class ExecResult:
    skill_name: str
    steps: list[ExecStep] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    message: str = ""

    def add(self, action: str, detail: str, ok: bool = True) -> None:
        self.steps.append(ExecStep(action, detail, ok))


WORD_RE = re.compile(r"\b([a-zA-Z]{3,})\b")


def extract_english_word(query: str) -> str | None:
    """从「给我做张 crazy 词的闪卡」类语句中提取目标单词。"""
    # 优先：英文 X 词 / X 单词（限制 ASCII，避免「单词」误捕获「单」）
    m = re.search(r"\b([a-zA-Z][a-zA-Z\-]*)\s*单词", query, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([a-zA-Z][a-zA-Z\-]*)\s*词", query, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"(?:做|生成|制作)\s*([a-zA-Z][a-zA-Z\-]*)", query, re.I)
    if m:
        return m.group(1).lower()
    # 最后一个英文词
    words = WORD_RE.findall(query)
    stop = {"flash", "card", "make", "the", "for", "word"}
    candidates = [w.lower() for w in words if w.lower() not in stop]
    return candidates[-1] if candidates else None


def _run_python_script(script: Path, args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or Path.cwd()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def execute_flash_card(
    meta: SkillMeta,
    loader: ProgressiveSkillLoader,
    query: str,
    output_dir: Path,
    *,
    use_llm: bool = False,
) -> ExecResult:
    result = ExecResult(skill_name=meta.name)
    word = extract_english_word(query)
    if not word:
        result.add("extract_word", "未能从 query 提取英文单词", ok=False)
        result.message = "请指定单词，例如：给我做张 resilient 词的闪卡"
        return result
    result.add("extract_word", f"目标单词: {word}")

    data_dir = meta.skill_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / f"{word}.json"

    if json_path.is_file():
        result.add("reuse_json", f"复用已有数据 {json_path}")
    elif use_llm:
        ok, msg = _llm_generate_flashcard_json(word, json_path)
        result.add("generate_json", msg, ok=ok)
        if not ok:
            result.message = msg
            return result
    else:
        # 离线：尝试从 work13/fixtures 或复制示例结构
        fixture = Path(__file__).parent / "fixtures" / f"{word}.json"
        if fixture.is_file():
            json_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            result.add("copy_fixture", f"使用 fixtures/{word}.json")
        else:
            result.add(
                "generate_json",
                f"无 API 且无 {word}.json 数据，跳过生成（可设置 DASHSCOPE_API_KEY 或放入 fixtures/）",
                ok=False,
            )
            result.message = f"缺少 {word} 的学习数据"
            return result

    script = loader.get_script_path("scripts/make_flashcard.py")
    if not script:
        result.add("run_script", "未找到 scripts/make_flashcard.py", ok=False)
        return result

    loader.load_resources_for_query(query, explicit=["scripts/make_flashcard.py"])
    out_html = output_dir / f"{word}.html"
    code, out = _run_python_script(script, [str(json_path), "-o", str(out_html)], cwd=output_dir)
    result.add("run_script", f"make_flashcard.py → {out_html}\n{out}")
    if code != 0:
        result.steps[-1].ok = False
        result.message = out
        return result

    result.outputs.append(out_html)
    result.message = f"已生成闪卡: {out_html}"
    return result


def _llm_generate_flashcard_json(word: str, out_path: Path) -> tuple[bool, str]:
    try:
        from openai import OpenAI
    except ImportError:
        return False, "需要 openai 包"

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "未设置 DASHSCOPE_API_KEY"

    base_url = os.environ.get(
        "AGENT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = os.environ.get("AGENT_MODEL", "qwen-plus")
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""为英语单词 "{word}" 生成闪卡 JSON，字段：
word, phonetic, pos, definition（中文）, examples（恰好3条，含en/zh）, synonyms（4-6个）
只输出 JSON，无 markdown 包裹。"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = resp.choices[0].message.content or ""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"LLM JSON 解析失败: {e}"

    data["word"] = word
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, f"LLM 生成 {out_path}"


def execute_baoyu_diagram(
    meta: SkillMeta,
    loader: ProgressiveSkillLoader,
    query: str,
    output_dir: Path,
    *,
    use_llm: bool = False,
) -> ExecResult:
    result = ExecResult(skill_name=meta.name)
    loader.ensure_body()
    refs = loader.load_resources_for_query(query)
    all_refs = list(loader.loaded.resources.keys())
    if all_refs:
        result.add("load_references", f"按需加载: {', '.join(all_refs)}")
    elif refs:
        result.add("load_references", f"按需加载: {', '.join(refs)}")
    else:
        result.add("load_references", "未命中图表类型关键词，仅加载 SKILL.md 正文")

    if not use_llm:
        result.message = (
            "baoyu-diagram 需 LLM 生成 SVG；当前为离线模式，已完成渐进加载演示。"
            f" 已加载资源: {loader.loaded.loaded_paths()}"
        )
        return result

    # LLM 生成简化 SVG（教学演示，非完整 baoyu 流程）
    svg_path = output_dir / "demo-diagram.svg"
    ok, msg = _llm_generate_simple_svg(query, loader.loaded, svg_path)
    result.add("llm_generate_svg", msg, ok=ok)
    if ok:
        result.outputs.append(svg_path)
        result.message = f"已生成示意图: {svg_path}"
    else:
        result.message = msg
    return result


def _llm_generate_simple_svg(query: str, loaded: LoadedSkill, out_path: Path) -> tuple[bool, str]:
    try:
        from openai import OpenAI
    except ImportError:
        return False, "需要 openai 包"

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "未设置 DASHSCOPE_API_KEY"

    base_url = os.environ.get(
        "AGENT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = os.environ.get("AGENT_MODEL", "qwen-plus")
    client = OpenAI(api_key=api_key, base_url=base_url)

    ref_excerpt = ""
    for rel, content in loaded.resources.items():
        if rel.startswith("references/"):
            ref_excerpt += f"\n--- {rel} (excerpt) ---\n{content[:2000]}\n"

    prompt = f"""根据以下 skill 指令与用户请求，输出一个完整的暗色主题 SVG 字符串（单文件，无 markdown 包裹）。

Skill 摘要:
{loaded.body[:3000]}
{ref_excerpt}

用户请求: {query}

要求: xmlns、viewBox、暗色背景 #0f172a，至少 3 个组件方框与箭头。"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = re.sub(r"^```(?:svg|xml)?\s*|\s*```$", "", text, flags=re.I)
    if "<svg" not in text:
        return False, "LLM 未返回有效 SVG"
    out_path.write_text(text, encoding="utf-8")
    return True, str(out_path)


def execute_skill(
    meta: SkillMeta,
    loader: ProgressiveSkillLoader,
    query: str,
    output_dir: Path,
    *,
    use_llm: bool = False,
) -> ExecResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if meta.name == "flash-card":
        return execute_flash_card(meta, loader, query, output_dir, use_llm=use_llm)
    if meta.name == "baoyu-diagram":
        return execute_baoyu_diagram(meta, loader, query, output_dir, use_llm=use_llm)

    # 通用 fallback
    loader.ensure_body()
    result = ExecResult(skill_name=meta.name)
    result.add("load_body", f"已加载 SKILL.md ({len(loader.loaded.body)} chars)")
    result.message = "该 skill 暂无专用执行器；请使用 LLM 模式或扩展 skill_executor.py"
    return result
