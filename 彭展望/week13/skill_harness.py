"""
第十三周作业：可实现「渐进式加载执行 Skills」的 Harness
=========================================================

作业要求：写一套可以实现渐进式加载执行 skills 的 harness。

一句话概括：Harness 不把所有 SKILL.md 一次性塞进 context，而是按
**渐进式披露（Progressive Disclosure）** 的三层机制，用到哪一层才加载哪一层。

对应课件《skills》Part 3「渐进式披露」的三层模型：

  ┌─ L1 常驻层 (Always Loaded) ─ 每个 skill 只暴露 1 行摘要(name+description)
  │                              → 组成索引，常驻 system prompt，< 200 tokens
  ├─ L2 触发层 (On Demand)     ─ 用户消息命中某个 skill 后，才加载它完整的
  │                              SKILL.md 正文（500~2000 tokens）
  └─ L3 执行层 (In Context)    ─ 执行 skill 时，再按需 read 它的 references/、
                                 跑它的 scripts/，产物写回工作目录

与课堂 ReAct（week12）的关系：week12 让 agent 学会「循环调用工具」；本作业
在其之上加一层「**先决定加载哪个 skill、再在 skill 指令约束下调用工具**」的
调度器（元技能思想），并全程做 token 计量，量化渐进式披露到底省了多少。

本 harness 直接对准 week13 课堂素材里的两个真实 skill：
    ../week13 skills和harness/skills/{flash-card, baoyu-diagram}
它们各自是一个目录，含 SKILL.md(带 YAML frontmatter) + scripts/ + references/ / data/。

运行（默认指向课堂 skills 目录）：
  # 列出被发现的 skills 与 L1 索引（只解析 frontmatter，不读正文）
  python skill_harness.py --list

  # 给一句话，走完 L1 匹配 → L2 加载正文 → L3 执行，并打印 token 账单
  python skill_harness.py --query "给我做一张 resilient 的单词闪卡"

  # 一次跑通内置的多条示例（用于交作业的 run_output.log）
  python skill_harness.py --demo

LLM 说明：
  有 API Key（DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY，复用 week12 课堂配置）时，
  L2 的 skill 匹配和 L3 的工具循环都交给真实 LLM（ReAct 手写解析版）。
  没有 Key 时，自动降级到「确定性规划器（演示模式）」：L1 匹配用关键词打分，
  L3 按 skill 目录里真实存在的 script/reference 跑一遍——离线也能产出真实文件，
  方便复现和交作业。渐进式披露的加载/计量机制两种模式完全一致。
"""

import os
import re
import sys
import json
import time
import shlex
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ══════════════════════════════════════════════════════════════════════════════
# 0. token 估算  ——  渐进式披露的收益全靠它量化，所以放最前面
# ══════════════════════════════════════════════════════════════════════════════
def estimate_tokens(text: str) -> int:
    """
    粗略 token 估算（不引入 tiktoken 依赖，够用来看数量级/对比即可）：
      - CJK 字符按 ~1 token/字
      - 其余按 ~1 token / 4 字符
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Skill 数据结构  ——  关键：frontmatter 与 body 分离加载
# ══════════════════════════════════════════════════════════════════════════════
_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    极简 YAML frontmatter 解析（只需要 name/description/version 等标量与折叠块）。
    返回 (元数据 dict, 正文 str)。不引第三方 yaml 依赖。
    """
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    key, buf, folding = None, [], False
    for line in raw.splitlines():
        if folding and (line.startswith("  ") or line.strip() == ""):
            buf.append(line.strip())
            continue
        if folding:
            meta[key] = " ".join(x for x in buf if x).strip()
            folding, buf = False, []
        m2 = re.match(r"^(\w[\w\-]*):\s*(.*)$", line)
        if not m2:
            continue
        k, v = m2.group(1), m2.group(2).strip()
        if v in (">-", ">", "|", "|-"):      # 折叠/保留块，后续缩进行是值
            key, folding = k, True
            continue
        meta[k] = v.strip("'\"")
    if folding and key:
        meta[key] = " ".join(x for x in buf if x).strip()
    return meta, body


@dataclass
class Skill:
    """一个 skill = 一个目录 + 目录内的 SKILL.md。frontmatter 常驻，body 懒加载。"""
    name: str
    description: str
    version: str
    skill_md: Path                       # SKILL.md 路径
    base_dir: Path                       # skill 根目录（{baseDir}）
    _body: Optional[str] = field(default=None, repr=False)   # L2 才填充

    # ---- L1：只暴露一行摘要（常驻索引用） ----
    def index_line(self) -> str:
        return f"- {self.name}: {self.description}"

    def index_tokens(self) -> int:
        return estimate_tokens(self.index_line())

    # ---- L2：按需加载完整正文 ----
    def load_body(self) -> str:
        if self._body is None:
            _, body = _parse_frontmatter(self.skill_md.read_text(encoding="utf-8"))
            self._body = body.strip()
        return self._body

    def is_body_loaded(self) -> bool:
        return self._body is not None

    def body_tokens(self) -> int:
        return estimate_tokens(self.load_body())

    # ---- L3：可被按需读取的 references、可被执行的 scripts ----
    def list_references(self) -> list[Path]:
        d = self.base_dir / "references"
        return sorted(d.glob("*.md")) if d.is_dir() else []

    def list_scripts(self) -> list[Path]:
        d = self.base_dir / "scripts"
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if p.suffix in (".py", ".ts", ".sh", ".js"))

    def full_footprint_tokens(self) -> int:
        """全量加载基线：正文 + 所有 references 全部塞进 context 会占多少 token。"""
        t = self.body_tokens()
        for ref in self.list_references():
            t += estimate_tokens(ref.read_text(encoding="utf-8", errors="ignore"))
        return t


# ══════════════════════════════════════════════════════════════════════════════
# 2. Registry  ——  L1 发现 & 索引 & 触发匹配
# ══════════════════════════════════════════════════════════════════════════════
class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: list[Skill] = []

    def discover(self) -> "SkillRegistry":
        """
        扫描 skills 目录，每个含 SKILL.md 的子目录注册为一个 skill。
        【渐进式披露的第一刀】只读 frontmatter 拿 name/description，绝不读正文。
        """
        if not self.skills_dir.is_dir():
            raise FileNotFoundError(f"skills 目录不存在：{self.skills_dir}")
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            text = md.read_text(encoding="utf-8")
            meta, _body = _parse_frontmatter(text)      # _body 丢弃，不进内存
            name = meta.get("name") or md.parent.name
            self.skills.append(Skill(
                name=name,
                description=meta.get("description", "").strip(),
                version=meta.get("version", ""),
                skill_md=md,
                base_dir=md.parent,
            ))
        return self

    # ---- L1 常驻索引：所有 skill 的一行摘要拼成的字符串 ----
    def index_text(self) -> str:
        header = "# 可用 Skills（渐进式加载索引 · 仅摘要）\n"
        return header + "\n".join(s.index_line() for s in self.skills)

    def index_tokens(self) -> int:
        return estimate_tokens(self.index_text())

    # ---- 全量加载基线：把所有 skill 正文+references 都塞进去要多少 token ----
    def full_load_tokens(self) -> int:
        return sum(s.full_footprint_tokens() for s in self.skills)

    def get(self, name: str) -> Optional[Skill]:
        return next((s for s in self.skills if s.name == name), None)

    # ---- L2 触发：给一句话，从索引里选出最匹配的一个 skill ----
    def match(self, query: str, client=None, model: str = "") -> tuple[Optional[Skill], str]:
        """
        返回 (选中的 skill, 决策说明)。
        关键：匹配阶段只能看到 L1 索引（name+description），看不到任何 skill 正文——
        这正是渐进式披露省 token 的地方。
        """
        if client is not None:
            return self._match_llm(query, client, model)
        return self._match_keyword(query)

    def _match_keyword(self, query: str) -> tuple[Optional[Skill], str]:
        """离线/降级：用 query 与 (name+description) 的词面重叠打分。"""
        q = query.lower()
        q_tokens = set(re.findall(r"[a-z]+|[一-鿿]", q))
        best, best_score = None, 0.0
        for s in self.skills:
            hay = (s.name + " " + s.description).lower()
            hay_tokens = set(re.findall(r"[a-z]+|[一-鿿]", hay))
            overlap = len(q_tokens & hay_tokens)
            # skill 名直接出现在 query 里，强加权
            score = overlap + (5 if s.name.lower() in q else 0)
            if score > best_score:
                best, best_score = s, score
        if best is None or best_score == 0:
            return None, "无 skill 命中（关键词打分为 0）"
        return best, f"关键词打分选中 [{best.name}]（score={best_score:.0f}，仅依据 L1 索引）"

    def _match_llm(self, query: str, client, model: str) -> tuple[Optional[Skill], str]:
        """在线：把 L1 索引给 LLM，让它只返回一个 skill 名或 NONE。"""
        prompt = (
            "你是 skill 调度器。下面是所有可用 skill 的摘要索引（只有名字和描述），"
            "根据用户输入选出**唯一**最合适的一个来处理；若都不合适输出 NONE。\n"
            "只输出 skill 名字或 NONE，不要多余内容。\n\n"
            f"{self.index_text()}\n\n用户输入：{query}\n\n选择："
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        choice = resp.choices[0].message.content.strip().splitlines()[0].strip()
        if choice.upper() == "NONE":
            return None, "LLM 判定无合适 skill"
        skill = self.get(choice) or next((s for s in self.skills if s.name in choice), None)
        if skill is None:
            return None, f"LLM 返回未知 skill 名：{choice!r}"
        return skill, f"LLM 依据 L1 索引选中 [{skill.name}]"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 执行层工具箱  ——  L3：read reference / run script / write file
# ══════════════════════════════════════════════════════════════════════════════
class Toolbox:
    """
    skill 执行期间可调用的工具。每次 read_reference 都会记账，用来证明
    references 是「用到才加载」的，而不是一开始就全进 context。
    """
    def __init__(self, skill: Skill, workdir: Path, logger):
        self.skill = skill
        self.workdir = workdir
        self.log = logger
        self.loaded_ref_tokens = 0          # L3 累计真正读进 context 的 reference token
        self.produced_files: list[Path] = []

    def read_reference(self, rel_path: str) -> str:
        """按需读取 skill 内的一个文件（通常是 references/*.md）——渐进式披露 L3。"""
        p = (self.skill.base_dir / rel_path).resolve()
        if not str(p).startswith(str(self.skill.base_dir.resolve())):
            return f"[拒绝] 越权路径：{rel_path}"     # 简单的路径围栏
        if not p.is_file():
            return f"[未找到] {rel_path}"
        text = p.read_text(encoding="utf-8", errors="ignore")
        tk = estimate_tokens(text)
        self.loaded_ref_tokens += tk
        self.log(f"   📎 read_reference({rel_path}) → 加载 {tk} tokens 进 context")
        return text

    def run_script(self, command: str) -> str:
        """执行 skill 的脚本（shell_exec）。工作目录 = workdir。"""
        self.log(f"   ⚙️  run_script: {command}")
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(self.workdir),
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return "[超时] 脚本执行超过 120s"
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"[exit={proc.returncode}]\n{out.strip()}"

    def write_file(self, rel_path: str, content: str) -> str:
        """把产物写到工作目录。"""
        p = (self.workdir / rel_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.produced_files.append(p)
        self.log(f"   💾 write_file({rel_path}) → {len(content)} 字节")
        return f"已写入 {p}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM 客户端（复用 week12 课堂配置，可选）
# ══════════════════════════════════════════════════════════════════════════════
def make_client() -> tuple[Optional[object], str]:
    try:
        from openai import OpenAI
    except ImportError:
        return None, ""
    if os.getenv("DASHSCOPE_API_KEY"):
        return OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"), \
               os.getenv("AGENT_MODEL", "qwen-max")
    if os.getenv("DEEPSEEK_API_KEY"):
        return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                      base_url="https://api.deepseek.com"), \
               os.getenv("AGENT_MODEL", "deepseek-chat")
    return None, ""


# ══════════════════════════════════════════════════════════════════════════════
# 5. Harness  ——  串起 L1 → L2 → L3，并出 token 账单
# ══════════════════════════════════════════════════════════════════════════════
# 执行层 system prompt：告诉 LLM 它现在被约束在某个 skill 的指令里，可用哪些工具
_EXEC_SYSTEM = """你是一个在 harness 中运行的 skill 执行器。下面 <SKILL> 里是当前
被触发的 skill 的完整指令，请**严格按它的执行流程**完成用户任务。

你可以调用以下工具，每次只输出一个动作，格式固定：

Thought: 你的思考
Action: read_reference | run_script | write_file
Action Input: {...}   # JSON

工具说明：
- read_reference  {"path": "references/xxx.md"}      按需读取 skill 内的参考文件（相对 baseDir）
- run_script      {"command": "python <脚本绝对路径> ..."}  在工作目录执行脚本
- write_file      {"path": "a.json", "content": "..."} 把产物写到工作目录

<运行环境>（由 harness 提供，SKILL.md 里的路径以此为准）
- 本 skill 根目录 baseDir（绝对路径）：{base_dir}
- 可用脚本（绝对路径）：{scripts}
- 可用参考文件（相对 baseDir 的 references/*.md）：{refs}
- 工作目录 workdir（产物输出到这里，run_script 的 cwd）：{workdir}
重要：SKILL.md 里若出现 `{baseDir}`、`.cursor/skills/<name>/` 之类的占位/示例路径，
一律替换为上面的真实 baseDir 绝对路径；脚本一律用上面给的绝对路径调用。
</运行环境>

完成任务后输出：
Thought: 已完成
Final Answer: 给用户的简短总结

<SKILL>
{skill_body}
</SKILL>
"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)")
_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)


class SkillHarness:
    def __init__(self, registry: SkillRegistry, workdir: Path, verbose: bool = True):
        self.reg = registry
        self.workdir = workdir
        self.verbose = verbose
        self.client, self.model = make_client()

    def log(self, msg: str = ""):
        if self.verbose:
            print(msg)

    # ---------------------------------------------------------------- 主流程
    def run(self, query: str) -> dict:
        self.log("\n" + "=" * 70)
        self.log(f"用户输入: {query}")
        self.log("=" * 70)

        mode = "LLM" if self.client else "确定性规划器(离线演示)"
        idx_tokens = self.reg.index_tokens()
        self.log(f"运行模式: {mode} | 已发现 {len(self.reg.skills)} 个 skill")
        self.log(f"[L1 常驻层] 索引常驻 context: {idx_tokens} tokens（只含各 skill 一行摘要）")

        # —— L2 触发：只看索引，选一个 skill ——
        skill, why = self.reg.match(query, self.client, self.model)
        self.log(f"[L2 触发层] {why}")
        if skill is None:
            self.log("→ 无 skill 处理，结束。")
            return {"skill": None, "index_tokens": idx_tokens}

        body_tokens = skill.body_tokens()     # 触发 load_body()
        self.log(f"[L2 触发层] 加载 [{skill.name}] 完整 SKILL.md → +{body_tokens} tokens 进 context")

        # —— L3 执行 ——
        self.log(f"[L3 执行层] 在 [{skill.name}] 指令约束下执行任务：")
        tools = Toolbox(skill, self.workdir, self.log)
        if self.client:
            answer = self._exec_llm(skill, query, tools)
        else:
            answer = self._exec_offline(skill, query, tools)

        # —— token 账单：渐进式 vs 全量加载 ——
        used = idx_tokens + body_tokens + tools.loaded_ref_tokens
        full = self.reg.full_load_tokens()
        saved = full - used
        pct = (saved / full * 100) if full else 0
        self.log("\n[Token 账单] —— 渐进式披露 vs 全量加载")
        self.log(f"   全量加载(所有 skill 正文+references 全塞进去): {full} tokens")
        self.log(f"   渐进式实际占用: {used} tokens"
                 f"  = 索引 {idx_tokens} + 正文 {body_tokens} + 按需reference {tools.loaded_ref_tokens}")
        self.log(f"   节省: {saved} tokens ({pct:.0f}%)")
        if tools.produced_files:
            self.log("   产物: " + ", ".join(str(p) for p in tools.produced_files))
        self.log(f"✅ {answer}")

        return {
            "skill": skill.name, "answer": answer,
            "index_tokens": idx_tokens, "body_tokens": body_tokens,
            "ref_tokens": tools.loaded_ref_tokens,
            "used_tokens": used, "full_tokens": full, "saved_pct": pct,
            "produced": [str(p) for p in tools.produced_files],
        }

    # ---------------------------------------------------------------- L3：LLM 版
    def _exec_llm(self, skill: Skill, query: str, tools: Toolbox, max_steps: int = 8) -> str:
        system = (_EXEC_SYSTEM
                  .replace("{base_dir}", str(skill.base_dir.resolve()))
                  .replace("{scripts}", str([str(p.resolve()) for p in skill.list_scripts()]))
                  .replace("{refs}", str([p.name for p in skill.list_references()]))
                  .replace("{workdir}", str(tools.workdir.resolve()))
                  .replace("{skill_body}", skill.load_body()))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": query}]
        for step in range(1, max_steps + 1):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0)
            text = resp.choices[0].message.content
            messages.append({"role": "assistant", "content": text})

            final = _FINAL_RE.search(text)
            if final:
                return final.group(1).strip()

            action_m, input_m = _ACTION_RE.search(text), _INPUT_RE.search(text)
            thought = _THOUGHT_RE.search(text)
            if thought:
                self.log(f"   🧠 {thought.group(1).strip()[:80]}")
            if not action_m:
                return text.strip()
            action = action_m.group(1)
            try:
                args = json.loads(input_m.group(1)) if input_m else {}
            except json.JSONDecodeError:
                args = {}
            obs = self._dispatch(action, args, tools)
            messages.append({"role": "user", "content": f"Observation: {obs[:1500]}"})
        return "（达到最大步数）"

    def _dispatch(self, action: str, args: dict, tools: Toolbox) -> str:
        if action == "read_reference":
            return tools.read_reference(args.get("path", ""))
        if action == "run_script":
            return tools.run_script(args.get("command", ""))
        if action == "write_file":
            return tools.write_file(args.get("path", ""), args.get("content", ""))
        return f"[未知工具] {action}"

    # ---------------------------------------------------------------- L3：离线确定性版
    def _exec_offline(self, skill: Skill, query: str, tools: Toolbox) -> str:
        """
        无 API Key 时的演示规划器。它不臆造 LLM 智能，只做两件确定性的事，
        用来跑通并展示 L3 的「按需加载 reference + 执行 script」机制：
          1. 若 skill 有 references/，读取与任务最相关的一个（证明按需加载）；
          2. 若 harness 内置了该 skill 的确定性执行手，跑它的真实 script 产出文件。
        """
        handler = _OFFLINE_HANDLERS.get(skill.name)
        if handler:
            return handler(self, skill, query, tools)

        # 通用兜底：演示「按需读取一个 reference」
        refs = skill.list_references()
        if refs:
            tools.read_reference(f"references/{refs[0].name}")
            return (f"[离线演示] 已按需加载 {refs[0].name}；该 skill 的完整执行需要 LLM "
                    f"生成内容，请设置 DASHSCOPE_API_KEY 后重跑。")
        return "[离线演示] 该 skill 无内置离线执行手，请设置 API Key 后由 LLM 执行。"


# —— 内置离线执行手（仅为无 Key 时能跑通真实产物；有 Key 时走上面的 LLM 通用循环）——
def _offline_flashcard(harness: "SkillHarness", skill: Skill, query: str, tools: Toolbox) -> str:
    """
    flash-card 的离线执行手：从 query 里抽单词，用 SKILL.md 示例的数据 schema
    造一份 JSON（离线无法真造词典数据，退回内置词库），写入 skill 的 data/ 并
    运行真实脚本 make_flashcard.py 产出 HTML。展示 L3「写文件 + 跑脚本」。
    """
    m = re.search(r"[a-zA-Z]{3,}", query)
    word = (m.group(0) if m else "resilient").lower()
    data = _MINI_LEXICON.get(word, {
        "word": word, "phonetic": "", "pos": "",
        "definition": "（离线内置词库无该词，仅演示流程）",
        "examples": [{"en": f"This is an example with {word}.", "zh": f"这是一个含 {word} 的例句。"}] * 3,
        "synonyms": [],
    })
    data["word"] = word
    data_path = skill.base_dir / "data" / f"{word}.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    harness.log(f"   💾 生成数据 data/{word}.json（依 SKILL.md 的字段 schema）")

    script = skill.base_dir / "scripts" / "make_flashcard.py"
    out_html = harness.workdir / f"{word}.html"
    cmd = f'{shlex.quote(sys.executable)} {shlex.quote(str(script))} ' \
          f'{shlex.quote(str(data_path))} -o {shlex.quote(str(out_html))}'
    obs = tools.run_script(cmd)
    if out_html.is_file():
        tools.produced_files.append(out_html)
    harness.log(f"   ↳ {obs.splitlines()[0] if obs else ''}")
    return f"已为单词 [{word}] 生成闪卡：{out_html}"


_OFFLINE_HANDLERS = {"flash-card": _offline_flashcard}

# 离线内置迷你词库（无网时也能产出有意义的闪卡；有 API Key 时由 LLM 生成，不用它）
_MINI_LEXICON = {
    "resilient": {
        "word": "resilient", "phonetic": "/rɪˈzɪliənt/", "pos": "adj.",
        "definition": "能迅速从困难、挫折中恢复过来的；有韧性的，适应力强的",
        "examples": [
            {"en": "She is a resilient child who bounces back quickly from setbacks.",
             "zh": "她是个有韧性的孩子，遇到挫折能很快恢复过来。"},
            {"en": "The economy proved remarkably resilient during the crisis.",
             "zh": "在危机期间，经济表现出了惊人的韧性。"},
            {"en": "A resilient mindset helps you cope with life's challenges.",
             "zh": "一种有韧性的心态能帮你应对生活中的挑战。"},
        ],
        "synonyms": ["tough", "flexible", "strong", "hardy", "buoyant", "springy"],
    },
    "meticulous": {
        "word": "meticulous", "phonetic": "/məˈtɪkjələs/", "pos": "adj.",
        "definition": "一丝不苟的；小心翼翼的；注重细节的",
        "examples": [
            {"en": "She kept meticulous records of every transaction.",
             "zh": "她对每一笔交易都做了一丝不苟的记录。"},
            {"en": "The restoration required meticulous attention to detail.",
             "zh": "这次修复需要对细节一丝不苟的关注。"},
            {"en": "He is meticulous about keeping his workspace clean.",
             "zh": "他对保持工作区整洁非常讲究。"},
        ],
        "synonyms": ["careful", "thorough", "precise", "scrupulous", "diligent"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_SKILLS = (Path(__file__).resolve().parent.parent
                   / "week13 skills和harness" / "skills")


def cmd_list(reg: SkillRegistry):
    print("=" * 70)
    print("已发现的 Skills（L1：仅解析 frontmatter，未读正文）")
    print("=" * 70)
    for s in reg.skills:
        body_state = "已加载" if s.is_body_loaded() else "未加载(懒加载)"
        print(f"\n● {s.name}  {('v'+s.version) if s.version else ''}")
        print(f"  描述: {s.description}")
        print(f"  L1 摘要占用: {s.index_tokens()} tokens | 正文: {body_state}")
        print(f"  references: {[p.name for p in s.list_references()]}")
        print(f"  scripts:    {[p.name for p in s.list_scripts()]}")
    print("\n" + "-" * 70)
    print(f"L1 常驻索引总占用: {reg.index_tokens()} tokens")
    print(f"若全量加载(所有正文+references): {reg.full_load_tokens()} tokens")
    print(f"→ 渐进式披露起步就省下 {reg.full_load_tokens() - reg.index_tokens()} tokens 的常驻开销")


def main():
    ap = argparse.ArgumentParser(description="渐进式加载执行 skills 的 harness（week13 作业）")
    ap.add_argument("--skills-dir", default=str(_DEFAULT_SKILLS), help="skills 根目录")
    ap.add_argument("--workdir", default=".", help="执行产物输出目录")
    ap.add_argument("--query", help="用户输入的一句话")
    ap.add_argument("--list", action="store_true", help="只列出被发现的 skills 与索引")
    ap.add_argument("--demo", action="store_true", help="跑内置多条示例")
    args = ap.parse_args()

    reg = SkillRegistry(Path(args.skills_dir)).discover()

    if args.list:
        cmd_list(reg)
        return

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    harness = SkillHarness(reg, workdir)

    if args.demo:
        cmd_list(reg)
        for q in [
            "给我做一张 resilient 的单词闪卡",
            "帮我画一张微服务系统的架构图",
            "帮我算一下今天天气怎么样",       # 故意没有匹配 skill，演示 NONE
        ]:
            harness.run(q)
        return

    if args.query:
        harness.run(args.query)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
