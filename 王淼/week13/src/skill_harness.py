"""
skill_harness.py — 渐进式 Skill 加载与执行框架

教学重点：
  1. Skill 接口设计：should_activate() + execute() 两阶段分离
     - should_activate：自判断"当前上下文是否需要我"（渐进式加载的核心）
     - execute：执行 Skill 逻辑，结果写入 SkillContext
  2. Harness 调度：按依赖顺序渐进式加载，避免一次性全部执行
     - 每个 Skill 独立判断是否激活，按需执行
     - Skill 间通过 SkillContext 传递状态，解耦
  3. 将现有四层记忆 + LLM 调用 + Flush 全部封装为可插拔 Skill
     - 新增 Skill 只需继承 Skill 基类，注册到 Harness 即可

使用方式：
  from src.skill_harness import Harness

  harness = Harness(db, loader, retriever, flusher)
  harness.register_all_skills()

  # 同步执行（CLI 用）
  result_ctx = harness.run(user_input="你好", session_id=1)
  print(result_ctx.response)

  # 流式执行（SSE 用）
  for event in harness.run_stream(user_input="你好", session_id=1):
      yield event
"""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.session_db import SessionDB
from src.memory_loader import MemoryLoader
from src.retrieval import HybridRetriever
from src.memory_flush import MemoryFlusher
from src.llm_config import get_chat_client

WORD_CARDS_PATH = Path(__file__).parent.parent / "memory" / "WORD_CARDS.md"


# ═══════════════════════════════════════════════════════════════════════════════
#  SkillContext — Skill 间共享的执行上下文
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SkillContext:
    """Skill 间传递的共享状态。每个 Skill 读自己需要的字段，写自己的产物。"""

    # ── 输入 ──────────────────────────────────────────────────────────
    user_input: str = ""
    session_id: int = 0

    # ── 中间产物（由各 Skill 填入，后续 Skill 读取）──────────────────
    system_prompt: str = ""
    history: list[dict] = field(default_factory=list)
    semantic_results: list[dict] = field(default_factory=list)
    layers_info: list[dict] = field(default_factory=list)

    # ── 最终产物 ──────────────────────────────────────────────────────
    response: str = ""

    # ── 扩展点：任意 Skill 可写入自定义数据 ──────────────────────────
    extras: dict = field(default_factory=dict)

    # ── 运行时标志 ──────────────────────────────────────────────────
    aborted: bool = False          # 某个 Skill 可设为 True 中止后续流程
    active_skills: list[str] = field(default_factory=list)  # 本次实际激活的 Skill 名


# ═══════════════════════════════════════════════════════════════════════════════
#  Skill 基类
# ═══════════════════════════════════════════════════════════════════════════════

class Skill(ABC):
    """
    Skill 基类。每个 Skill 封装一个独立的能力单元。

    渐进式加载的关键：
      should_activate() 决定是否需要执行 —— 不是所有 Skill 每次都跑。
      例如：SemanticSearchSkill 只在有 user_input 时才激活；
            MemoryFlushSkill 只在显式触发 flush 时才激活。
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def should_activate(self, ctx: SkillContext) -> bool:
        """判断当前上下文是否应激活此 Skill。"""
        ...

    @abstractmethod
    def execute(self, ctx: SkillContext) -> None:
        """执行 Skill 逻辑，结果写入 ctx。"""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  具体 Skill 实现
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryLoadSkill(Skill):
    """
    Layer 3 长期记忆：加载 Markdown 文件组装 System Prompt。

    始终激活（System Prompt 是所有对话的基础）。
    """

    name = "memory_load"
    description = "加载 SOUL / USER / AGENTS / MEMORY.md 组装系统提示"

    def __init__(self, loader: MemoryLoader):
        self.loader = loader

    def should_activate(self, ctx: SkillContext) -> bool:
        return True  # 每次对话都需要 System Prompt

    def execute(self, ctx: SkillContext) -> None:
        prompt_result = self.loader.build_system_prompt(recent_memory_limit=10)
        ctx.system_prompt = prompt_result.system_prompt
        ctx.layers_info = [
            {"name": l.name, "source": l.source_file, "chars": l.char_count}
            for l in prompt_result.layers
        ]


class SemanticSearchSkill(Skill):
    """
    Layer 4 语义检索：FAISS + FTS5 混合检索。

    只在有 user_input 时激活（首轮对话也可触发）。
    """

    name = "semantic_search"
    description = "对用户输入做混合检索（向量 0.7 + BM25 0.3）"

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def should_activate(self, ctx: SkillContext) -> bool:
        return bool(ctx.user_input.strip())

    def execute(self, ctx: SkillContext) -> None:
        ctx.semantic_results = self.retriever.search(ctx.user_input, top_k=3)


class SessionHistorySkill(Skill):
    """
    Layer 2 短期记忆：加载当前会话的历史消息。

    始终激活（需要历史来维持多轮上下文）。
    """

    name = "session_history"
    description = "加载 SQLite 中的会话历史消息"

    def __init__(self, db: SessionDB):
        self.db = db

    def should_activate(self, ctx: SkillContext) -> bool:
        return True

    def execute(self, ctx: SkillContext) -> None:
        ctx.history = self.db.get_session_messages(ctx.session_id)


class ContextAssemblySkill(Skill):
    """
    上下文组装：将 Layer 2/3/4 的产物拼接成最终的 API messages。

    依赖 MemoryLoadSkill、SemanticSearchSkill、SessionHistorySkill 的产物。
    """

    name = "context_assembly"
    description = "拼装 System Prompt + 历史 + 用户输入为 LLM 消息列表"

    def should_activate(self, ctx: SkillContext) -> bool:
        return True

    def execute(self, ctx: SkillContext) -> None:
        # 拼接语义检索结果到 system prompt
        system_prompt = ctx.system_prompt
        if ctx.semantic_results:
            snippets = [
                f"- [{r['category']}] {r.get('title', '')}: {r['content'][:100]}"
                for r in ctx.semantic_results
            ]
            system_prompt += "\n\n## 语义检索到的相关记忆\n" + "\n".join(snippets)

        # 组装成 API messages
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in ctx.history:
            api_messages.append({"role": m["role"], "content": m["content"]})
        api_messages.append({"role": "user", "content": ctx.user_input})

        ctx.extras["api_messages"] = api_messages
        ctx.extras["system_chars"] = len(system_prompt)
        ctx.extras["history_turns"] = len(ctx.history)


class LLMResponseSkill(Skill):
    """
    LLM 调用：发送 messages 给 LLM，获取回复。

    依赖 ContextAssemblySkill 的产物（api_messages）。
    """

    name = "llm_response"
    description = "调用 LLM 生成回复（支持流式和非流式）"

    def __init__(self):
        pass

    def should_activate(self, ctx: SkillContext) -> bool:
        return bool(ctx.user_input.strip())

    def execute(self, ctx: SkillContext) -> None:
        """非流式调用（供 CLI / 一次性获取完整回复使用）"""
        api_messages = ctx.extras.get("api_messages", [])
        if not api_messages:
            return

        client, model = get_chat_client()
        stream = client.chat.completions.create(
            model=model, messages=api_messages, temperature=0.7, stream=True
        )
        response_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response_text += delta
        ctx.response = response_text

    def execute_stream(self, ctx: SkillContext) -> str:
        """流式调用（供 SSE 使用，逐 token yield）"""
        api_messages = ctx.extras.get("api_messages", [])
        if not api_messages:
            return

        client, model = get_chat_client()
        stream = client.chat.completions.create(
            model=model, messages=api_messages, temperature=0.7, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                ctx.response += delta
                yield delta


class SaveMessageSkill(Skill):
    """
    消息持久化：将 user + assistant 消息写入 SQLite。

    依赖 LLMResponseSkill 的产物（ctx.response）。
    """

    name = "save_message"
    description = "将对话消息写入 SQLite 持久化"

    def __init__(self, db: SessionDB):
        self.db = db

    def should_activate(self, ctx: SkillContext) -> bool:
        return bool(ctx.response)

    def execute(self, ctx: SkillContext) -> None:
        self.db.add_message(ctx.session_id, "user", ctx.user_input)
        self.db.add_message(ctx.session_id, "assistant", ctx.response)
        ctx.extras["message_count"] = self.db.get_message_count(ctx.session_id)


class MemoryFlushSkill(Skill):
    """
    Memory Flush：三阶段记忆归档。

    只在显式触发 flush 时激活（通过 ctx.extras["trigger_flush"] 标记）。
    """

    name = "memory_flush"
    description = "触发 Memory Flush：提取用户信息 + 记忆条目 + 向量化"

    def __init__(self, db: SessionDB, flusher: MemoryFlusher):
        self.db = db
        self.flusher = flusher

    def should_activate(self, ctx: SkillContext) -> bool:
        return ctx.extras.get("trigger_flush", False)

    def execute(self, ctx: SkillContext) -> None:
        messages = self.db.get_session_messages(ctx.session_id)
        user_messages = [m for m in messages if m["role"] in ("user", "assistant")]
        if not user_messages:
            ctx.extras["flush_result"] = None
            return

        result = self.flusher.flush(user_messages, ctx.session_id)
        self.db.mark_flushed(ctx.session_id)
        ctx.extras["flush_result"] = result


class WordCardSkill(Skill):
    """
    英语单词卡 Skill：帮助用户学习英语单词。

    功能：
      1. 用户请求单词学习时（如"学单词"、"单词卡"、"考我单词"），生成单词卡片
      2. 用户输入具体英语单词时，生成该词的详细卡片（音标、释义、例句、记忆技巧）
      3. 自动保存学过的单词到 WORD_CARDS.md，支持复习模式

    激活条件：
      - 包含单词学习关键词：学单词 / 单词卡 / 单词 / vocabulary / word / 复习
      - 或用户输入以英语单词开头（检测到英文字母）
    """

    name = "word_card"
    description = "生成英语单词卡片（音标、释义、例句、记忆技巧），持久化学习记录"

    # 触发关键词（用户说这些时激活）
    TRIGGER_KEYWORDS = [
        "学单词", "单词卡", "单词", "单词本", "背单词", "考我",
        "单词复习", "复习单词", "英语单词", "词汇",
        "vocabulary", "word card", "word", "learn english",
    ]

    def __init__(self):
        self._client = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            self._client, self._model = get_chat_client()
        return self._client

    @property
    def model(self):
        if self._model is None:
            self._client, self._model = get_chat_client()
        return self._model

    def should_activate(self, ctx: SkillContext) -> bool:
        text = ctx.user_input.lower().strip()
        if not text:
            return False
        # 关键词匹配
        for kw in self.TRIGGER_KEYWORDS:
            if kw in text:
                return True
        # 检测是否为纯英文单词查询（如 "apple", "serendipity"）
        # 如果用户输入主要是英文字母，判断为查词
        english_chars = re.findall(r"[a-zA-Z]", text)
        if len(english_chars) >= len(text) * 0.6 and len(text.split()) <= 3:
            return True
        return False

    def execute(self, ctx: SkillContext) -> None:
        user_input = ctx.user_input.strip()
        mode, params = self._detect_mode(user_input)

        if mode == "review":
            word_cards = self._load_word_cards()
            if not word_cards:
                ctx.extras["word_card"] = {
                    "mode": "info",
                    "message": "你的单词本还是空的，先学几个单词吧！试试输入「学单词」或直接输入一个英文单词。",
                }
                return
            # 从已学单词中随机选 3 个复习
            cards = random.sample(word_cards, min(3, len(word_cards)))
            ctx.extras["word_card"] = {
                "mode": "review",
                "cards": cards,
                "count": len(cards),
            }
            # 把复习内容注入 system prompt，让 LLM 引导用户复习
            review_text = self._format_review_prompt(cards)
            ctx.system_prompt += "\n\n## 单词复习任务\n\n" + review_text
            return

        if mode == "show_word":
            # 用户输入了具体的英文单词，生成卡片
            word = params
            card = self._generate_word_card(word)
            if card:
                self._save_word_card(card)
                ctx.extras["word_card"] = {
                    "mode": "card",
                    "card": card,
                }
                card_prompt = self._format_card_prompt(card)
                ctx.system_prompt += "\n\n## 单词卡学习\n\n" + card_prompt
                return

        if mode == "learn_new":
            # 用户想学新单词（没指定具体词），自动生成一批常用词
            cards = self._generate_batch_cards(count=5)
            for card in cards:
                self._save_word_card(card)
            ctx.extras["word_card"] = {
                "mode": "batch",
                "cards": cards,
                "count": len(cards),
            }
            batch_text = self._format_batch_prompt(cards)
            ctx.system_prompt += "\n\n## 今日单词学习\n\n" + batch_text
            return

        # 默认：把用户输入中的单词作为卡片展示
        word = user_input.split()[0] if user_input.split() else user_input
        if re.match(r"^[a-zA-Z'-]+$", word):
            card = self._generate_word_card(word)
            if card:
                self._save_word_card(card)
                ctx.extras["word_card"] = {"mode": "card", "card": card}
                ctx.system_prompt += "\n\n## 单词卡\n\n" + self._format_card_prompt(card)

    def _detect_mode(self, text: str) -> tuple:
        """检测用户意图模式"""
        lower = text.lower()

        # 复习模式
        if any(kw in lower for kw in ["复习", "review", "考我", "测试"]):
            return "review", None

        # 学习新单词（没指定具体词）
        if any(kw in lower for kw in ["学单词", "单词卡", "单词本", "背单词"]) or lower in ["vocabulary", "word"]:
            return "learn_new", None

        # 查具体单词（英文开头或包含英文单词）
        words = text.split()
        for w in words:
            if re.match(r"^[a-zA-Z][a-zA-Z'-]+$", w) and len(w) > 1:
                return "show_word", w.lower()

        # 默认：尝试解析第一个词
        first_word = words[0].lower() if words else ""
        if re.match(r"^[a-zA-Z'-]+$", first_word) and len(first_word) > 1:
            return "show_word", first_word

        return "learn_new", None

    def _generate_word_card(self, word: str) -> Optional[dict]:
        """用 LLM 生成单个单词的卡片"""
        prompt = f"""请为单词 "{word}" 生成一张学习卡片，严格按以下 JSON 格式输出（只输出 JSON，不要其他文字）：
{{
  "word": "{word}",
  "phonetic": "音标（IPA 格式）",
  "pos": "词性（如 n. / v. / adj.）",
  "meaning": "中文释义",
  "example": "英文例句",
  "example_trans": "例句中文翻译",
  "memory_tip": "记忆技巧（词根词缀/联想记忆）",
  "synonyms": ["同义词1", "同义词2"],
  "level": "难度（CET-4 / CET-6 / 雅思 / GRE / 专八）"
}}"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            # 提取 JSON
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                card = json.loads(match.group())
                card["learned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                return card
        except Exception:
            pass
        return None

    def _generate_batch_cards(self, count: int = 5) -> list[dict]:
        """生成一批常用高频单词卡片"""
        prompt = f"""请生成 {count} 个适合英语学习者的常用高频单词卡片。
每个单词严格按以下 JSON 数组格式输出（只输出 JSON，不要其他文字）：
[
  {{
    "word": "单词",
    "phonetic": "音标",
    "pos": "词性",
    "meaning": "中文释义",
    "example": "英文例句",
    "example_trans": "例句翻译",
    "memory_tip": "记忆技巧",
    "synonyms": ["同义词"],
    "level": "难度"
  }}
]

要求：
- 覆盖不同词性（名词、动词、形容词、副词）
- 适合日常使用
- 包含至少一个 CET-4 级词汇"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                cards = json.loads(match.group())
                for c in cards:
                    c["learned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                return cards
        except Exception:
            pass
        return []

    def _load_word_cards(self) -> list[dict]:
        """从 WORD_CARDS.md 加载已学单词"""
        if not WORD_CARDS_PATH.exists():
            return []
        text = WORD_CARDS_PATH.read_text(encoding="utf-8")
        cards = []
        # 按 ### 分割条目
        entries = re.split(r"(?=### \[)", text)
        for entry in entries:
            entry = entry.strip()
            if not entry or not entry.startswith("### ["):
                continue
            # 尝试解析 word 和 meaning
            word_match = re.search(r"### \[.*?\]\s+(\S+)", entry)
            meaning_match = re.search(r"释义[:：]\s*(.+)", entry)
            if word_match:
                cards.append({
                    "word": word_match.group(1),
                    "meaning": meaning_match.group(1).strip() if meaning_match else "",
                    "raw": entry[:200],
                })
        return cards

    def _save_word_card(self, card: dict) -> None:
        """保存单词卡片到 WORD_CARDS.md"""
        WORD_CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not WORD_CARDS_PATH.exists():
            WORD_CARDS_PATH.write_text(
                "# WORD_CARDS.md — 英语学习单词本\n\n"
                "> 由 WordCardSkill 自动维护。记录你学过的所有单词。\n\n"
                "<!-- CARDS_START -->\n<!-- CARDS_END -->\n",
                encoding="utf-8",
            )
        text = WORD_CARDS_PATH.read_text(encoding="utf-8")
        start = text.find("<!-- CARDS_START -->")
        end = text.find("<!-- CARDS_END -->")
        if start == -1 or end == -1:
            return

        body = text[start + len("<!-- CARDS_START -->"):end].strip()
        new_entry = self._card_to_md(card)
        updated_body = (body + "\n" + new_entry).strip()
        new_text = (
            text[:start + len("<!-- CARDS_START -->")]
            + "\n" + updated_body + "\n"
            + text[end:]
        )
        WORD_CARDS_PATH.write_text(new_text, encoding="utf-8")

    def _card_to_md(self, card: dict) -> str:
        """将单词卡转为 Markdown 格式"""
        syns = card.get("synonyms", [])
        syn_str = ", ".join(syns) if syns else "—"
        return f"""### [{card.get('level', '?')}] {card.get('word', '?')}
音标：{card.get('phonetic', '?')}
词性：{card.get('pos', '?')}
释义：{card.get('meaning', '?')}
例句：{card.get('example', '?')}
翻译：{card.get('example_trans', '?')}
记忆技巧：{card.get('memory_tip', '?')}
同义词：{syn_str}
学习时间：{card.get('learned_at', '?')}"""

    def _format_card_prompt(self, card: dict) -> str:
        """将单词卡格式化为注入 system prompt 的文本"""
        syns = card.get("synonyms", [])
        syn_str = ", ".join(syns) if syns else "无"
        return f"""请用以下单词卡帮助用户学习，要求生动有趣：

单词：{card.get('word')}
音标：{card.get('phonetic')}
词性：{card.get('pos')}
中文释义：{card.get('meaning')}
例句：{card.get('example')} ({card.get('example_trans')})
记忆技巧：{card.get('memory_tip')}
同义词：{syn_str}

请用自然的口语化风格介绍这个单词，引导用户造句练习。"""

    def _format_batch_prompt(self, cards: list[dict]) -> str:
        """将批量单词卡格式化为 system prompt 文本"""
        parts = [f"今日学习 {len(cards)} 个新单词，请逐个生动介绍："]
        for i, c in enumerate(cards, 1):
            syns = c.get("synonyms", [])
            syn_str = ", ".join(syns) if syns else "无"
            parts.append(
                f"\n{i}. **{c.get('word')}** [{c.get('level', '')}]\n"
                f"   音标：{c.get('phonetic')}  词性：{c.get('pos')}\n"
                f"   释义：{c.get('meaning')}\n"
                f"   例句：{c.get('example')} ({c.get('example_trans')})\n"
                f"   技巧：{c.get('memory_tip')}  同义词：{syn_str}"
            )
        return "\n".join(parts) + "\n\n请用有趣的方式逐个讲解，并在最后邀请用户造句。"

    def _format_review_prompt(self, cards: list[dict]) -> str:
        """将复习模式格式化为 system prompt 文本"""
        parts = [f"请帮用户复习以下 {len(cards)} 个学过的单词："]
        for c in cards:
            parts.append(f"- **{c.get('word')}** — {c.get('meaning', '?')}")
        parts.append("\n请先展示单词让用户回忆释义，然后公布答案并讲解用法。")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Harness — 渐进式 Skill 调度器
# ═══════════════════════════════════════════════════════════════════════════════


class Harness:
    """
    Skill 渐进式加载调度器。

    核心流程：
      1. 按注册顺序遍历所有 Skill
      2. 对每个 Skill 调用 should_activate(ctx) 判断是否需要执行
      3. 激活的 Skill 执行 execute(ctx)，产物写入 ctx 供后续 Skill 使用
      4. 任何 Skill 可设 ctx.aborted=True 中止后续流程

    这样实现了"渐进式加载"——不是所有 Skill 每次都跑，
    而是根据上下文按需激活。
    """

    def __init__(
        self,
        db: SessionDB,
        loader: MemoryLoader,
        retriever: HybridRetriever,
        flusher: MemoryFlusher,
    ):
        self.db = db
        self.loader = loader
        self.retriever = retriever
        self.flusher = flusher
        self._skills: list[Skill] = []

    def register_skill(self, skill: Skill) -> None:
        """注册一个 Skill 到 Harness。"""
        self._skills.append(skill)

    def register_all_skills(self) -> None:
        """按依赖顺序注册所有内置 Skill。"""
        self.register_skill(MemoryLoadSkill(self.loader))
        self.register_skill(WordCardSkill())                    # 单词卡（在 MemoryLoad 之后，修改 system_prompt）
        self.register_skill(SemanticSearchSkill(self.retriever))
        self.register_skill(SessionHistorySkill(self.db))
        self.register_skill(ContextAssemblySkill())
        self.register_skill(LLMResponseSkill())
        self.register_skill(SaveMessageSkill(self.db))
        self.register_skill(MemoryFlushSkill(self.db, self.flusher))

    def run(self, user_input: str, session_id: int, trigger_flush: bool = False) -> SkillContext:
        """
        同步执行所有激活的 Skill（非流式，适合 CLI）。

        Args:
            user_input: 用户输入文本
            session_id: 当前会话 ID
            trigger_flush: 是否触发 Memory Flush

        Returns:
            执行完成后的 SkillContext（包含所有产物）
        """
        ctx = SkillContext(
            user_input=user_input,
            session_id=session_id,
        )
        if trigger_flush:
            ctx.extras["trigger_flush"] = True

        for skill in self._skills:
            if ctx.aborted:
                break
            if skill.should_activate(ctx):
                skill.execute(ctx)
                ctx.active_skills.append(skill.name)

        return ctx

    def run_stream(self, user_input: str, session_id: int):
        """
        流式执行（供 SSE 使用），逐事件 yield 给前端。

        Yields 事件 dict：
          - {"type": "skill_activate", "name": "...", "description": "..."}
          - {"type": "skill_result", "name": "...", "data": {...}}
          - {"type": "token", "text": "..."}
          - {"type": "done", "response": "...", "active_skills": [...]}
        """
        ctx = SkillContext(user_input=user_input, session_id=session_id)

        for skill in self._skills:
            if ctx.aborted:
                break
            if not skill.should_activate(ctx):
                continue

            yield {"type": "skill_activate", "name": skill.name, "description": skill.description}
            ctx.active_skills.append(skill.name)

            # ── LLM Skill 特殊处理：流式 token 输出 ────────────────
            if isinstance(skill, LLMResponseSkill):
                full_response = ""
                for delta in skill.execute_stream(ctx):
                    full_response += delta
                    yield {"type": "token", "text": delta}
                ctx.response = full_response
                yield {
                    "type": "skill_result",
                    "name": skill.name,
                    "data": {"response": full_response},
                }
                continue

            # ── 普通 Skill：一次性执行 ──────────────────────────────
            skill.execute(ctx)

            # 收集 skill 结果用于 SSE 推送
            result_data = {}
            if skill.name == "memory_load":
                result_data = {
                    "layers": [
                        {"name": l["name"], "source": l["source"], "chars": l["chars"]}
                        for l in ctx.layers_info
                    ],
                    "total_chars": sum(l["chars"] for l in ctx.layers_info),
                }
            elif skill.name == "semantic_search":
                result_data = {
                    "query": user_input,
                    "results": [
                        {
                            "category": r.get("category", ""),
                            "title": r.get("title", ""),
                            "content": r.get("content", "")[:120],
                            "score": round(r["score"], 3),
                            "source": r.get("source", ""),
                        }
                        for r in ctx.semantic_results
                    ],
                }
            elif skill.name == "session_history":
                result_data = {"history_turns": len(ctx.history)}
            elif skill.name == "context_assembly":
                result_data = {
                    "system_chars": ctx.extras.get("system_chars", 0),
                    "history_turns": ctx.extras.get("history_turns", 0),
                    "layers_used": [l["name"] for l in ctx.layers_info]
                    + (["faiss_semantic"] if ctx.semantic_results else []),
                }
            elif skill.name == "save_message":
                result_data = {
                    "message_count": ctx.extras.get("message_count", 0),
                }
            elif skill.name == "word_card":
                wc = ctx.extras.get("word_card", {})
                result_data = {
                    "mode": wc.get("mode", "unknown"),
                    "word_card": wc,
                }

            yield {"type": "skill_result", "name": skill.name, "data": result_data}

        yield {
            "type": "done",
            "response": ctx.response,
            "session_id": session_id,
            "active_skills": ctx.active_skills,
            "message_count": ctx.extras.get("message_count", 0),
        }
