"""
后台回顾 Agent（优化版）：在原版基础上压缩提示词。

优化点（方案 5）：
  - 原版把 policies.md 全文 + 当前所有 Skill 全文都塞进 prompt，单次 ~6K token
  - 优化版根据失败样本的关键词，仅注入相关 policies 小节
  - 当前 Skill 只在 system prompt 出现一次，user 里只放失败样本（避免重复）
  - teaching_mode 开关：
      True  → 全量注入（原版行为）
      False → 按需注入相关小节
  - 统计压缩前后字符数，便于对比

教学重点（保留原版）：
  1. 后台回顾 Agent 是独立的 LLM 调用，与主 Agent 解耦（Hermes 是异步 spawn）
  2. 回顾 Agent 拥有 policies.md（完整政策），主 Agent 没有——体现权限分层
  3. 回顾 Agent 输出结构化 JSON，由 SkillManager 执行 create/patch
"""

import os
import json
import re
from openai import OpenAI
from pathlib import Path
from skill_manager import SkillManager


# ── 政策小节路由表：关键词 → 章节标题片段 ────────────────────────────────────
# policies.md 的章节结构（基于实际文件内容）
POLICY_SECTION_RULES: list[tuple[list[str], list[str]]] = [
    # 退款相关
    (["退", "签收", "完好", "质量问题"], ["1.1", "普通商品退款"]),
    (["VIP", "银卡", "金卡", "白卡"], ["1.2", "VIP用户退款特权", "2.", "VIP会员等级"]),
    (["限时", "特惠", "满减", "优惠券", "促销"], ["1.3", "促销商品特殊规则"]),
    (["电子书", "激活码", "会员卡", "点卡", "数字商品"], ["1.4", "数字商品规则"]),
    (["积分", "抵扣"], ["1.5", "积分抵扣订单"]),
    (["物流", "快递", "发货", "取消订单", "拦截"], ["三、", "物流"]),
    (["支付", "账户", "银行卡", "余额", "到账"], ["五、", "支付", "退款到账"]),
]

# 永远注入的章节（基础退款规则，作为判定基准）
ALWAYS_INCLUDE_SECTIONS = ["1.1", "普通商品退款"]


REVIEWER_SYSTEM = """你是云购商城客服系统的"技能优化专家"。

以下给你的全部都是 Agent 最近一轮中**答错或推脱**的样本。你要做的是用最小改动
修复它们，让下次遇到同类问题能答对。

## 核心原则（严格遵守）

1. **仅修复观察到的失败**：只针对输入样本里出现的问题类型做改动，不要扩展到
   "政策里还有但样本里没出现"的场景
2. **最小改动优先**：
   - 能在已有 Skill 里追加或改一条分支解决的，不要新建 Skill
   - patch 的 old_text 只包含要改的那几行，不要把整段抄下来重写
3. **聚焦核心**：如果失败涉及多种类型，按失败条数从高到低，**只修复 1~2 类**
   —— 留出进化梯度，不要一次改完所有问题

你拥有相关政策的节选，仅用于**核对 Agent 答错的具体数字/规则是否与政策一致**，
它是判定标准，不是 Agent 的知识补全大纲。

## 相关政策节选（判定标准）
{policies}

## 当前已有 Skill
{current_skills_summary}

## 输出格式
{{
  "analysis": "本轮失败 N 条，主要失败类型是 XXX",
  "actions": [
    {{"action": "create", "skill_name": "...", "reason": "修复哪条失败",
      "content": "完整SKILL.md（含frontmatter）"}},
    {{"action": "patch",  "skill_name": "...", "reason": "修复哪条失败",
      "old_text": "精确的原始文本", "new_text": "替换文本"}}
  ]
}}

只输出 JSON，不要有其他文字。如果发现失败数其实很少、没有清晰模式，可以只返回
1 条 action 甚至 0 条。"""

REVIEWER_USER = """## 本轮失败样本（共 {n} 条，都是 Agent 答错或推脱的）

{history_text}

## 当前 Skill 完整内容
{current_skills_full}

按核心原则给出最小必要的修复方案。"""


class BackgroundReviewer:
    def __init__(
        self,
        policies_path: str,
        skill_manager: SkillManager,
        model: str = "deepseek-chat",
        teaching_mode: bool = False,
    ):
        """
        teaching_mode:
          True  → 全量注入 policies.md（原版行为）
          False → 仅注入与失败样本相关的 policies 小节
        """
        self.policies_full = Path(policies_path).read_text(encoding="utf-8")
        self.skill_manager = skill_manager
        self.model = model
        self.teaching_mode = teaching_mode
        self.last_analysis = ""   # UI 可读取最近一次回顾的分析文本
        # 统计字段
        self.compression_stats = {
            "full_chars": 0,
            "compressed_chars": 0,
            "compression_count": 0,
        }
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def review(self, failed_turns: list[dict]) -> list[dict]:
        """
        分析失败样本列表，返回最小必要的 Skill 操作。
        调用方应**仅传入本轮失败的条目**（每条形如 {question, answer, fail_reason}）；
        空列表直接返回 []，不做 LLM 调用。
        """
        if not failed_turns:
            return []

        current_skills = self.skill_manager.load_all()
        skills_summary = "\n".join(
            f"- {name}: {self._extract_description(content)}"
            for name, content in sorted(current_skills.items())
        ) or "（暂无已有Skill）"
        skills_full = "\n\n---\n\n".join(
            f"### {name}\n{content}" for name, content in sorted(current_skills.items())
        ) or "（暂无已有Skill）"

        # ── 按需选择 policies 小节 ──────────────────────────────────────
        if self.teaching_mode:
            policies_text = self.policies_full
        else:
            policies_text = self._select_relevant_policies(failed_turns)

        # 统计压缩效果
        full_len = len(self.policies_full)
        comp_len = len(policies_text)
        self.compression_stats["full_chars"] += full_len
        self.compression_stats["compressed_chars"] += comp_len
        self.compression_stats["compression_count"] += 1

        system_msg = REVIEWER_SYSTEM.format(
            policies=policies_text,
            current_skills_summary=skills_summary,
        )
        user_msg = REVIEWER_USER.format(
            n=len(failed_turns),
            history_text=self._format_history(failed_turns),
            current_skills_full=skills_full,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=3000,
        )
        return self._parse_actions(response.choices[0].message.content.strip())

    def _select_relevant_policies(self, failed_turns: list[dict]) -> str:
        """
        按需注入：根据失败样本的关键词，从 policies.md 切出相关小节。
        策略：
          1. 收集所有失败样本的 question + fail_reason 文本
          2. 按 POLICY_SECTION_RULES 匹配相关章节标题
          3. 永远包含基础退款规则（ALWAYS_INCLUDE_SECTIONS）
          4. 按章节标题切分 policies.md，只保留命中章节的内容
          5. 如果没有任何命中，回退到全量（保证安全）
        """
        # 合并所有失败样本文本作为匹配源
        all_text = " ".join(
            t.get("question", "") + " " + t.get("fail_reason", "") + " " + t.get("answer", "")
            for t in failed_turns
        )

        # 收集命中的章节关键词
        hit_section_keys: set[str] = set()
        for keywords, section_titles in POLICY_SECTION_RULES:
            if any(kw in all_text for kw in keywords):
                hit_section_keys.update(section_titles)

        # 永远包含基础章节
        hit_section_keys.update(ALWAYS_INCLUDE_SECTIONS)

        if not hit_section_keys:
            # 没有任何命中，回退到全量
            return self.policies_full

        # 切分 policies.md 为章节块
        sections = self._split_policy_sections(self.policies_full)

        # 拼接命中章节
        selected: list[str] = []
        for section in sections:
            title = section.split("\n", 1)[0]
            if any(key in title for key in hit_section_keys):
                selected.append(section)

        if not selected:
            return self.policies_full

        # 顶部加一个简短的全局说明
        header = "# 云购商城客服政策手册（节选：与本次失败样本相关的小节）\n\n"
        return header + "\n\n---\n\n".join(selected)

    def _split_policy_sections(self, text: str) -> list[str]:
        """
        按 markdown 标题（### 或 ##）切分 policies.md。
        保留每个标题下的内容块。
        """
        # 匹配 ## 或 ### 开头的行
        pattern = re.compile(r"^(#{2,3}\s+.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            return [text]

        sections: list[str] = []
        # 标题前的内容（通常是 # 一级标题或前言）
        if matches[0].start() > 0:
            sections.append(text[: matches[0].start()].strip())

        # 各章节
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(text[start:end].strip())

        return [s for s in sections if s]

    def _format_history(self, turns: list[dict]) -> str:
        lines = []
        for i, t in enumerate(turns, 1):
            lines.append(f"[{i}] 用户: {t['question']}")
            lines.append(f"    客服: {t['answer'][:200]}{'...' if len(t['answer']) > 200 else ''}")
            # 评估结果（如果调用方附带了失败原因）
            if t.get("fail_reason"):
                lines.append(f"    ✗ 判定：{t['fail_reason']}")
        return "\n".join(lines)

    def _extract_description(self, content: str) -> str:
        m = re.search(r"description:\s*(.+)", content)
        return m.group(1).strip() if m else "(无描述)"

    def _parse_actions(self, raw: str) -> list[dict]:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                print(f"  [Reviewer] 无法提取 JSON，原始输出：{raw[:200]}")
                self.last_analysis = ""
                return []
            data = json.loads(json_match.group())
            self.last_analysis = data.get("analysis", "")
            print(f"  [Reviewer] 分析：{self.last_analysis[:100]}")
            return data.get("actions", [])
        except json.JSONDecodeError as e:
            print(f"  [Reviewer] JSON 解析失败: {e}\n原始: {raw[:300]}")
            self.last_analysis = ""
            return []

    def compression_summary(self) -> dict:
        """返回 policies 压缩统计，便于教学展示。"""
        n = self.compression_stats["compression_count"]
        full_avg = self.compression_stats["full_chars"] / n if n else 0
        comp_avg = self.compression_stats["compressed_chars"] / n if n else 0
        save_rate = (1 - comp_avg / full_avg) if full_avg else 0.0
        return {
            "mode": "teaching (full policies)" if self.teaching_mode else "optimized (sectioned policies)",
            "review_calls": n,
            "avg_full_chars": round(full_avg),
            "avg_compressed_chars": round(comp_avg),
            "avg_save_rate": round(save_rate, 3),
        }
