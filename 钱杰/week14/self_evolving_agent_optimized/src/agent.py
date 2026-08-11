"""
客服 Agent（优化版）：在原版基础上实现 Skill 按需注入。

优化点（方案 1）：
  - 每次应答前，先根据 question 关键词路由选择相关 Skill 子集注入 system prompt
  - 始终兜底注入 refund Skill，避免边界场景漏判
  - 通过 teaching_mode 开关切换：
      True  → 全量注入（原版行为，教学透明）
      False → 按需注入（生产优化，省 50%+ token）
  - 统计注入 token 数，便于对比优化效果

教学重点（保留原版）：
  1. Agent 的唯一知识来源是 Skill 文件，不直接访问 policies.md
  2. Nudge 计数器：每 N 次调用后触发后台回顾（对应 Hermes _iters_since_skill）
  3. 每次回答前动态加载最新 Skills，保证进化后立即生效
"""

import os
from openai import OpenAI
from skill_manager import SkillManager


SYSTEM_TEMPLATE = """你是云购商城的智能客服助手。

你的所有知识来源于以下技能文档，严格基于文档内容回答，不要自行推断或编造政策。

## 回答规则（严格遵守）
- 【能回答】如果技能文档覆盖了用户问题：直接给出完整具体的答案（含具体天数/金额/
  工作日数等政策细节）。**不要在答案中加"建议联系人工客服"之类的推脱话**。
- 【不能回答】如果技能文档确实不覆盖：**仅回答一句** "需要联系人工客服"，
  不要编造答案，也不要列举可能的情况。

{skills_section}
"""

SKILLS_SECTION_TEMPLATE = """## 当前知识库（共{count}个技能）

{skills_content}
"""


# ── Skill 路由表：关键词 → Skill 名 ──────────────────────────────────────────
# 顺序敏感：先匹配更具体的（如 digital_goods 优先于 refund）
SKILL_ROUTING_RULES: list[tuple[list[str], str]] = [
    # 数字商品优先匹配，避免被 refund 兜底吞掉
    (["电子书", "激活码", "会员卡", "游戏点卡", "点卡", "数字商品", "数字"], "digital_goods_refund"),
    (["VIP", "银卡", "金卡", "白卡"], "vip_benefits"),
    (["限时", "特惠", "满减", "优惠券", "券", "促销"], "promotions"),
    (["物流", "快递", "发货", "取消订单", "拦截"], "logistics"),
    (["支付", "账户", "积分", "银行卡", "余额"], "payment_account"),
    (["退", "退款", "退货", "签收"], "refund"),
    (["运费"], "refund"),
]


class CustomerServiceAgent:
    def __init__(
        self,
        skill_manager: SkillManager,
        nudge_interval: int = 20,
        model: str = "deepseek-chat",
        teaching_mode: bool = False,
    ):
        """
        teaching_mode:
          True  → 每次注入全部 Skill（原版行为，教学透明）
          False → 按需注入相关 Skill（生产优化，省 token）
        """
        self.skill_manager = skill_manager
        self.nudge_interval = nudge_interval
        self.model = model
        self.teaching_mode = teaching_mode
        self._iters_since_nudge = 0
        # conversation_history 仅为后台回顾 Agent 保留观察样本；
        # 主 Agent 每次 answer() 都不把它传给 LLM（保证每题独立评估）。
        self.conversation_history: list[dict] = []
        # 统计字段，便于展示优化效果
        self.routing_stats = {
            "full_injection_count": 0,   # 全量注入次数
            "partial_injection_count": 0, # 按需注入次数
            "total_skills_tokens": 0,    # 累计注入的 Skill 字符数
            "full_tokens_baseline": 0,   # 若全量注入会消耗的字符数（对比基准）
        }

        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def answer(self, question: str) -> str:
        """
        回答单个问题。每次调用都会重新加载最新 Skills（保证 Nudge 后立即生效），
        且 messages 里只含系统提示 + 当前问题（不携带 conversation_history），
        这样每次评估互不干扰。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt(question)},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=400,
        )
        answer_text = response.choices[0].message.content.strip()

        # 对话历史只做滚动观察窗口，供后台回顾 Agent 使用
        self.conversation_history.append({
            "question": question,
            "answer": answer_text,
            "skills_used": list(self.skill_manager.load_all().keys()),
        })
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

        self._iters_since_nudge += 1
        return answer_text

    def should_trigger_nudge(self) -> bool:
        """判断是否应该触发后台回顾。触发后计数器归零。"""
        if self.nudge_interval > 0 and self._iters_since_nudge >= self.nudge_interval:
            self._iters_since_nudge = 0
            return True
        return False

    def reset_nudge_counter(self):
        """Agent 主动调用 skill_manage 时手动归零，避免重复触发。"""
        self._iters_since_nudge = 0

    def _select_relevant_skills(self, question: str) -> dict[str, str]:
        """
        按需注入：根据 question 关键词路由选择相关 Skill 子集。
        - 始终兜底注入 refund（覆盖最广，避免漏判基础退款场景）
        - 多个 Skill 关键词同时命中时全部注入（如"VIP 退款"同时命中 vip_benefits + refund）
        - 如果没有任何命中，回退到全量注入（保证安全）
        """
        all_skills = self.skill_manager.load_all()
        selected: dict[str, str] = {}

        # 按规则匹配
        for keywords, skill_name in SKILL_ROUTING_RULES:
            if skill_name in all_skills and skill_name not in selected:
                if any(kw in question for kw in keywords):
                    selected[skill_name] = all_skills[skill_name]

        # 兜底：refund 始终注入（除非 question 明显与退款无关，这里简化为总是注入）
        if "refund" in all_skills and "refund" not in selected:
            selected["refund"] = all_skills["refund"]

        # 极端情况：没有任何 Skill 被选中（说明 Skill 库为空或关键词全不匹配）
        # → 回退到全量，保证安全
        if not selected:
            return all_skills

        return selected

    def _build_system_prompt(self, question: str) -> str:
        """构建 system prompt。根据 teaching_mode 决定全量或按需注入。"""
        if self.teaching_mode:
            # 教学模式：全量注入，保留原版透明度
            skills = self.skill_manager.load_all()
            self.routing_stats["full_injection_count"] += 1
        else:
            # 优化模式：按需注入
            skills = self._select_relevant_skills(question)
            self.routing_stats["partial_injection_count"] += 1

        # 统计 token 节省（用字符数近似）
        injected_chars = sum(len(c) for c in skills.values())
        all_chars = sum(len(c) for c in self.skill_manager.load_all().values())
        self.routing_stats["total_skills_tokens"] += injected_chars
        self.routing_stats["full_tokens_baseline"] += all_chars

        if not skills:
            skills_section = "（暂无技能文档，请依据通用客服原则回答）"
        else:
            parts = []
            for name, content in sorted(skills.items()):
                parts.append(f"### 技能：{name}\n{content}")
            skills_content = "\n\n---\n\n".join(parts)
            skills_section = SKILLS_SECTION_TEMPLATE.format(
                count=len(skills),
                skills_content=skills_content,
            )
        return SYSTEM_TEMPLATE.format(skills_section=skills_section)

    def routing_summary(self) -> dict:
        """返回路由统计，便于教学展示优化效果。"""
        s = self.routing_stats
        total_calls = s["full_injection_count"] + s["partial_injection_count"]
        saved_chars = s["full_tokens_baseline"] - s["total_skills_tokens"]
        save_rate = (saved_chars / s["full_tokens_baseline"]) if s["full_tokens_baseline"] else 0.0
        return {
            "mode": "teaching (full injection)" if self.teaching_mode else "optimized (partial injection)",
            "total_calls": total_calls,
            "full_injection_count": s["full_injection_count"],
            "partial_injection_count": s["partial_injection_count"],
            "injected_chars": s["total_skills_tokens"],
            "baseline_chars_if_full": s["full_tokens_baseline"],
            "saved_chars": saved_chars,
            "char_save_rate": round(save_rate, 3),
        }
