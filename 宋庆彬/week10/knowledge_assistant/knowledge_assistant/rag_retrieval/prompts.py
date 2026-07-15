from __future__ import annotations


def build_answer_prompt(context: str, history: str, question: str, support_phone: str) -> str:
    return f"""
你是一个智能学习助手，负责根据知识库和对话历史回答用户问题。

请遵守以下规则：
1. 如果上下文包含答案，优先基于上下文回答，并说明“根据知识库资料”。
2. 如果对话历史与当前问题相关，可以结合历史补全指代和背景。
3. 如果上下文和历史都不足以回答，请回复：信息不足，无法回答，请联系人工客服，电话：{support_phone}。
4. 回答要清晰、准确，不编造课程、价格、老师、服务承诺等事实。

上下文：
{context}

对话历史：
{history}

问题：
{question}

回答：
""".strip()


def build_hyde_prompt(query: str) -> str:
    return f"请为下面的问题生成一个简短、可能正确的假设答案，用于检索。\n问题：{query}\n假设答案："


def build_subquery_prompt(query: str) -> str:
    return f"请将下面复杂问题拆成 2 到 4 个可独立检索的子问题，每行一个。\n问题：{query}\n子问题："


def build_backtracking_prompt(query: str) -> str:
    return f"请将下面问题改写成一个更基础、更容易检索的问题，只输出改写后的问题。\n问题：{query}\n改写："


def build_strategy_prompt(query: str) -> str:
    return f"""
从以下检索策略中为用户问题选择一个最合适的策略，只输出策略名称：
- 直接检索：问题明确，查询某个具体信息。
- 假设答案检索：问题抽象，适合先生成假设答案再检索。
- 子问题检索：问题包含多个方面，需要拆分。
- 回溯检索：问题复杂，需要先简化成基础问题。

用户问题：{query}
策略：
""".strip()

