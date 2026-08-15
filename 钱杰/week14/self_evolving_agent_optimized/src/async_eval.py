"""
并发评估工具（优化版）：用 asyncio.Semaphore 控制并发跑 Agent 评估。

优化点（方案 3）：
  - 原版 baseline/final/probe 都是串行跑 60/30 题，墙钟时间长
  - 优化版用 asyncio + asyncio.Semaphore 并发跑，Agent 无状态（每题独立）天然支持
  - 提供两种回调模式：
      - run_eval_concurrent_silent:    静默模式，等所有题跑完返回
      - run_eval_concurrent_streaming: 流式模式，每题完成立即回调（供 SSE 用）
  - teaching_mode=True 时退化为串行（原版行为）

注意：
  - DeepSeek API 默认支持并发，建议并发数 5（避免限流）
  - SSE 流式输出顺序会乱，回调里带 qid 让客户端按 id 排序
  - Agent.answer() 是同步阻塞调用，用 asyncio.to_thread 包装成异步
"""

import asyncio
from typing import Callable, Awaitable


async def run_eval_concurrent_silent(
    agent,
    evaluator,
    question_ids: list[int],
    max_concurrency: int = 5,
    teaching_mode: bool = False,
) -> dict:
    """
    并发跑评估，静默返回完整结果。

    Args:
        agent: CustomerServiceAgent 实例（需有 .answer(question) -> str）
        evaluator: Evaluator 实例（需有 .questions 和 .evaluate_answer）
        question_ids: 要跑的题目 id 列表
        max_concurrency: 最大并发数（建议 5）
        teaching_mode: True 时退化为串行（原版行为）

    Returns:
        与原版 run_probe_eval 同构的结果字典
    """
    if teaching_mode or max_concurrency <= 1:
        # 教学模式：串行跑，保持原版行为
        return await _run_eval_serial(agent, evaluator, question_ids)

    sem = asyncio.Semaphore(max_concurrency)
    results: dict[int, tuple[str, bool, str]] = {}

    async def one(qid: int):
        async with sem:
            q = evaluator.questions[qid]
            # Agent.answer 是同步阻塞，用 to_thread 包装
            answer = await asyncio.to_thread(agent.answer, q["question"])
            ok, reason = evaluator.evaluate_answer(answer, qid)
            results[qid] = (answer, ok, reason)

    await asyncio.gather(*[one(qid) for qid in question_ids])

    # 按题目顺序汇总
    return _aggregate_results(results, evaluator, question_ids)


async def run_eval_concurrent_streaming(
    agent,
    evaluator,
    question_ids: list[int],
    on_result: Callable[[int, str, bool, str], Awaitable[None]],
    max_concurrency: int = 5,
    teaching_mode: bool = False,
) -> dict:
    """
    并发跑评估，每题完成立即通过 on_result 回调通知（供 SSE 流式用）。

    Args:
        on_result: async 回调 (qid, answer, correct, fail_reason) -> None
        其余参数同 run_eval_concurrent_silent
    """
    if teaching_mode or max_concurrency <= 1:
        # 教学模式：串行跑，每题完成立即回调
        results: dict[int, tuple[str, bool, str]] = {}
        for qid in question_ids:
            q = evaluator.questions[qid]
            answer = await asyncio.to_thread(agent.answer, q["question"])
            ok, reason = evaluator.evaluate_answer(answer, qid)
            results[qid] = (answer, ok, reason)
            await on_result(qid, answer, ok, reason)
        return _aggregate_results(results, evaluator, question_ids)

    sem = asyncio.Semaphore(max_concurrency)
    results: dict[int, tuple[str, bool, str]] = {}

    async def one(qid: int):
        async with sem:
            q = evaluator.questions[qid]
            answer = await asyncio.to_thread(agent.answer, q["question"])
            ok, reason = evaluator.evaluate_answer(answer, qid)
            results[qid] = (answer, ok, reason)
            # 每题完成立即回调（SSE 友好）
            await on_result(qid, answer, ok, reason)

    await asyncio.gather(*[one(qid) for qid in question_ids])

    return _aggregate_results(results, evaluator, question_ids)


async def _run_eval_serial(agent, evaluator, question_ids: list[int]) -> dict:
    """串行评估（原版行为，教学模式用）。"""
    results: dict[int, tuple[str, bool, str]] = {}
    for qid in question_ids:
        q = evaluator.questions[qid]
        answer = await asyncio.to_thread(agent.answer, q["question"])
        ok, reason = evaluator.evaluate_answer(answer, qid)
        results[qid] = (answer, ok, reason)
    return _aggregate_results(results, evaluator, question_ids)


def _aggregate_results(
    results: dict[int, tuple[str, bool, str]],
    evaluator,
    question_ids: list[int],
) -> dict:
    """把并发的 per-question 结果聚合成 summary 格式（与原版兼容）。"""
    total = 0
    correct = 0
    by_category: dict = {}
    answers: dict = {}

    for qid in question_ids:
        answer, ok, reason = results[qid]
        q = evaluator.questions[qid]
        cat = q["category"]
        total += 1
        by_category.setdefault(cat, {"total": 0, "correct": 0})
        by_category[cat]["total"] += 1
        if ok:
            correct += 1
            by_category[cat]["correct"] += 1
        answers[str(qid)] = {"answer": answer, "correct": ok, "fail_reason": reason if not ok else ""}

    for v in by_category.values():
        v["accuracy"] = round(v["correct"] / v["total"], 3)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "by_category": by_category,
        "answers": answers,
    }
