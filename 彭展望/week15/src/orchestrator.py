"""Orchestrator（主 Agent）—— 下发 subagent 并行完成多项工作。

三步流水线（Orchestrator-Workers 拓扑）：
  1. plan()      : 让 LLM 把一个复杂任务动态拆解成 N 个互相独立的子任务（不是硬编码）
  2. dispatch()  : 用 ThreadPoolExecutor 把 N 个 subagent 并行跑起来
  3. synthesize(): 让 LLM 把 N 份子结果汇总成一份最终交付物

并行价值：N 个子任务的 wall-clock ≈ max(单个耗时) 而非 sum(全部耗时)。
本模块同时提供 serial 版本用于对照，量化并行加速比。
"""
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import llm_chat
from subagent import SubTask, SubResult, run_subagent

logger = logging.getLogger(__name__)

PLAN_SYSTEM = """你是一个多智能体系统的主调度器（orchestrator）。
用户会给你一个较复杂的任务。你的职责不是自己回答，而是把它拆解成若干个
【互相独立、可并行、无先后依赖】的子任务，之后每个子任务会交给一个子智能体并行完成。

拆解要求：
- 子任务数量控制在 3~5 个，覆盖任务的不同侧面，彼此不重叠、合起来完整。
- 每个子任务必须能被单独执行，不依赖其他子任务的输出。
- 只输出 JSON，不要任何解释或 markdown 代码围栏。

输出格式（严格）：
{"subtasks": [
  {"title": "子任务简短标题", "instruction": "交给子智能体的具体、自足的指令"},
  ...
]}"""

SYNTH_SYSTEM = """你是多智能体系统的主调度器。多个子智能体已并行完成了各自的子课题，
下面给你它们的结果。请你把它们综合成一份结构清晰、逻辑连贯的最终交付物：
- 按子课题分小节组织，去重、消解冲突。
- 开头一句话总览，结尾给出关键结论与不确定性提示。
- 不要简单罗列，要有整合与提炼。"""


class Orchestrator:
    def __init__(self, max_workers: int = 5, verbose: bool = True):
        self.max_workers = max_workers
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # ---------- 1. 规划：动态拆解子任务 ----------
    def plan(self, task: str) -> list[SubTask]:
        self._log(f"\n[主Agent] 正在拆解任务：{task}")
        raw = llm_chat(PLAN_SYSTEM, task, temperature=0.2, max_tokens=800)
        subtasks = self._parse_plan(raw)
        self._log(f"[主Agent] 拆解出 {len(subtasks)} 个可并行子任务：")
        for st in subtasks:
            self._log(f"    - 子{st.id}: {st.title}")
        return subtasks

    @staticmethod
    def _parse_plan(raw: str) -> list[SubTask]:
        # 容错：剥掉可能的 ```json 围栏
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        data = json.loads(text)
        out = []
        for i, item in enumerate(data["subtasks"], 1):
            out.append(SubTask(id=i, title=item["title"].strip(),
                               instruction=item["instruction"].strip()))
        return out

    # ---------- 2a. 并行下发 ----------
    def dispatch_parallel(self, subtasks: list[SubTask]) -> tuple[list[SubResult], float]:
        self._log(f"\n[主Agent] 并行下发 {len(subtasks)} 个 subagent（线程池 max_workers={self.max_workers}）...")
        t0 = time.perf_counter()
        results: list[SubResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers,
                                thread_name_prefix="subagent") as pool:
            futures = {pool.submit(run_subagent, st, self._log): st for st in subtasks}
            for fut in as_completed(futures):
                results.append(fut.result())
        wall = time.perf_counter() - t0
        results.sort(key=lambda r: r.id)
        self._log(f"[主Agent] 全部 subagent 并行完成，墙钟耗时 {wall:.1f}s")
        return results, wall

    # ---------- 2b. 串行下发（仅用于对照） ----------
    def dispatch_serial(self, subtasks: list[SubTask]) -> tuple[list[SubResult], float]:
        t0 = time.perf_counter()
        results = [run_subagent(st) for st in subtasks]
        return results, time.perf_counter() - t0

    # ---------- 3. 汇总 ----------
    def synthesize(self, task: str, results: list[SubResult]) -> str:
        self._log("\n[主Agent] 汇总各 subagent 结果，生成最终交付物...")
        blocks = []
        for r in results:
            body = r.content if not r.error else f"（该子任务失败：{r.error}）"
            blocks.append(f"### 子课题：{r.title}\n{body}")
        user = f"原始任务：{task}\n\n以下是各子智能体的并行调研结果：\n\n" + "\n\n".join(blocks)
        return llm_chat(SYNTH_SYSTEM, user, temperature=0.4, max_tokens=1600)

    # ---------- 完整流水线 ----------
    def run(self, task: str) -> dict:
        overall = time.perf_counter()
        subtasks = self.plan(task)
        results, parallel_wall = self.dispatch_parallel(subtasks)
        final = self.synthesize(task, results)
        total = time.perf_counter() - overall
        serial_est = sum(r.elapsed for r in results)  # 串行理论耗时 = 各子任务之和
        return {
            "task": task,
            "subtasks": [st.__dict__ for st in subtasks],
            "results": [r.__dict__ for r in results],
            "final_report": final,
            "timing": {
                "parallel_wall_sec": round(parallel_wall, 2),
                "serial_sum_sec": round(serial_est, 2),
                "speedup": round(serial_est / parallel_wall, 2) if parallel_wall else None,
                "total_pipeline_sec": round(total, 2),
                "per_subagent": [
                    {"id": r.id, "title": r.title, "elapsed": round(r.elapsed, 2)}
                    for r in results
                ],
            },
        }
