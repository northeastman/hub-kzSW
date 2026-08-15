"""
主 Agent + 并行 Reviewer 编排（代码审查 subagent 项目）

核心逻辑：
  1. 主 agent 有 2 个工具：
     - scan_directory：扫目录列代码文件（第一步了解项目结构）
     - dispatch_reviewers：派发多个 reviewer 并行审查（按文件拆分）
     主 agent 根据目录内容自主挑关键文件——不是固定拓扑，是 LLM 自主决策
  2. 每个 reviewer subagent 也是 ReAct 循环（只 read_file 工具），
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单reviewer时长)，
     而非 sum——这就是 subagent 并行的核心价值
  3. 每个 reviewer 的 trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」

架构对应 PPT Part 6.3 的 Orchestrator-Workers 拓扑（动态：主 agent 决定审哪些文件）。
"""

import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from file_tools import scan_directory, read_file, format_scan_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是代码审查主审查员。你有 2 个工具：
- scan_directory：扫描目录列出代码文件（参数=目录路径）。用于第一步了解项目结构
- dispatch_reviewers：派发多个子审查员并行审查（参数=用 | 分隔的多个文件路径）

【关键决策原则】
- 第一步必须 scan_directory 看项目里有哪些代码文件
- 拿到文件列表后，挑出关键源码文件（跳过测试文件 test_*.py、配置文件、__init__.py 等），
  最多挑 8 个文件，用 dispatch_reviewers 把文件路径用 | 分隔派发。
  注意：dispatch_reviewers 的文件路径要用 scan_directory 返回的「项目根目录 + 相对路径」拼成完整路径，
  或直接用相对项目根目录的路径（工具会解析）。
  示例：Action: dispatch_reviewers
        Action Input: src/main.py | src/utils.py | app/server.py
- 只有一个文件时也可直接派发（1 个 reviewer）
- 收齐子审查结果后，综合成结构化报告：
  按文件分组，每组列出问题（严重级别 + 行号 + 描述 + 建议），
  末尾给整体结论与最严重问题 Top 清单

【示例】
Question: 审查项目 /path/to/project
Thought: 先扫描目录看有哪些代码文件
Action: scan_directory
Action Input: /path/to/project
Observation: 项目根目录: /path/to/project\n找到 5 个代码文件:\n  - src/main.py (120 行)...
Thought: 找到 5 个关键文件，派发子审查员并行审查，跳过测试和 __init__
Action: dispatch_reviewers
Action Input: /path/to/project/src/main.py | /path/to/project/src/utils.py | /path/to/project/src/db.py
Observation: 并行审查完成：3 个子审查员...（各文件审查结果）
Thought: 已收齐三个文件的并行审查结果，综合成报告
Final Answer: （分文件报告 + 整体结论）"""

REVIEWER_SYSTEM = """你是代码审查员，负责审查单个代码文件。你有 1 个工具：
- read_file：读取文件内容（参数=文件路径）

【审查维度】一次性覆盖以下维度（不按维度拆分）：
- 正确性/Bug：逻辑错误、边界条件、空指针/None、异常吞掉（裸 except）
- 安全：注入（SQL/命令）、硬编码密钥/密码、不安全反序列化、路径穿越
- 性能：低效循环、重复计算、资源未释放（文件/连接未关闭）
- 可读性/规范：命名不清晰、重复代码、魔法数字、缺少关键注释

【输出格式】
每条问题用如下格式（一行一条）：
[严重级别 高/中/低] 行号 — 问题描述 → 修复建议

文件无明显问题时明确说明「未发现明显问题」。

【示例】
Question: 审查文件 src/main.py
Thought: 读取文件内容
Action: read_file
Action Input: src/main.py
Observation: （带行号的文件内容）
Thought: 已读取完整文件，逐行审查发现 3 个问题
Final Answer:
[高] 行23 — SQL 查询用字符串拼接 user_input，存在注入风险 → 改用参数化查询
[中] 行45 — 裸 except: 吞掉所有异常，难以调试 → 改为 except Exception as e: 并记录日志
[低] 行67 — 魔法数字 86400，可读性差 → 定义常量 SECONDS_PER_DAY = 86400"""


def _dispatch_reviewers(action_input: str, shared_state: dict = None,
                        on_reviewer_step: Callable = None,
                        on_reviewer_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_reviewers 工具实现。
    action_input: "文件1 | 文件2 | ..."（管道分隔）
    派发 N 个 reviewer 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs serial_sum。
    ⚠️ 用真实 reviewer id 发 dispatch 事件（与 reviewer_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    file_paths = [s.strip() for s in action_input.split("|") if s.strip()][:8]
    if not file_paths:
        return "未解析出文件路径"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("reviewers", {})

    # 构造 (rid, reviewer, filepath) 三元组
    defs = []
    for fpath in file_paths:
        rid = f"rev_{uuid.uuid4().hex[:6]}"
        rev = ReActLoop(
            agent_name=rid,
            tools={"read_file": (lambda p, **_: read_file(p),
                                 "读取文件内容，参数是文件路径")},
            max_steps=4, model_tag="qwen-plus(reviewer)",
            system_prompt=REVIEWER_SYSTEM)
        defs.append((rid, rev, fpath))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 reviewer id
    dispatch_info = {"file_paths": file_paths,
                     "reviewer_ids": [rid for rid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 reviewer_step 对得上

    t0 = time.time()
    results = {}
    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for 循环) ──
    def _run_one(sid=None, sub=None, topic=None):
        return sid, sub.run(f"审查文件 {topic}", on_step=(
            lambda step, sid=sid: on_reviewer_step(sid, step) if on_reviewer_step else None))

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for rid, rev, fpath in defs:
            rid, res = _run_one(rid, rev, fpath)
            fpath = next(f for r, _, f in defs if r == rid)
            results[rid] = (fpath, res)
            shared_state["reviewers"][rid] = {
                "file_path": fpath, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_reviewer_done:
                on_reviewer_done(rid, res["duration"], fpath)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, rid, rev, fpath): rid for rid, rev, fpath in defs}
            for fut in as_completed(futs):
                rid, res = fut.result()
                fpath = next(f for r, _, f in defs if r == rid)
                results[rid] = (fpath, res)
                shared_state["reviewers"][rid] = {
                    "file_path": fpath, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_reviewer_done:
                    on_reviewer_done(rid, res["duration"], fpath)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_reviewers": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【文件: {fpath}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for rid, (fpath, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行审查完成：{len(defs)} 个审查员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_review(project_dir: str, on_main_step: Callable = None,
               on_reviewer_step: Callable = None,
               on_reviewer_done: Callable = None,
               on_dispatch: Callable = None,
               serial: bool = False) -> dict:
    """执行一次代码审查。返回 {final_answer, main_trace, reviewers, parallel_stats, dispatches}。
    serial=True 时 reviewer 串行执行（eval A/B 对比基线）。"""
    shared_state = {"reviewers": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        # dispatch 事件由 _dispatch_reviewers 用真实 reviewer id 发出
        return _dispatch_reviewers(action_input, shared_state=info,
                                   on_reviewer_step=on_reviewer_step,
                                   on_reviewer_done=on_reviewer_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "scan_directory": (lambda p, **_: format_scan_result(scan_directory(p)),
                               "扫描目录列出代码文件，参数=目录路径"),
            "dispatch_reviewers": (dispatch_tool,
                                   "派发多个子审查员并行审查，参数=用 | 分隔的多个文件路径"),
        },
        max_steps=8,
        model_tag="qwen-plus(主)",
        system_prompt=MAIN_SYSTEM,   # ← 传主 agent 的派发引导 prompt
    )
    # 把 shared_state 注入主 agent run
    result = main.run(f"审查项目 {project_dir}", on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "reviewers": shared_state["reviewers"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    # 测试：审查 sample_project 目录
    project = os.path.join(os.path.dirname(__file__), "..", "sample_project")
    r = run_review(project)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | reviewer 数: {len(r['reviewers'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:300]}")
