"""
主 Agent + 并行 Subagent 编排

核心能力：
  1. 主 agent（ReAct）持有 web_search + dispatch_subagents 两个工具
  2. dispatch_subagents 用 ThreadPoolExecutor 并行派发 N 个子 agent
  3. 每个 subagent 也是 ReAct 循环，独立完成任务后汇总
"""
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from search import format_search_result, tavily_search

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实问题
- dispatch_subagents：派发多个子 agent 并行调研（参数=用 | 分隔的多个子课题）

【决策原则】
- 问题涉及 2 个及以上侧面（调研/分析/概况/趋势等）→ 必须用 dispatch_subagents 并行派发
- 单一事实问题 → 直接 web_search
- 收齐子 agent 结果后，综合成结构化报告

【示例】
Question: 2024中国咖啡市场调研：市场规模、主要品牌、消费趋势
Thought: 三个侧面，必须并行派发子 agent
Action: dispatch_subagents
Action Input: 2024中国咖啡市场规模 | 主要品牌竞争格局 | 消费趋势与人群
Observation: 并行调研完成...
Thought: 已收齐结果，综合报告
Final Answer: （分维度报告）"""


def _make_subagent(sid: str) -> ReActLoop:
    return ReActLoop(
        agent_name=sid,
        tools={
            "web_search": (
                lambda q, **_: format_search_result(tavily_search(q)),
                "联网搜索，参数是查询词",
            )
        },
        max_steps=4,
        model_tag="deepseek-chat(子)",
    )


def dispatch_subagents(
    action_input: str,
    shared_state: dict = None,
    on_subagent_step: Callable = None,
    on_subagent_done: Callable = None,
    on_dispatch: Callable = None,
    serial: bool = False,
) -> str:
    """派发 N 个 subagent 并行（或串行）执行，返回汇总 Observation。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"

    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        defs.append((sid, _make_subagent(sid), topic))

    dispatch_info = {
        "subtopics": subtopics,
        "subagent_ids": [sid for sid, _, _ in defs],
    }
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid, sub, topic):
        res = sub.run(
            topic,
            on_step=(lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None),
        )
        return sid, topic, res

    if serial:
        for sid, sub, topic in defs:
            sid, topic, res = _run_one(sid, sub, topic)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic,
                "trace": res["trace"],
                "duration": res["duration"],
                "final_answer": res["final_answer"],
            }
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futures = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futures):
                sid, topic, res = fut.result()
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic,
                    "trace": res["trace"],
                    "duration": res["duration"],
                    "final_answer": res["final_answer"],
                }
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    speedup = round(serial_sum / wall, 2) if wall else 0
    shared_state.setdefault("parallel_stats", []).append(
        {
            "n_subagents": len(defs),
            "wall_clock": wall,
            "serial_sum": serial_sum,
            "speedup": speedup,
        }
    )

    parts = [
        f"【子课题: {topic}】(用时 {r['duration']}s)\n{r['final_answer'][:500]}"
        for sid, (topic, r) in results.items()
    ]
    return (
        f"并行调研完成：{len(defs)} 个子 agent，wall-clock {wall}s "
        f"(串行需 {serial_sum}s，加速 {speedup}×)\n\n" + "\n\n".join(parts)
    )


def run_research(
    question: str,
    on_main_step: Callable = None,
    on_subagent_step: Callable = None,
    on_subagent_done: Callable = None,
    on_dispatch: Callable = None,
    serial: bool = False,
) -> dict:
    """执行一次调研任务，返回主 agent 答案、trace 与并行统计。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        return dispatch_subagents(
            action_input,
            shared_state=shared_state or {},
            on_subagent_step=on_subagent_step,
            on_subagent_done=on_subagent_done,
            on_dispatch=on_dispatch,
            serial=serial,
        )

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (
                lambda q, **_: format_search_result(tavily_search(q)),
                "联网搜索一次，参数=查询词",
            ),
            "dispatch_subagents": (
                dispatch_tool,
                "派发多个子 agent 并行调研，参数=用 | 分隔的子课题",
            ),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
        "duration": result["duration"],
    }
