"""
主 Agent + 并行 Subagent 编排（旅行规划场景）

教学重点：
  1. 主 agent 是 ReAct 循环，有 2 个工具：
     - web_search：单次联网搜索（单一事实直接用）
     - dispatch_subagents：派发多个 subagent 并行收集（多维度行程规划用）
     主 agent 根据 query 自行决定用哪个——LLM 自主路由，非固定拓扑
  2. 并行优势：dispatch 一次派发 N 个 subagent，ThreadPoolExecutor 并行跑，
     wall-clock ≈ max(单agent时长)，而非 sum
  3. 每个 subagent 也是 ReAct（只 web_search），trace 存入 shared_state 供可视化

架构：动态 Orchestrator-Workers —— 主 agent 决定派几个、派什么。
场景：旅行规划（交通 / 住宿 / 景点 / 美食 / 天气注意），与市场调研同构换域。
"""

import time, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是旅行规划主协调员。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行收集（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「行程规划」「旅行攻略」「XX 游」「周末去哪」等，
  通常含交通/住宿/景点/美食/天气），必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，
  不要自己串行 web_search 多次。
  示例："成都三日游规划：交通、住宿、景点、美食" → Action: dispatch_subagents
        Action Input: 成都三日游市内交通与到达方式 | 成都三日游住宿区域与价位建议 | 成都三日必去景点与路线 | 成都特色美食与推荐餐厅
- 只有单一事实问题（如"成都大熊猫基地门票多少钱"）才直接 web_search
- 拿到子调研结果后，综合成可执行的行程方案

行程方案要求：按天/维度组织，每条建议带来源要点，末尾给出预算粗估与注意事项。

【示例】
Question: 杭州两日游攻略：交通、景点、美食
Thought: 这是多维度行程规划（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 杭州两日游交通与到达方式 | 杭州两日必去景点与游玩顺序 | 杭州特色美食与餐厅推荐
Observation: 并行收集完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行结果，综合成行程方案
Final Answer: （分天/分维度行程方案）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    serial=True 时串行（eval A/B 基线）；默认 ThreadPool 并行。
    用真实 subagent id 发 dispatch 事件，与后续 step 事件对齐。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=4, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid=sid, sub=sub, topic=topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行收集完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_plan(question: str, on_main_step: Callable = None,
             on_subagent_step: Callable = None,
             on_subagent_done: Callable = None,
             on_dispatch: Callable = None,
             serial: bool = False) -> dict:
    """执行一次旅行规划。返回 {final_answer, main_trace, subagents, parallel_stats, dispatches}。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子调研员并行收集，参数=用 | 分隔的多个子课题"),
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
    }


# 兼容旧调用名（与市场调研项目对称）
run_research = run_plan


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "成都三日游规划：交通到达、住宿区域、必去景点、特色美食"
    r = run_plan(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n方案头:\n{r['final_answer'][:200]}")
