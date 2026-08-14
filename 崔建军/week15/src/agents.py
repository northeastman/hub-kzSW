"""主 Agent + 并行 Subagent 编排：爆款历史人物公众号文章生成

教学重点：
  1. 主 agent 自己是 ReAct 循环，有 3 个工具：
     - web_search：单次联网搜索（单一事实快速确认）
     - dispatch_subagents：派发多个 subagent 并行收集素材（生平/轶事/争议等多侧面）
     - recommend_next_figure：根据上一篇/参考人物，推荐下一篇该写谁
     主 agent 根据用户输入自行决定用哪个——不是固定拓扑，是 LLM 自主路由
  2. 并行优势凸显：dispatch_subagents 一次派发 N 个 subagent，
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单agent时长)，而非 sum
  3. 每个 subagent 也是 ReAct 循环（只 web_search 工具），
     trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」
  4. 主 agent 拿到并行素材后，综合成一篇有爆款潜质的公众号文章作为 Final Answer

架构对应 Orchestrator-Workers 拓扑（动态：主 agent 决定派几个、派什么方向）。
"""
import os, time, json, logging, uuid, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result
from llm_client import llm_chat, reset_token_usage, get_token_usage
from article_history import save_article, get_last_figure, get_written_figures

logger = logging.getLogger(__name__)


def _parse_json_loose(text: str) -> dict:
    """宽松解析 LLM 输出里的 JSON（兼容 ```json 包裹、多余文字）。失败返回 {}。"""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

# ── 子 agent 系统提示：历史素材调研员 ───────────────────────────
SUBAGENT_SYSTEM = """你是历史人物素材调研员，能用 web_search 联网搜索收集素材。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（查询词）

工具执行后会得到 Observation。多轮调用直到素材足够，最后用：
Thought: 素材已足够
Final Answer: 整理好的素材，分点输出。每点格式严格为：
  【事实】关键事实/原话/年份/人名（人物原话用引号原样保留，不改写不翻译）
  【来源】如《宋史·辛弃疾传》/人民日报X年/URL
  【可信度】正史 / 野史 / 传说
500字内，宁可少一条，不要把不确定的当事实。

规则：
- Action 必须是上面列出的工具名之一
- 每轮只调一次工具，等 Observation 再决定下一步
- 每条事实必须标注来源；正史与野史/传说分开标注，野史标[野史]
- 人物原话用引号原样保留，不改写不翻译
- 素材要具体（人名、年份、地点、原话），不要泛泛而谈
- 优先查有画面感、有情绪张力、有争议的细节
- 通常搜 1-2 次就够，不要超过 3 次搜索；拿到关键事实立即 Final Answer，避免拖到步数上限"""

# ── 主 agent 合成阶段系统提示（收到并行素材后，综合成文） ─────
# 注意：本提示直接喂给 llm_chat，不走 .format()，故无需转义花括号。
SYNTH_SYSTEM = """你是爆款历史人物公众号文章主笔。你会收到子调研员并行收集的【素材】（已标注来源与可信度），请综合成一篇有爆款潜质、又经得起事实核查的文章。

【最高原则：不虚构】
- 只能使用素材中明确出现的事实、原话、年份、人名、因果。
- 禁止虚构人物对话、心理活动、具体场景细节（除非素材原文有原话/场景）。
- 素材不足的部分宁可略写，绝不编细节、编原话、编数字凑字。
- 野史/传说类素材，使用时必须标注"据传""野史""后世演绎"。
- 爆款靠结构与选题，不靠编造。

【标题】
第一行必须输出恰好 3 个备选标题，每个用「」括起、之间用空格分隔，禁止用 # 或《》。
示例：「他22岁率50骑活捉叛将，却被南宋冷落40年」 「临终大喊杀贼，南宋却防了他一辈子」 「辛弃疾：一个被自己人困死的战神」
（先在脑内构思 8 个，按 悬念/反差/具体/不夸大/有画面 五维度打分，取最高 3 个；只输出这 3 个，不输出打分过程）
3 个「」标题之后直接是正文第一段，不要再写 # 标题或《》标题行。

【叙事弧线结构】
情绪节奏：好奇 → 共情 → 震撼 → 唏嘘，逐层递进。
1. 反差开篇（1段）：用素材中有的画面感/反差场景或事实切入，禁止"某某，字XX，生于XX年"式平铺；素材无场景则用反差事实切入。
2. 核心冲突与抉择（2-3段）：故事化展开，融入素材轶事与原话。
3. 关键转折/高潮（1-2段）：命运转折点。
4. 争议两面（1段）：呈现正反评价（素材中的），引发讨论。
5. 余韵+互动（1段）：回味收尾 + 一个互动问题。
每节结尾留一句金句或反问；用 2-3 个小标题分段，每段不超 4 行。

【其他】
- 全文 2000-2200 字（公众号常规字数），口语化，有情绪起伏，像真人写的而非百科。
- 尽量保留素材中的具体年份、原话、出处。
直接输出文章全文（标题行+正文），不要加任何解释或前后缀。"""


# ── 事实核查员：逐条比对草稿与素材，列出存疑项（不改正） ───────
FACT_CHECK_SYSTEM = """你是严谨的历史事实核查员。给你【素材】和【文章草稿】，逐条检查草稿中的事实性陈述（年份/事件/人物关系/原话/数字/因果/场景细节），判定每条：
- ✅有据：素材中能找到对应
- ⚠️存疑：素材中没有，疑似虚构或夸大

只列出⚠️存疑项（若全部有据则回答"全部有据"），格式：
1. [草稿原句摘录] → 存疑原因（素材中无 / 与素材矛盾 / 疑似编造对话或心理）
不要改正文章，只列清单。"""


# ── 编辑：按核查结果修订草稿，删虚构/补真实，保持爆款结构 ─────
REVISE_SYSTEM = """你是公众号文章编辑。根据【核查结果】修订【草稿】：
1. 把⚠️存疑的【具体对话/心理活动/场景细节/数字】删除，或改为不确定语气（"据传""或云""后世演绎"）。
2. 若删除导致断裂或字数不足，用【素材】中的其他真实细节替换/补充，不得新增素材外的虚构内容。
3. 保留文章的爆款叙事结构与情绪，但事实必须落到素材有据。
4. 保持字数 2000-2200；标题行（3 个「」标题）保持不变，仍为第一行。
直接输出修订后的完整文章（标题行+正文），不要解释。"""


# ── 爆款角度提炼 + 素材预检（合并为一次调用，省 1 次 LLM + 1 次素材输入 token） ─
ANGLE_CHECK_SYSTEM = """你是爆款文章选题策划兼素材评估员。给【人物】和【素材】，完成两件事：
1. 提炼一句话"核心张力/悬念句"（基于素材中的真实事实/转折/争议；让人忍不住想读下去的问题或反差；不超过30字；不要引号）
2. 判断素材是否包含：①至少1个有画面感的具体场景(有时间/地点/动作) ②至少2句人物原话(带引号引用)

严格按以下两行输出，不要其他内容：
ANGLE: 核心张力句
CHECK: {"scene": true/false, "quotes": true/false, "missing": "缺什么的一句话"}"""


# ── 吸引力自评：从标题/开头/高潮/金句评爆款潜质 ─────────────
APPEAL_CHECK_SYSTEM = """你是爆款文章吸引力评估员。给【文章】，按4维度各打1-10分并指出最弱处：
1. title 标题抓人度
2. opening 开头钩子(前3句是否让人想读下去)
3. climax 情绪高潮(有没有让人震撼/共情的点)
4. quotes 金句记忆点(有没有值得截图转发的句子)
只输出一行 JSON：{"title":分,"opening":分,"climax":分,"quotes":分,"weakest":"最弱维度名","advice":"一句改进建议"}
不要输出其他内容。"""


# ── 润色师：按吸引力评估补强弱项，不增虚构 ───────────────────
POLISH_SYSTEM = """你是爆款文章润色师。根据【吸引力评估】润色【文章】的弱项：
- 标题不够狠 → 重写 3 个更抓人的(仍用「」,不夸大造假)
- 开头不够钩 → 用素材中已有的画面感场景重写前3句
- 缺高潮 → 强化素材中转折点的情绪
- 缺金句 → 在节尾加一句有记忆点的话(基于素材,不编造)
硬约束：不得新增素材外的事实/原话；保持字数 2000-2200；3 个「」标题保留为第一行。
直接输出润色后的完整文章（标题行+正文），不要解释。"""


# ── recommend_next_figure 工具实现 ──────────────────────────────
def _parse_figure_from_recommend(text: str) -> str:
    """从推荐结果文本里提取人物名（第一行 '人物名：XXX'）。"""
    for line in text.splitlines():
        m = re.search(r"人物名\s*[:：]\s*(.+)", line)
        if m:
            # 去掉后续说明，只留名字
            return m.group(1).strip().split("，")[0].split(" ")[0].strip()
    return ""


def recommend_next_figure(action_input: str, shared_state: dict = None) -> str:
    """recommend_next_figure 工具：根据参考人物 + 已写历史，推荐下一篇该写谁。
    会做一次 web_search 找相关/对比人物线索，再让 LLM 在反差/关联/爆款潜质上决策。
    把推荐出的人物名写入 shared_state['recommended_figure']，供落盘用。"""
    ref = (action_input or "").strip() or get_last_figure() or ""
    written = get_written_figures()

    if ref:
        sr = tavily_search(f"{ref} 同时代 相关 历史人物 对比 反差", max_results=5)
        search_text = format_search_result(sr)
    else:
        search_text = "（无参考人物，请从零推荐一个有爆款潜质的历史人物）"

    prompt = f"""你是爆款历史人物公众号选题策划。已写过的人物：{written or '（无）'}
参考人物：{ref or '（无）'}
相关人物线索：{search_text[:800]}

请推荐【1个】下一篇该写的历史人物。要求：
1. 与已写人物有反差/关联/对比（同时代对比、命运反差、因果关联），增强系列感
2. 该人物本身有足够轶事、争议、戏剧性，具备爆款潜质
3. 避免与已写人物重复

严格按以下格式输出（第一行必须含人物名）：
人物名：XXX
选题角度：一句话点出爆款切入角度
爆款潜质：2-3点原因（情绪钩子/争议点/反差感）"""
    rec = llm_chat("你是历史选题策划", prompt, temperature=0.8, max_tokens=400)
    fig = _parse_figure_from_recommend(rec)
    if shared_state is not None and fig:
        shared_state["recommended_figure"] = fig
    return rec


# ── dispatch_subagents 工具实现（并行核心） ─────────────────────
def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "调研方向1 | 调研方向2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致）。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出调研方向"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=5, model_tag="deepseek-chat(子)",
            system_prompt=SUBAGENT_SYSTEM)
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
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

    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for) ──
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
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=min(len(defs), 4)) as pool:
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

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【调研方向: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:1000]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


# ── 辅助：从用户输入提取人物名 / 从文章提取标题 ─────────────────
def _extract_figure_from_request(request: str) -> str:
    """从用户输入提取历史人物名。正则优先 → 启发式 → 轻量 LLM 抽取。"""
    req = (request or "").strip()
    # 正则优先：匹配"关于XX的""写一篇关于XX"等常见句式，省一次 LLM 调用
    m = re.search(r"关于\s*([\u4e00-\u9fa5·]{2,6}?)\s*的", req)
    if m:
        return m.group(1).strip()
    # 启发式：输入很短且不含常见动词，直接当人名
    if 2 <= len(req) <= 6 and not any(w in req for w in
                                      ["写", "生成", "推荐", "文章", "关于", "篇",
                                       "历史", "人物", "下一篇", "上一篇", "根据"]):
        return req
    try:
        name = llm_chat("你只输出人物名本身，不输出任何其他字。",
                        f"从下面这句话里提取【一个】历史人物名，只输出人物名本身；"
                        f"若没有则输出“无”：\n{request}",
                        temperature=0.0, max_tokens=20).strip()
        if name and len(name) <= 10 and name not in ("无", "没有", "空"):
            return name
    except Exception as e:
        logger.warning(f"提取人物名失败: {e}")
    return ""


def _extract_title(article: str) -> str:
    """从文章正文提取一个标题（第一个「...」或首行）。"""
    m = re.search(r"「(.+?)」", article or "")
    if m:
        return m.group(1)
    lines = [l for l in (article or "").splitlines() if l.strip()]
    return lines[0].strip().lstrip("#")[:30] if lines else "历史人物"


# ── 顶层入口：主 agent 编排，生成一篇文章 ─────────────────────
def run_article(request: str, on_main_step: Callable = None,
                on_subagent_step: Callable = None,
                on_subagent_done: Callable = None,
                on_dispatch: Callable = None,
                serial: bool = False, mode: str = "auto") -> dict:
    """主 agent 编排执行一次文章生成。返回 {final_answer, figure, title,
    main_trace, subagents, parallel_stats, dispatches}。

    mode:
      - "direct"   直接为 request 中的人物生成
      - "recommend" 推荐一个新人物并生成（request 里若有人名则作"参考对比"，不作主题）
      - "auto"      按 request 文本含"推荐/下一篇"自动判别

    设计说明（为何用结构性编排而非纯 LLM ReAct 路由）：
      实测 DeepSeek-chat 在"写历史人物文章"任务上倾向于第一轮直接 Final Answer
      凭记忆写文，跳过 dispatch_subagents，导致并发子 agent 形同虚设。
      为保证「主 agent 下发并发子 agent → 汇总」这一核心能力必然发生，
      这里由主 agent（本编排函数）显式执行三阶段：
        ① 确定人物：模式B 调 recommend_next_figure；模式A 从输入抽取
        ② 并行派发：调 _dispatch_subagents，ThreadPoolExecutor 并行跑 N 个子 agent
        ③ 汇总合成：把并行素材喂给 LLM（SYNTH_SYSTEM）综合成爆款文章
      每个阶段都通过 on_main_step 回调把 trace 推给前端，子 agent 并行步骤通过
      on_subagent_step/on_subagent_done 推出，前端"多列同时滚动"即并行的直观证据。
    """
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}
    main_trace: list[dict] = []
    _idx = [0]
    reset_token_usage()  # 每次生成重置 token 计数

    def emit(thought, action, action_input, observation=None, final=False):
        """记录并推送一个主 agent 步骤（同一 idx 可再次调用以补 observation）。"""
        step = {"idx": _idx[0], "agent": "main", "thought": thought,
                "action": action, "action_input": action_input,
                "observation": observation, "final": final, "done": observation is not None or final}
        main_trace.append(step)
        _idx[0] += 1
        if on_main_step:
            on_main_step(step)
        return step

    # ── ① 确定人物 ──
    if mode == "auto":
        wants_recommend = any(w in request for w in ["推荐", "下一篇"])
    else:
        wants_recommend = (mode == "recommend")
    if wants_recommend:
        # 输入里若有人名，作为"参考对比人物"（推荐一个与之反差/关联的新人物）；
        # 输入为空则用上一篇已写人物作参考。绝不把参考人物当主题再写一遍。
        ref = _extract_figure_from_request(request)
        emit(f"用户要推荐下一篇，根据参考人物【{ref or '（用上一篇）'}】选一个有爆款潜质的新人物",
             "recommend_next_figure", ref or "（留空=用上一篇）")
        rec = recommend_next_figure(ref, shared_state=shared_state)
        figure = (shared_state.get("recommended_figure")
                  or _parse_figure_from_recommend(rec)
                  or "历史人物")
        if not figure or figure == ref:
            # 兜底：推荐结果退化成参考人物本身时，避免重复，退回历史人物泛指
            figure = "历史人物"
        main_trace[-1]["observation"] = rec
        if on_main_step:
            on_main_step(main_trace[-1])
    else:
        figure = _extract_figure_from_request(request) or request.strip() or "历史人物"

    # ── ② 并行派发子 agent 收集素材（并发核心） ──
    subtopics = (f"{figure} 生平主线与关键事件时间线"
                 f" | {figure} 性格轶事与画面感细节故事"
                 f" | {figure} 历史争议与不同评价")
    emit(f"为【{figure}】并行派发 3 个子调研员收集素材（生平/轶事/争议）",
         "dispatch_subagents", subtopics)
    materials = _dispatch_subagents(
        subtopics, shared_state=shared_state,
        on_subagent_step=on_subagent_step, on_subagent_done=on_subagent_done,
        on_dispatch=on_dispatch, serial=serial)
    main_trace[-1]["observation"] = materials[:1000]
    if on_main_step:
        on_main_step(main_trace[-1])

    # ── 兜底：若所有子 agent 都没产出有效素材，降级为"模型自身知识+声明"，不直接失败 ──
    valid_subs = [s for s in shared_state["subagents"].values()
                  if len((s.get("final_answer") or "").strip()) >= 20]
    online_ok = bool(valid_subs)
    if not online_ok:
        logger.warning("所有子调研员均无有效素材，降级为模型自身知识成文（未经联网核实）")
        emit("联网素材全失败，降级为模型知识成文（将标注未核实）", "degrade",
             "（降级：基于模型知识，未经联网核实）")
        materials = (f"（联网搜索全部失败。以下文章将基于模型自身知识生成，"
                     f"未经联网核实，请读者注意甄别。）")
        main_trace[-1]["done"] = True
        if on_main_step:
            on_main_step(main_trace[-1])

    # ── ②.5 爆款角度提炼 + 素材预检与定向补搜（合成前质量闸，仅联网成功时） ──
    # 角度+预检合并为一次 LLM 调用，省 1 次 API 调用 + 1 次 materials 输入 token
    angle = ""
    if online_ok:
        emit("提炼爆款角度 + 素材预检(合并)", "angle_check", "（分析中…）")
        ac_raw = ""
        try:
            ac_raw = llm_chat(ANGLE_CHECK_SYSTEM,
                              f"【人物】{figure}\n【素材】\n{materials[:2500]}",
                              temperature=0.3, max_tokens=200)
            # 解析 ANGLE: 行
            for line in ac_raw.splitlines():
                if line.startswith("ANGLE:"):
                    angle = line[6:].strip()
                    break
            if len(angle) > 60:
                angle = angle.split("。")[0][:60]
            # 解析 CHECK: 行的 JSON
            mc = _parse_json_loose(ac_raw)
            main_trace[-1]["action_input"] = (
                f"（角度：{angle or '无'} | 预检：{mc.get('missing', '无缺失')}）")
            main_trace[-1]["observation"] = ac_raw[:300]
        except Exception as e:
            logger.warning(f"角度+预检失败: {e}")
            angle = ""
            mc = {}
            main_trace[-1]["action_input"] = f"（角度+预检失败：{str(e)[:30]}）"
        main_trace[-1]["done"] = True
        if on_main_step:
            on_main_step(main_trace[-1])

        # 定向补搜：缺场景或缺原话时补一次针对性搜索
        if mc and (not mc.get("scene") or not mc.get("quotes")):
            lack = []
            if not mc.get("scene"):
                lack.append("具体场景")
            if not mc.get("quotes"):
                lack.append("人物原话")
            lack_text = "、".join(lack)
            sup_query = f"{figure} {lack_text} 典故 原话 史书记载"
            emit(f"素材缺{lack_text}，定向补搜", "supplement_search", sup_query)
            try:
                sup = format_search_result(tavily_search(sup_query, max_results=5))
                if sup and "搜索失败" not in sup and sup != "无结果":
                    materials = materials + "\n\n【定向补搜素材】\n" + sup
                    main_trace[-1]["observation"] = sup[:300]
                    main_trace[-1]["action_input"] = "（补搜成功，素材已扩充）"
                else:
                    main_trace[-1]["action_input"] = "（补搜无结果，沿用原素材）"
            except Exception as e:
                logger.warning(f"补搜失败: {e}")
                main_trace[-1]["action_input"] = f"（补搜失败：{str(e)[:30]}）"
            main_trace[-1]["done"] = True
            if on_main_step:
                on_main_step(main_trace[-1])

    # ── ③ 合成草稿（爆款结构，受素材约束；注入角度+系列承接） ──
    last_fig = get_last_figure() if wants_recommend else None
    series_hint = ""
    if last_fig and last_fig != figure:
        series_hint = (f"\n【系列承接】这是系列文章的下一篇，上一篇写的是【{last_fig}】。"
                       f"可在开头用一句话巧妙呼应上一篇（对比/反差/因果），但不要喧宾夺主，"
                       f"仍以【{figure}】为主角。\n")
    angle_hint = (f"\n【核心张力句】请在全文贯穿这条悬念线"
                  f"（开头点出、高潮呼应、结尾回味）：{angle}\n" if angle else "")
    synth_user = (f"用户想写一篇关于【{figure}】的爆款公众号文章。\n"
                  f"{angle_hint}{series_hint}"
                  f"\n以下是 {len(shared_state['subagents'])} 个子调研员并行收集的素材：\n{materials}\n\n"
                  f"请综合以上素材，写出一篇有爆款潜质又经得起核查的文章。")
    emit("综合并行素材合成草稿（爆款叙事弧线）", "synthesize", "（生成中…）")
    try:
        article = llm_chat(SYNTH_SYSTEM, synth_user, temperature=0.8, max_tokens=2400)
    except Exception as e:
        logger.error(f"合成草稿失败: {e}")
        emit("合成失败", "Final Answer", f"（合成失败：{str(e)[:60]}）", final=True)
        return {"final_answer": f"（合成失败：{e}）", "figure": figure, "title": "生成失败",
                "main_trace": main_trace, "subagents": shared_state["subagents"],
                "parallel_stats": shared_state["parallel_stats"],
                "dispatches": shared_state["dispatches"], "token_usage": get_token_usage()}
    main_trace[-1]["action_input"] = f"（草稿 {len(article)} 字）"
    main_trace[-1]["done"] = True
    if on_main_step:
        on_main_step(main_trace[-1])

    # ── ④ 事实核查 + 修订（防虚构最后一道闸；任一步失败均降级沿用草稿） ──
    check = ""
    if online_ok:
        emit("事实核查:逐条比对草稿与素材,列出存疑项", "fact_check", "（核查中…）")
        try:
            check = llm_chat(FACT_CHECK_SYSTEM,
                             f"【素材】\n{materials}\n\n【文章草稿】\n{article}\n\n"
                             f"请逐条核查并列出存疑项。",
                             temperature=0.0, max_tokens=900)
            main_trace[-1]["action_input"] = "（核查完成）"
            main_trace[-1]["observation"] = check[:700]
        except Exception as e:
            logger.warning(f"事实核查失败，跳过核查沿用草稿: {e}")
            check = ""
            main_trace[-1]["action_input"] = f"（核查失败：{str(e)[:40]}，跳过）"
        main_trace[-1]["done"] = True
        if on_main_step:
            on_main_step(main_trace[-1])

        emit("按核查结果修订:删虚构/改不确定语气/补真实细节", "revise", "（修订中…）", final=True)
        if check:
            try:
                revised = llm_chat(REVISE_SYSTEM,
                                   f"【素材】\n{materials}\n\n【核查结果】\n{check}\n\n"
                                   f"【草稿】\n{article}\n\n请修订出终稿。",
                                   temperature=0.6, max_tokens=2400)
                if revised and revised.strip():
                    article = revised
                main_trace[-1]["action_input"] = f"（终稿 {len(article)} 字，见下方）"
            except Exception as e:
                logger.warning(f"修订失败，沿用草稿: {e}")
                main_trace[-1]["action_input"] = f"（修订失败，沿用草稿 {len(article)} 字）"
        else:
            main_trace[-1]["action_input"] = f"（无核查结果，沿用草稿 {len(article)} 字）"
        if on_main_step:
            on_main_step(main_trace[-1])

        # ── ⑤ 吸引力自评 + 弱项润色（质量最后一道抛光） ──
        emit("吸引力自评:标题/开头/高潮/金句四维度打分", "appeal_check", "（评分中…）")
        ap_raw = ""
        try:
            ap_raw = llm_chat(APPEAL_CHECK_SYSTEM,
                              f"【文章】\n{article}",
                              temperature=0.0, max_tokens=150)
            ap = _parse_json_loose(ap_raw)
            main_trace[-1]["action_input"] = (
                f"（评分：标题{ap.get('title','?')}/开头{ap.get('opening','?')}"
                f"/高潮{ap.get('climax','?')}/金句{ap.get('quotes','?')}）")
            main_trace[-1]["observation"] = ap_raw[:200]
        except Exception as e:
            logger.warning(f"吸引力自评失败: {e}")
            ap = {}
            main_trace[-1]["action_input"] = f"（自评失败：{str(e)[:30]}）"
        main_trace[-1]["done"] = True
        if on_main_step:
            on_main_step(main_trace[-1])

        # 弱项润色：任一维度 < 7 分则润色
        score_vals = [ap.get(k) for k in ("title", "opening", "climax", "quotes")]
        score_vals = [s for s in score_vals if isinstance(s, (int, float))]
        need_polish = bool(score_vals) and any(s < 7 for s in score_vals)
        if need_polish:
            emit(f"弱项润色(最弱:{ap.get('weakest','?')})", "polish", "（润色中…）")
            try:
                polished = llm_chat(POLISH_SYSTEM,
                                    f"【吸引力评估】\n{ap_raw}\n\n【文章】\n{article}",
                                    temperature=0.7, max_tokens=2400)
                if polished and polished.strip():
                    article = polished
                main_trace[-1]["action_input"] = f"（润色后 {len(article)} 字）"
            except Exception as e:
                logger.warning(f"润色失败: {e}")
                main_trace[-1]["action_input"] = f"（润色失败，沿用修订稿 {len(article)} 字）"
            main_trace[-1]["done"] = True
            if on_main_step:
                on_main_step(main_trace[-1])
    else:
        emit("未联网核实，跳过事实核查，直接成文", "Final Answer",
             f"（终稿 {len(article)} 字，未联网核实）", final=True)
        if on_main_step:
            on_main_step(main_trace[-1])

    # ── 落盘 ──
    title = _extract_title(article)
    try:
        save_article(figure, article, title)
    except Exception as e:
        logger.warning(f"保存文章失败: {e}")

    return {
        "final_answer": article,
        "figure": figure,
        "title": title,
        "main_trace": main_trace,
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
        "fact_check": check,
        "token_usage": get_token_usage(),
    }


# ── 自测 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    req = "写一篇关于苏轼的爆款公众号文章"
    r = run_article(req)
    print(f"\n{'=' * 60}\n人物: {r['figure']} | 标题: {r['title']}")
    print(f"主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n文章头:\n{r['final_answer'][:300]}")
