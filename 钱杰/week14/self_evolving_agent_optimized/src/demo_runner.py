"""
演示主程序（优化版）：在原版基础上接入所有优化方案。

接入的优化：
  - 方案 1：Agent 按需注入 Skill（teaching_mode 开关）
  - 方案 2：SkillManager mtime 缓存（透明生效）
  - 方案 3：评估并发化（asyncio.Semaphore，max_concurrency 开关）
  - 方案 5：Reviewer 提示词压缩（teaching_mode 开关）
  - 方案 6：基线评估结果缓存（use_baseline_cache 开关）

使用方式：
  cd self_evolving_agent_optimized
  python src/demo_runner.py                          # 默认优化模式
  python src/demo_runner.py --teaching               # 教学模式（原版行为）
  python src/demo_runner.py --concurrency 10         # 指定并发数
  python src/demo_runner.py --no-baseline-cache     # 禁用基线缓存
  python src/demo_runner.py --stats                  # 输出优化统计

依赖：
  pip install openai
  set DEEPSEEK_API_KEY=your_key
"""

import os
import sys
import json
import shutil
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_manager import SkillManager
from evaluator import Evaluator
from agent import CustomerServiceAgent
from background_reviewer import BackgroundReviewer
from async_eval import run_eval_concurrent_silent
from baseline_cache import (
    try_load_baseline_cache,
    save_baseline_cache,
    clear_baseline_cache,
)

SKILLS_DIR    = ROOT / "skills"
SKILLS_ORIG   = ROOT / "outputs" / "skills_original"
EVAL_SET      = ROOT / "data" / "eval_set.json"
DEMO_SCRIPT   = ROOT / "data" / "demo_script.json"
POLICIES      = ROOT / "data" / "policies.md"
VERSIONS_DIR  = ROOT / "outputs" / "skill_versions"
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"
EVOL_LOG      = ROOT / "outputs" / "evolution_log.json"
BASELINE_CACHE = ROOT / "outputs" / "baseline_cache.json"


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="优化版自进化 Agent 实验")
    p.add_argument("--teaching", action="store_true",
                   help="教学模式：全量注入 Skill + 全量 policies + 串行评估（原版行为）")
    p.add_argument("--concurrency", type=int, default=5,
                   help="评估并发数（默认 5，教学模式下忽略）")
    p.add_argument("--no-baseline-cache", action="store_true",
                   help="禁用基线评估缓存")
    p.add_argument("--stats", action="store_true",
                   help="实验结束后输出优化统计")
    return p.parse_args()


# ── 备份 / 还原 ──────────────────────────────────────────────────────────────

def ensure_original(sm: SkillManager):
    """首次运行时创建原始备份（并保存快照 v1），之后只读不写。"""
    if not SKILLS_ORIG.exists():
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)
        print(f"✓ 首次运行：原始 Skills 备份至 {SKILLS_ORIG.name}/")
        for skill_name, content in sm.load_all().items():
            sm._save_version(skill_name, content, action="initial", reason="初始版本")
    else:
        print("✓ 检测到原始备份，已跳过覆盖")

def restore_from_original(clear_cache: bool = True):
    """每次实验前从原始备份还原，保证可重复运行。"""
    if not SKILLS_ORIG.exists():
        raise RuntimeError("原始备份不存在，请删除 outputs/ 并重新运行")
    if SKILLS_DIR.exists():
        shutil.rmtree(SKILLS_DIR)
    shutil.copytree(SKILLS_ORIG, SKILLS_DIR)
    if VERSIONS_DIR.exists():
        shutil.rmtree(VERSIONS_DIR)
    snapshots = ROOT / "outputs" / "skill_snapshots"
    if snapshots.exists():
        shutil.rmtree(snapshots)
    if EVAL_RUNS_DIR.exists():
        shutil.rmtree(EVAL_RUNS_DIR)
    if clear_cache:
        clear_baseline_cache(BASELINE_CACHE)
    print("✓ 已还原初始 Skills，清空上次版本历史")


# ── 并发 Probe / Full Eval ───────────────────────────────────────────────────

async def run_eval_async(
    agent: CustomerServiceAgent,
    evaluator: Evaluator,
    question_ids: list[int],
    run_id: str,
    label: str,
    sm: SkillManager,
    max_concurrency: int,
    teaching_mode: bool,
) -> dict:
    """并发跑评估，保存到 eval_runs/{run_id}.json，返回完整结果字典。"""
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    result = await run_eval_concurrent_silent(
        agent, evaluator, question_ids,
        max_concurrency=max_concurrency,
        teaching_mode=teaching_mode,
    )

    result.update({
        "run_id": run_id,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "skill_versions_active": sm.get_active_versions(),
    })

    (EVAL_RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


async def run_full_eval_async(agent, evaluator, run_id, label, sm,
                               max_concurrency, teaching_mode, use_cache=False):
    """全量评估（含基线缓存）。"""
    if use_cache and run_id == "baseline":
        cached = try_load_baseline_cache(BASELINE_CACHE, sm.load_all(), EVAL_SET)
        if cached:
            # 命中缓存，补全 run_id/label 等字段后保存一份到 eval_runs
            cached.update({
                "run_id": run_id,
                "label": label,
                "cached": True,
                "skill_versions_active": sm.get_active_versions(),
            })
            (EVAL_RUNS_DIR / f"{run_id}.json").write_text(
                json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return cached

    all_ids = list(evaluator.questions.keys())
    result = await run_eval_async(
        agent, evaluator, all_ids, run_id, label, sm,
        max_concurrency, teaching_mode,
    )

    if use_cache and run_id == "baseline":
        save_baseline_cache(BASELINE_CACHE, result, sm.load_all(), EVAL_SET)

    return result


# ── Evolution Log ─────────────────────────────────────────────────────────────

class EvolutionLog:
    """收集实验过程中所有结构化数据，实验结束后写入 evolution_log.json。"""

    def __init__(self):
        self.eval_runs: list[dict] = []
        self.nudge_events: list[dict] = []
        self.question_history: dict[str, list] = {}

    def add_eval_run(self, result: dict):
        self.eval_runs.append({
            "run_id": result["run_id"],
            "label": result["label"],
            "timestamp": result["timestamp"],
            "skill_versions_active": result["skill_versions_active"],
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "by_category": {k: v["accuracy"] for k, v in result["by_category"].items()},
        })
        for qid_str, ans_data in result["answers"].items():
            self.question_history.setdefault(qid_str, []).append({
                "run_id": result["run_id"],
                "label": result["label"],
                "skill_versions": result["skill_versions_active"],
                "answer": ans_data["answer"],
                "correct": ans_data["correct"],
                "fail_reason": ans_data.get("fail_reason", ""),
            })

    def add_nudge_event(self, seq, block, actions_taken, accuracy_before, skill_versions_after):
        self.nudge_events.append({
            "after_seq": seq,
            "block": block,
            "timestamp": datetime.now().isoformat(),
            "accuracy_before_this_block": round(accuracy_before, 3),
            "actions_taken": actions_taken,
            "skill_versions_after": skill_versions_after,
        })

    def save(self, sm: SkillManager, evaluator: Evaluator):
        skill_snapshots = {}
        for skill_dir in SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                name = skill_dir.name
                history = sm.get_version_history(name)
                skill_snapshots[name] = [
                    {
                        "version": h["version"],
                        "time": h["time"],
                        "action": h["action"],
                        "reason": h["reason"][:120],
                        "snapshot_file": h.get("snapshot_file", ""),
                    }
                    for h in history
                ]

        question_comparison = {}
        for qid_str, history in self.question_history.items():
            qid = int(qid_str)
            q = evaluator.questions.get(qid)
            if q:
                question_comparison[qid_str] = {
                    "question": q["question"],
                    "category": q["category"],
                    "difficulty": q["difficulty"],
                    "ground_truth": q["ground_truth"],
                    "history": history,
                }

        log = {
            "generated_at": datetime.now().isoformat(),
            "skill_snapshots": skill_snapshots,
            "eval_runs": self.eval_runs,
            "nudge_events": self.nudge_events,
            "question_comparison": question_comparison,
        }
        EVOL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ evolution_log.json 已保存 ({EVOL_LOG})")


# ── 主实验流程 ────────────────────────────────────────────────────────────────

async def run_experiment(args):
    print("=" * 60)
    mode = "教学（原版行为）" if args.teaching else "优化"
    print(f"  云购商城客服 Agent 自进化实验 v3（{mode}模式）")
    print("=" * 60)
    print(f"  并发数: {args.concurrency if not args.teaching else 1}")
    print(f"  基线缓存: {'禁用' if args.no_baseline_cache or args.teaching else '启用'}")
    print(f"  Agent 按需注入: {'否' if args.teaching else '是'}")
    print(f"  Reviewer 提示词压缩: {'否' if args.teaching else '是'}")

    (ROOT / "outputs").mkdir(exist_ok=True)
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))

    ensure_original(sm)
    restore_from_original(clear_cache=not (args.teaching or args.no_baseline_cache))

    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    agent = CustomerServiceAgent(
        sm, nudge_interval=0, teaching_mode=args.teaching,
    )
    reviewer = BackgroundReviewer(
        str(POLICIES), sm, teaching_mode=args.teaching,
    )
    evaluator = Evaluator(str(EVAL_SET))
    elog = EvolutionLog()

    for skill_name, content in sm.load_all().items():
        sm._save_version(skill_name, content, action="initial", reason="初始版本")

    script_data = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))
    demo_questions = script_data["questions"]
    nudge_interval = script_data["nudge_interval"]
    probe_ids = script_data.get("probe_question_ids", list(range(1, 31)))

    # ── 基线评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("基线评估（初始 Skills，无进化）")
    print("─" * 60)
    use_cache = not args.no_baseline_cache and not args.teaching
    baseline = await run_full_eval_async(
        agent, evaluator, "baseline", "基线（初始Skills）", sm,
        max_concurrency=args.concurrency,
        teaching_mode=args.teaching,
        use_cache=use_cache,
    )
    elog.add_eval_run(baseline)
    _print_summary(baseline)
    agent.conversation_history.clear()

    # ── 演示脚本 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"演示脚本运行（{len(demo_questions)} 题，Nudge 间隔={nudge_interval}）")
    print("─" * 60)

    iters = 0
    block_correct = 0
    block_total = 0
    block_failed_turns: list[dict] = []

    for item in demo_questions:
        seq = item["seq"]
        eval_id = item["eval_id"]
        question = item["question"]
        block = item.get("block", "")

        # 演示脚本的逐题应答保留串行（教学上需要按顺序展示失败累积）
        answer = agent.answer(question)
        ok, reason = evaluator.evaluate_answer(answer, eval_id)
        block_correct += int(ok)
        block_total += 1
        iters += 1
        if not ok:
            block_failed_turns.append({"question": question, "answer": answer, "fail_reason": reason})

        status = "✓" if ok else "✗"
        note = f"  [{item['note']}]" if item.get("note") else ""
        print(f"  Q{seq:02d} {status}  {question[:50]:<50}{note}")

        if iters >= nudge_interval:
            block_acc = block_correct / block_total if block_total else 0
            print(f"\n{'━'*60}")
            print(f"  本块 [{block}] 完成: {block_correct}/{block_total} = {block_acc:.1%}")

            if not block_failed_turns:
                print(f"  ✓ 本块全对，跳过 Nudge 和 Probe eval")
                elog.add_nudge_event(seq, block, [], block_acc, sm.get_active_versions())
            else:
                print(f"  🔔 Nudge 触发（{len(block_failed_turns)} 条失败样本注入 Reviewer）")
                actions = reviewer.review(block_failed_turns)
                executed_actions = []
                for act in (actions or []):
                    try:
                        if act["action"] == "create":
                            ok_act = sm.create(act["skill_name"], act["content"], reason=act.get("reason",""))
                        elif act["action"] == "patch":
                            ok_act = sm.patch(act["skill_name"], act["old_text"], act["new_text"], reason=act.get("reason",""))
                        else:
                            ok_act = False
                        if ok_act:
                            executed_actions.append({"action": act["action"], "skill": act["skill_name"], "reason": act.get("reason","")[:80]})
                    except Exception as e:
                        print(f"  [Reviewer] 执行失败: {e}")
                print(f"  ✓ 执行了 {len(executed_actions)} 个 Skill 操作")

                # Probe eval 并发跑
                probe_run_id = f"after_nudge_seq{seq}"
                probe_label = f"Nudge后（seq={seq}, block={block}）"
                probe_result = await run_eval_async(
                    agent, evaluator, probe_ids, probe_run_id, probe_label, sm,
                    max_concurrency=args.concurrency,
                    teaching_mode=args.teaching,
                )
                elog.add_eval_run(probe_result)
                print(f"  Probe eval: {probe_result['correct']}/{probe_result['total']} = {probe_result['accuracy']:.1%}")
                elog.add_nudge_event(seq, block, executed_actions, block_acc, sm.get_active_versions())

            iters = 0
            block_correct = 0
            block_total = 0
            block_failed_turns = []
            agent.conversation_history = agent.conversation_history[-5:]
            print(f"{'━'*60}\n")

    # ── 最终评估 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("最终评估（进化后 Skills）")
    print("─" * 60)
    final = await run_full_eval_async(
        agent, evaluator, "final", "最终（进化后）", sm,
        max_concurrency=args.concurrency,
        teaching_mode=args.teaching,
        use_cache=False,  # 最终评估不缓存（Skills 已变）
    )
    elog.add_eval_run(final)
    _print_summary(final)

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  实验结果汇总")
    print("=" * 60)
    base_acc = baseline["accuracy"]
    final_acc = final["accuracy"]
    print(f"  基线准确率:   {base_acc:.1%}  ({baseline['correct']}/{baseline['total']})")
    print(f"  进化后准确率: {final_acc:.1%}  ({final['correct']}/{final['total']})")
    print(f"  准确率提升:   +{(final_acc - base_acc):.1%}")

    print(f"\n  进化轨迹（probe eval 准确率）:")
    for run in elog.eval_runs:
        if run["run_id"].startswith("after_nudge"):
            print(f"    {run['label']:<35}  {run['correct']:>2}/{run['total']:>2} = {run['accuracy']:.1%}")

    print(f"\n  Skill 版本历史:")
    for name, versions in sorted(sm.get_all_version_summaries().items()):
        print(f"    {name}: {len(versions)} 个版本")
        for v in versions:
            print(f"      v{v.get('time','')[:19]} [{v['action']}] {v['reason'][:60]}")

    elog.save(sm, evaluator)

    # ── 优化统计 ─────────────────────────────────────────────────────────────
    if args.stats:
        print(f"\n{'='*60}")
        print("  优化统计")
        print("=" * 60)
        print("\n[SkillManager mtime 缓存]")
        stats = sm.cache_stats()
        print(f"  命中 {stats['hits']} 次，未命中 {stats['misses']} 次，命中率 {stats['hit_rate']:.1%}")

        print("\n[Agent Skill 路由]")
        rs = agent.routing_summary()
        print(f"  模式: {rs['mode']}")
        print(f"  总调用: {rs['total_calls']}")
        print(f"  全量注入: {rs['full_injection_count']} 次")
        print(f"  按需注入: {rs['partial_injection_count']} 次")
        print(f"  实际注入字符: {rs['injected_chars']}")
        print(f"  若全量基线字符: {rs['baseline_chars_if_full']}")
        print(f"  节省字符: {rs['saved_chars']} ({rs['char_save_rate']:.1%})")

        print("\n[Reviewer policies 压缩]")
        cs = reviewer.compression_summary()
        print(f"  模式: {cs['mode']}")
        print(f"  Reviewer 调用: {cs['review_calls']} 次")
        print(f"  全量平均字符: {cs['avg_full_chars']}")
        print(f"  压缩后平均字符: {cs['avg_compressed_chars']}")
        print(f"  平均节省率: {cs['avg_save_rate']:.1%}")

        print(f"\n  评估并发数: {1 if args.teaching else args.concurrency}")
        print(f"  基线缓存: {'命中' if baseline.get('cached') else '未命中/禁用'}")
        print(f"  评估详情见 outputs/eval_runs/")
        print(f"  各版本 Skill 见 outputs/skill_snapshots/")


def _print_summary(result: dict):
    print(f"总体准确率: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
    print("分类准确率:")
    for cat, stats in sorted(result["by_category"].items()):
        bar = "█" * int(stats["accuracy"] * 20)
        print(f"  {cat:<22} {stats['correct']:>2}/{stats['total']:>2}  {bar} {stats['accuracy']:.0%}")


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    args = parse_args()
    asyncio.run(run_experiment(args))
