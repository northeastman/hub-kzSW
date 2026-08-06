"""
Skill 优化对比测试脚本
功能：对比初始 Skill 和优化后 Skill 在 token 消耗、准确率、响应时间上的差异
用法：python src/skill_optimization_test.py
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_manager import SkillManager
from evaluator import Evaluator
from agent import CustomerServiceAgent

SKILLS_DIR = ROOT / "skills"
EVAL_SET = ROOT / "data" / "eval_set.json"
OUTPUT_DIR = ROOT / "outputs" / "optimization_test"
BACKUP_DIR = ROOT / "outputs" / "logistics_v1_backup"

# 物流类别的题目ID（从评估集中提取）
LOGISTICS_QUESTION_IDS = [46, 47, 48, 49, 50, 51, 52, 53]


def backup_initial_skill():
    """备份初始 logistics skill"""
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    logistics_dir = SKILLS_DIR / "logistics"
    if logistics_dir.exists():
        shutil.copytree(logistics_dir, BACKUP_DIR / "logistics")
        print(f"✓ 初始 Skill 已备份至 {BACKUP_DIR}")


def restore_initial_skill():
    """还原初始 logistics skill"""
    logistics_dir = SKILLS_DIR / "logistics"
    if logistics_dir.exists():
        shutil.rmtree(logistics_dir)
    shutil.copytree(BACKUP_DIR / "logistics", logistics_dir)
    print("✓ 已还原初始 Skill")


def measure_skill(skill_version: str, skills_to_use: list[str] = None) -> dict:
    """
    测量指定 skill 版本的性能指标
    
    Args:
        skill_version: 版本名称（用于日志）
        skills_to_use: 要加载的 skill 名称列表，None 则加载全部
    
    Returns:
        包含各项指标的字典
    """
    sm = SkillManager(str(SKILLS_DIR))
    evaluator = Evaluator(str(EVAL_SET))
    agent = CustomerServiceAgent(sm, nudge_interval=0)
    
    results = {
        "version": skill_version,
        "timestamp": datetime.now().isoformat(),
        "questions": [],
        "summary": {},
        "token_stats": {},
    }
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0
    correct_count = 0
    total_questions = len(LOGISTICS_QUESTION_IDS)
    
    print(f"\n{'─'*60}")
    print(f"测试 Skill 版本: {skill_version}")
    print(f"{'─'*60}")
    
    for qid in LOGISTICS_QUESTION_IDS:
        q = evaluator.questions[qid]
        question = q["question"]
        
        # 构建 system prompt 并计算 token
        system_prompt = agent._build_system_prompt()
        # 简单估算：1 个中文 ≈ 1 token，1 个英文单词 ≈ 1.3 token
        input_tokens_estimate = len(system_prompt) + len(question)
        
        # 测量响应时间和 token
        start_time = time.time()
        
        # 直接调用 API 获取详细的 token 信息
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        
        response = client.chat.completions.create(
            model=agent.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=400,
        )
        
        elapsed = time.time() - start_time
        
        answer = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        ok, reason = evaluator.evaluate_answer(answer, qid)
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_time += elapsed
        if ok:
            correct_count += 1
        
        question_result = {
            "id": qid,
            "question": question,
            "answer": answer,
            "correct": ok,
            "fail_reason": reason if not ok else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_time_ms": round(elapsed * 1000, 1),
        }
        results["questions"].append(question_result)
        
        status = "✓" if ok else "✗"
        print(f"  Q{qid:02d} {status} | 输入:{input_tokens}t 输出:{output_tokens}t 总:{total_tokens}t | 耗时:{elapsed*1000:.0f}ms | {reason if not ok else ''}")
    
    accuracy = round(correct_count / total_questions, 3) if total_questions > 0 else 0
    
    results["summary"] = {
        "total_questions": total_questions,
        "correct": correct_count,
        "accuracy": accuracy,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_input_tokens_per_q": round(total_input_tokens / total_questions, 1),
        "avg_output_tokens_per_q": round(total_output_tokens / total_questions, 1),
        "avg_time_ms": round(total_time * 1000 / total_questions, 1),
        "total_time_ms": round(total_time * 1000, 1),
    }
    
    # 计算 Skill 本身的 token 数
    all_skills = sm.load_all()
    skill_token_counts = {}
    for name, content in all_skills.items():
        # 粗略估算：1 个中文字符 ≈ 1 token
        skill_token_counts[name] = len(content)
    results["token_stats"] = {
        "skill_token_counts": skill_token_counts,
        "total_skill_tokens": sum(skill_token_counts.values()),
    }
    
    return results


def print_comparison(result_v1: dict, result_v2: dict):
    """打印优化前后的对比报告"""
    s1 = result_v1["summary"]
    s2 = result_v2["summary"]
    
    print(f"\n{'='*70}")
    print(f"  Skill 优化对比报告")
    print(f"{'='*70}")
    
    print(f"\n{'指标':<25} {'优化前 (v1)':<20} {'优化后 (v2)':<20} {'变化':<15}")
    print(f"{'─'*80}")
    
    def fmt_change(v1, v2, is_percent=False, is_lower_better=True):
        if v1 == 0:
            return "N/A"
        change = (v2 - v1) / v1 * 100
        arrow = "↓" if (is_lower_better and change < 0) or (not is_lower_better and change > 0) else "↑"
        sign = "+" if change > 0 else ""
        if is_percent:
            return f"{sign}{change:.1f}% {arrow}"
        return f"{sign}{change:.1f}% {arrow}"
    
    rows = [
        ("准确率", f"{s1['accuracy']:.1%}", f"{s2['accuracy']:.1%}", 
         fmt_change(s1['accuracy'], s2['accuracy'], is_lower_better=False)),
        ("正确/总数", f"{s1['correct']}/{s1['total_questions']}", f"{s2['correct']}/{s2['total_questions']}", ""),
        ("平均输入token/题", f"{s1['avg_input_tokens_per_q']:.0f}", f"{s2['avg_input_tokens_per_q']:.0f}", 
         fmt_change(s1['avg_input_tokens_per_q'], s2['avg_input_tokens_per_q'])),
        ("平均输出token/题", f"{s1['avg_output_tokens_per_q']:.0f}", f"{s2['avg_output_tokens_per_q']:.0f}", 
         fmt_change(s1['avg_output_tokens_per_q'], s2['avg_output_tokens_per_q'])),
        ("总token消耗", f"{s1['total_tokens']}", f"{s2['total_tokens']}", 
         fmt_change(s1['total_tokens'], s2['total_tokens'])),
        ("平均响应时间(ms)", f"{s1['avg_time_ms']:.0f}", f"{s2['avg_time_ms']:.0f}", 
         fmt_change(s1['avg_time_ms'], s2['avg_time_ms'])),
    ]
    
    for row in rows:
        print(f"  {row[0]:<23} {row[1]:<20} {row[2]:<20} {row[3]}")
    
    # Skill token 对比
    st1 = result_v1["token_stats"]
    st2 = result_v2["token_stats"]
    print(f"\n{'─'*80}")
    print(f"Skill 文件 token 对比:")
    print(f"  所有 Skill 总 token: {st1['total_skill_tokens']} → {st2['total_skill_tokens']} "
          f"({fmt_change(st1['total_skill_tokens'], st2['total_skill_tokens'])})")
    
    for skill_name in st1["skill_token_counts"]:
        t1 = st1["skill_token_counts"][skill_name]
        t2 = st2["skill_token_counts"].get(skill_name, 0)
        print(f"    {skill_name}: {t1} → {t2} ({fmt_change(t1, t2)})")
    
    # 逐题对比
    print(f"\n{'─'*80}")
    print(f"逐题对比:")
    print(f"  {'ID':<5} {'题目':<35} {'v1':<8} {'v2':<8} {'输入t v1':<10} {'输入t v2':<10} {'变化':<15}")
    print(f"  {'─'*90}")
    
    for i, q1 in enumerate(result_v1["questions"]):
        q2 = result_v2["questions"][i]
        status_v1 = "✓" if q1["correct"] else "✗"
        status_v2 = "✓" if q2["correct"] else "✗"
        change = fmt_change(q1["input_tokens"], q2["input_tokens"])
        print(f"  Q{q1['id']:02d}  {q1['question'][:32]:<35} {status_v1:<8} {status_v2:<8} "
              f"{q1['input_tokens']:<10} {q2['input_tokens']:<10} {change}")
    
    # 优化亮点总结
    print(f"\n{'─'*80}")
    print(f"优化效果总结:")
    
    token_saved = s1['total_tokens'] - s2['total_tokens']
    token_saved_pct = (token_saved / s1['total_tokens'] * 100) if s1['total_tokens'] > 0 else 0
    accuracy_change = (s2['accuracy'] - s1['accuracy']) * 100
    
    print(f"  • Token节省: {token_saved} tokens ({token_saved_pct:.1f}%)")
    print(f"  • 准确率变化: {accuracy_change:+.1f}%")
    print(f"  • Skill文件精简: {st1['total_skill_tokens']} → {st2['total_skill_tokens']} tokens")
    
    # 效率指标：每题平均节省的 token 和时间
    avg_token_saved_per_q = s1['avg_input_tokens_per_q'] - s2['avg_input_tokens_per_q']
    print(f"  • 每题平均节省输入 token: {avg_token_saved_per_q:.1f}")
    print(f"  • 每题平均响应时间变化: {s1['avg_time_ms']:.0f}ms → {s2['avg_time_ms']:.0f}ms")
    
    if accuracy_change >= 0 and token_saved_pct > 0:
        print(f"\n  ✓ 优化成功: 在保持/提升准确率的同时，显著减少了 token 消耗")
    elif accuracy_change < 0 and token_saved_pct > 10:
        print(f"\n  ⚠ 警告: 虽然大幅节省了 token，但准确率有所下降，需权衡")
    elif accuracy_change > 0 and token_saved_pct <= 0:
        print(f"\n  ⚠ 提示: 准确率提升了，但 token 消耗反而增加了")


def save_results(result_v1: dict, result_v2: dict):
    """保存测试结果到 JSON 文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = OUTPUT_DIR / f"optimization_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 计算对比数据
    s1 = result_v1["summary"]
    s2 = result_v2["summary"]
    comparison = {
        "accuracy_change": round((s2['accuracy'] - s1['accuracy']) * 100, 1),
        "total_tokens_saved": s1['total_tokens'] - s2['total_tokens'],
        "token_saving_percent": round((s1['total_tokens'] - s2['total_tokens']) / s1['total_tokens'] * 100, 1) if s1['total_tokens'] > 0 else 0,
        "avg_input_tokens_saved": round(s1['avg_input_tokens_per_q'] - s2['avg_input_tokens_per_q'], 1),
    }
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "comparison": comparison,
        "before": result_v1,
        "after": result_v2,
    }
    
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 对比报告已保存: {output_file}")


def main():
    print("=" * 60)
    print("  Skill 优化对比测试")
    print("=" * 60)
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    
    # Step 1: 备份初始 skill
    print("\n📦 Step 1: 备份初始 Skill...")
    backup_initial_skill()
    
    # Step 2: 测试初始 skill
    print("\n🧪 Step 2: 测试初始 Skill (优化前)...")
    restore_initial_skill()
    result_v1 = measure_skill("v1_初始版本")
    
    s1 = result_v1["summary"]
    print(f"\n  初始版本结果: 准确率={s1['accuracy']:.1%}, 总token={s1['total_tokens']}, 平均输入token={s1['avg_input_tokens_per_q']:.0f}")
    
    # 打印初始 skill 内容和 token 统计
    print(f"\n  初始 Skill 内容:")
    for name, tokens in result_v1["token_stats"]["skill_token_counts"].items():
        content = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
        line_count = content.count('\n') + 1
        char_count = len(content)
        print(f"    {name}: {line_count}行, {char_count}字符, ~{tokens} token")
    
    # Step 3: 等待用户优化 skill（由外部脚本完成）
    print(f"\n🔧 Step 3: 请现在优化 logistics skill...")
    print(f"  优化策略: 精简内容、去除冗余、使用更紧凑的表达")
    print(f"  优化完成后，将自动进行对比测试")
    
    input(f"\n  优化完成后，请按 Enter 键继续...")
    
    # Step 4: 测试优化后的 skill
    print("\n🧪 Step 4: 测试优化后 Skill...")
    result_v2 = measure_skill("v2_优化版本")
    
    s2 = result_v2["summary"]
    print(f"\n  优化版本结果: 准确率={s2['accuracy']:.1%}, 总token={s2['total_tokens']}, 平均输入token={s2['avg_input_tokens_per_q']:.0f}")
    
    # Step 5: 打印对比报告
    print_comparison(result_v1, result_v2)
    
    # Step 6: 保存结果
    save_results(result_v1, result_v2)
    
    print(f"\n{'='*60}")
    print("  测试完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
