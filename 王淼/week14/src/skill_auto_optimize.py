"""
Skill 自动化优化对比脚本
完整流程：测试初始 Skill → LLM 优化 Skill → 测试优化后 Skill → 生成对比报告

优化目标：在保持准确率不降的前提下，最小化 token 消耗
优化策略：精简冗余内容、使用表格、压缩表达、移除非必要示例
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
from openai import OpenAI

SKILLS_DIR = ROOT / "skills"
EVAL_SET = ROOT / "data" / "eval_set.json"
OUTPUT_DIR = ROOT / "outputs" / "optimization_test"
BACKUP_DIR = ROOT / "outputs" / "logistics_v1_backup"
POLICIES_FILE = ROOT / "data" / "policies.md"

LOGISTICS_QUESTION_IDS = [46, 47, 48, 49, 50, 51, 52, 53]


def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


def backup_skill():
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    logistics_dir = SKILLS_DIR / "logistics"
    if logistics_dir.exists():
        shutil.copytree(logistics_dir, BACKUP_DIR / "logistics")
        print(f"✓ 初始 Skill 已备份")


def restore_skill():
    logistics_dir = SKILLS_DIR / "logistics"
    if logistics_dir.exists():
        shutil.rmtree(logistics_dir)
    shutil.copytree(BACKUP_DIR / "logistics", logistics_dir)


def measure_skill(skill_version: str) -> dict:
    """测量 skill 性能"""
    sm = SkillManager(str(SKILLS_DIR))
    evaluator = Evaluator(str(EVAL_SET))
    agent = CustomerServiceAgent(sm, nudge_interval=0)
    client = get_client()

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

        system_prompt = agent._build_system_prompt()

        start_time = time.time()

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

        results["questions"].append({
            "id": qid,
            "question": question,
            "answer": answer,
            "correct": ok,
            "fail_reason": reason if not ok else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_time_ms": round(elapsed * 1000, 1),
        })

        status = "✓" if ok else "✗"
        print(f"  Q{qid:02d} {status} | 输入:{input_tokens}t 输出:{output_tokens}t | {elapsed*1000:.0f}ms")

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
    }

    all_skills = sm.load_all()
    skill_token_counts = {}
    for name, content in all_skills.items():
        skill_token_counts[name] = len(content)
    results["token_stats"] = {
        "skill_token_counts": skill_token_counts,
        "total_skill_tokens": sum(skill_token_counts.values()),
    }

    return results


def optimize_skill_with_llm():
    """使用 LLM 优化 logistics skill"""
    client = get_client()
    
    # 读取当前 logistics skill
    skill_content = (SKILLS_DIR / "logistics" / "SKILL.md").read_text(encoding="utf-8")
    
    # 读取 policies.md 作为参考
    policies = POLICIES_FILE.read_text(encoding="utf-8")
    
    # 构建优化提示
    optimize_prompt = f"""你是一个专业的 Skill 优化专家。你的任务是优化下面的 Skill 文件，使其在保持所有关键信息和规则准确性的前提下，尽可能减少 token 消耗。

## 优化原则
1. **保持准确性**：所有数字（天数、金额、百分比）和规则优先级必须准确无误
2. **精简冗余**：删除重复表述、合并相似条目、使用缩写
3. **压缩结构**：用表格替代列表、用短句替代长句、移除 frontmatter 中的非必要字段
4. **保留格式**：保留 markdown 格式，但尽可能紧凑
5. **关键信息优先**：高频触发的规则放在前面

## 当前 Skill 内容
```
{skill_content}
```

## 参考政策文档（供你核对）
```
{policies}
```

## 输出要求
请直接输出优化后的 Skill 内容（完整的 markdown），不需要任何解释。优化后的 Skill 应该：
- 去除 frontmatter 中的 description 字段（减少 token）
- 用表格整理配送方式、退货政策等对比信息
- 合并重复条款
- 使用更紧凑的语言表达
- 确保所有评估集需要的关键信息都保留（见下方检查清单）

## 关键信息检查清单（必须全部保留）
- [ ] 标准配送：3-7个工作日，满99元免运费
- [ ] 次日达：仅限北上广深，14:00前下单，加收10元
- [ ] 偏远地区：不支持次日达，固定12元运费
- [ ] 未发货订单取消：48小时内待发货状态可直接取消
- [ ] 取消后24小时内退款，积分原数退回不打折
- [ ] 超过48小时或已发货：走退货退款流程
- [ ] 退货需等收货后，商品保持完好
- [ ] 普通用户/白卡：自付运费（8-15元），30天退货期
- [ ] 银卡/金卡：平台承担运费，60天/90天退货期
- [ ] 白卡VIP不享受VIP退款特权（关键陷阱）
- [ ] 数字商品不支持退货（此规则优先于VIP）

请输出优化后的 Skill 内容："""

    print("\n🔧 正在调用 LLM 优化 Skill...")
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的技术文档优化专家，擅长在保持信息完整性的前提下精简文档。"},
            {"role": "user", "content": optimize_prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    
    elapsed = time.time() - start_time
    optimized_content = response.choices[0].message.content.strip()
    
    # 确保输出以 --- 开头（frontmatter）
    if not optimized_content.startswith("---"):
        # 添加基本的 frontmatter
        optimized_content = f"""---
name: logistics
description: 物流与配送政策
type: knowledge
version: 2
---

{optimized_content}"""
    
    print(f"  LLM 优化完成，耗时 {elapsed:.1f}s")
    print(f"  原始大小: {len(skill_content)} 字符")
    print(f"  优化后大小: {len(optimized_content)} 字符")
    print(f"  压缩率: {(1 - len(optimized_content)/len(skill_content))*100:.1f}%")
    
    # 保存优化后的 skill
    skill_file = SKILLS_DIR / "logistics" / "SKILL.md"
    skill_file.write_text(optimized_content, encoding="utf-8")
    print(f"✓ 优化后的 Skill 已保存")
    
    return optimized_content


def verify_skill_coverage(skill_content: str) -> list[str]:
    """验证优化后的 skill 是否覆盖了所有关键信息"""
    issues = []
    
    checks = [
        ("标准配送", "3", "7", "工作日", "99"),
        ("次日达", "北京", "上海", "广州", "深圳", "14", "10元"),
        ("偏远地区", "新疆", "西藏", "12元"),
        ("取消订单", "48小时", "待发货", "24小时", "退回"),
        ("退货", "完好", "30天", "60天", "90天"),
        ("白卡", "30天", "自付"),
        ("数字商品", "不支持", "优先"),
    ]
    
    skill_lower = skill_content.lower()
    
    for check_group in checks:
        section_name = check_group[0]
        keywords = check_group[1:]
        missing = [kw for kw in keywords if kw.lower() not in skill_lower]
        if missing:
            issues.append(f"  ⚠ [{section_name}] 可能缺失关键词: {', '.join(missing)}")
    
    return issues


def print_comparison(result_v1: dict, result_v2: dict):
    """打印对比报告"""
    s1 = result_v1["summary"]
    s2 = result_v2["summary"]
    
    print(f"\n{'='*70}")
    print(f"  Skill 优化对比报告")
    print(f"{'='*70}")
    
    def fmt_change(v1, v2):
        if v1 == 0:
            return "N/A"
        change = (v2 - v1) / v1 * 100
        arrow = "↓" if change < 0 else "↑"
        sign = "+" if change > 0 else ""
        return f"{sign}{change:.1f}% {arrow}"
    
    rows = [
        ("准确率", f"{s1['accuracy']:.1%}", f"{s2['accuracy']:.1%}", fmt_change(s1['accuracy'], s2['accuracy'])),
        ("正确/总数", f"{s1['correct']}/{s1['total_questions']}", f"{s2['correct']}/{s2['total_questions']}", ""),
        ("平均输入token/题", f"{s1['avg_input_tokens_per_q']:.0f}", f"{s2['avg_input_tokens_per_q']:.0f}", fmt_change(s1['avg_input_tokens_per_q'], s2['avg_input_tokens_per_q'])),
        ("平均输出token/题", f"{s1['avg_output_tokens_per_q']:.0f}", f"{s2['avg_output_tokens_per_q']:.0f}", fmt_change(s1['avg_output_tokens_per_q'], s2['avg_output_tokens_per_q'])),
        ("总token消耗", f"{s1['total_tokens']}", f"{s2['total_tokens']}", fmt_change(s1['total_tokens'], s2['total_tokens'])),
        ("平均响应时间(ms)", f"{s1['avg_time_ms']:.0f}", f"{s2['avg_time_ms']:.0f}", fmt_change(s1['avg_time_ms'], s2['avg_time_ms'])),
    ]
    
    header = f"  {'指标':<25} {'优化前':<20} {'优化后':<20} {'变化':<15}"
    print(header)
    print(f"  {'─'*75}")
    for row in rows:
        print(f"  {row[0]:<23} {row[1]:<20} {row[2]:<20} {row[3]}")
    
    st1 = result_v1["token_stats"]
    st2 = result_v2["token_stats"]
    print(f"\n  Skill 文件 token 对比:")
    print(f"    所有 Skill 总 token: {st1['total_skill_tokens']} → {st2['total_skill_tokens']} ({fmt_change(st1['total_skill_tokens'], st2['total_skill_tokens'])})")
    for skill_name in st1["skill_token_counts"]:
        t1 = st1["skill_token_counts"][skill_name]
        t2 = st2["skill_token_counts"].get(skill_name, 0)
        print(f"      {skill_name}: {t1} → {t2} ({fmt_change(t1, t2)})")
    
    print(f"\n  逐题对比:")
    print(f"    {'ID':<5} {'题目':<30} {'v1':<5} {'v2':<5} {'输入t v1':<10} {'输入t v2':<10}")
    print(f"    {'─'*75}")
    for i, q1 in enumerate(result_v1["questions"]):
        q2 = result_v2["questions"][i]
        s_v1 = "✓" if q1["correct"] else "✗"
        s_v2 = "✓" if q2["correct"] else "✗"
        print(f"    Q{q1['id']:02d}  {q1['question'][:27]:<30} {s_v1:<5} {s_v2:<5} {q1['input_tokens']:<10} {q2['input_tokens']:<10}")
    
    token_saved = s1['total_tokens'] - s2['total_tokens']
    token_saved_pct = (token_saved / s1['total_tokens'] * 100) if s1['total_tokens'] > 0 else 0
    accuracy_change = (s2['accuracy'] - s1['accuracy']) * 100
    
    print(f"\n  ─── 优化效果总结 ───")
    print(f"  • Token节省: {token_saved} tokens ({token_saved_pct:.1f}%)")
    print(f"  • 准确率变化: {accuracy_change:+.1f}%")
    print(f"  • Skill文件大小: {st1['total_skill_tokens']} → {st2['total_skill_tokens']} 字符")
    
    if accuracy_change >= 0 and token_saved_pct > 0:
        print(f"\n  ✓ 优化成功: 在保持/提升准确率的同时，显著减少了 token 消耗")
    elif accuracy_change < 0 and token_saved_pct > 10:
        print(f"\n  ⚠ 警告: 大幅节省了 token，但准确率有所下降")
    elif accuracy_change > 0 and token_saved_pct <= 0:
        print(f"\n  ⚠ 提示: 准确率提升了，但 token 消耗反而增加了")


def save_results(result_v1: dict, result_v2: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = OUTPUT_DIR / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    s1 = result_v1["summary"]
    s2 = result_v2["summary"]
    comparison = {
        "accuracy_change_percent": round((s2['accuracy'] - s1['accuracy']) * 100, 1),
        "total_tokens_saved": s1['total_tokens'] - s2['total_tokens'],
        "token_saving_percent": round((s1['total_tokens'] - s2['total_tokens']) / s1['total_tokens'] * 100, 1) if s1['total_tokens'] > 0 else 0,
        "avg_input_tokens_saved": round(s1['avg_input_tokens_per_q'] - s2['avg_input_tokens_per_q'], 1),
        "skill_tokens_before": result_v1["token_stats"]["total_skill_tokens"],
        "skill_tokens_after": result_v2["token_stats"]["total_skill_tokens"],
    }
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "comparison_summary": comparison,
        "before_optimization": result_v1,
        "after_optimization": result_v2,
    }
    
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 完整对比报告已保存: {output_file}")


def main():
    print("=" * 60)
    print("  Skill 自动化优化对比测试")
    print("=" * 60)
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 请先设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    
    # Step 1: 备份
    print("\n📦 Step 1: 备份初始 Skill...")
    backup_skill()
    restore_skill()
    
    # Step 2: 测试初始 skill
    print("\n🧪 Step 2: 测试初始 Skill (优化前)...")
    result_v1 = measure_skill("v1_初始版本")
    s1 = result_v1["summary"]
    print(f"\n  初始结果: 准确率={s1['accuracy']:.1%}, 总token={s1['total_tokens']}, 平均输入token={s1['avg_input_tokens_per_q']:.0f}")
    
    # 打印初始 skill 内容
    print(f"\n  初始 Skill 详情:")
    for name, tokens in result_v1["token_stats"]["skill_token_counts"].items():
        content = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
        print(f"    {name}: {len(content)} 字符, ~{tokens} token")
    
    # Step 3: LLM 优化 skill
    print("\n🔧 Step 3: LLM 优化 Skill...")
    optimized_content = optimize_skill_with_llm()
    
    # 验证覆盖度
    print("\n📋 Step 3.1: 验证关键信息覆盖度...")
    issues = verify_skill_coverage(optimized_content)
    if issues:
        print("  ⚠ 发现潜在覆盖问题:")
        for issue in issues:
            print(issue)
    else:
        print("  ✓ 所有关键信息均已覆盖")
    
    # Step 4: 测试优化后 skill
    print("\n🧪 Step 4: 测试优化后 Skill...")
    result_v2 = measure_skill("v2_优化版本")
    s2 = result_v2["summary"]
    print(f"\n  优化后结果: 准确率={s2['accuracy']:.1%}, 总token={s2['total_tokens']}, 平均输入token={s2['avg_input_tokens_per_q']:.0f}")
    
    # Step 5: 打印对比
    print_comparison(result_v1, result_v2)
    
    # Step 6: 保存结果
    save_results(result_v1, result_v2)
    
    # Step 7: 还原初始 skill（方便后续使用）
    print("\n🔄 Step 7: 还原初始 Skill...")
    # 保留优化后的版本，放在单独目录
    optimized_backup = ROOT / "outputs" / "logistics_v2_optimized"
    if optimized_backup.exists():
        shutil.rmtree(optimized_backup)
    shutil.copytree(SKILLS_DIR / "logistics", optimized_backup)
    restore_skill()
    print(f"  优化版已保存至 {optimized_backup}")
    print(f"  已还原初始 Skill")
    
    print(f"\n{'='*60}")
    print("  实验完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
