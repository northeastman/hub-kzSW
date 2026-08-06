"""
Skill 优化效果静态对比分析脚本
无需 API Key，通过静态分析对比优化前后的 Skill

分析维度：
1. Token 消耗估算（基于字符数）
2. 关键信息覆盖度
3. 结构清晰度
4. 理论准确率影响
"""

import json
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "skills"
EVAL_SET = ROOT / "data" / "eval_set.json"
OUTPUT_DIR = ROOT / "outputs" / "optimization_test"
BACKUP_DIR = ROOT / "outputs" / "logistics_v1_backup"

LOGISTICS_QUESTION_IDS = [46, 47, 48, 49, 50, 51, 52, 53]


def estimate_tokens(text: str) -> int:
    """
    估算 token 数（基于经验值）
    中文：1 字 ≈ 1 token
    英文：1 词 ≈ 1.3 token
    标点/符号：1 个 ≈ 1 token
    """
    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 英文单词
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    # 数字
    digits = len(re.findall(r'\d+', text))
    # 标点和其他字符
    other_chars = len(text) - chinese_chars - len(re.findall(r'[a-zA-Z]', text)) - len(re.findall(r'\d', text))
    
    return int(chinese_chars * 1 + english_words * 1.3 + digits * 0.5 + other_chars * 0.8)


def get_skill_content(skill_name: str, version: str = "current") -> str:
    """读取指定版本的 skill 内容"""
    if version == "v1":
        skill_file = BACKUP_DIR / "logistics" / "SKILL.md"
    else:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
    return skill_file.read_text(encoding="utf-8")


def analyze_structure(content: str) -> dict:
    """分析文档结构"""
    lines = content.split('\n')
    headers = [l for l in lines if l.startswith('#')]
    tables = [l for l in lines if '|' in l and not re.match(r'^\|[-| :]+\|$', l)]
    list_items = [l for l in lines if re.match(r'^[-*]\s', l)]
    bold_items = [l for l in lines if '**' in l]
    
    return {
        "total_lines": len(lines),
        "total_chars": len(content),
        "header_count": len(headers),
        "table_rows": max(0, len(tables) - len([l for l in lines if re.match(r'^\|[-| :]+\|$', l)])),
        "list_items": len(list_items),
        "bold_sections": len(bold_items),
    }


def check_keyword_coverage(content: str, question_data: dict) -> dict:
    """
    检查 skill 内容对评估题目的关键词覆盖度
    基于评估集中的 ground truth required 关键词
    """
    content_lower = content.lower()
    results = []
    
    for qid in LOGISTICS_QUESTION_IDS:
        q = question_data[qid]
        gt = q["ground_truth"]
        required = [kw.lower() for kw in gt.get("required", [])]
        forbidden = [kw.lower() for kw in gt.get("forbidden", [])]
        
        # 检查 required 关键词是否存在
        found_required = []
        missing_required = []
        for kw in required:
            if kw in content_lower:
                found_required.append(kw)
            else:
                missing_required.append(kw)
        
        results.append({
            "id": qid,
            "question": q["question"],
            "required_found": found_required,
            "required_missing": missing_required,
            "coverage_ratio": round(len(found_required) / len(required), 2) if required else 1.0,
        })
    
    return results


def calculate_effective_coverage(coverage_results: list[dict]) -> dict:
    """计算有效覆盖率"""
    total_required = 0
    total_found = 0
    fully_covered = 0
    partially_covered = 0
    not_covered = 0
    
    for r in coverage_results:
        total_required += len(r["required_found"]) + len(r["required_missing"])
        total_found += len(r["required_found"])
        
        if r["coverage_ratio"] >= 0.8:
            fully_covered += 1
        elif r["coverage_ratio"] >= 0.5:
            partially_covered += 1
        else:
            not_covered += 1
    
    return {
        "total_questions": len(coverage_results),
        "fully_covered": fully_covered,
        "partially_covered": partially_covered,
        "not_covered": not_covered,
        "coverage_score": round(total_found / total_required, 3) if total_required > 0 else 0,
        "avg_coverage_per_q": round(sum(r["coverage_ratio"] for r in coverage_results) / len(coverage_results), 3),
    }


def estimate_accuracy(coverage_results: list[dict]) -> dict:
    """
    基于关键词覆盖度估算理论准确率
    注意：这是估算，实际准确率需 LLM 测试
    """
    # 简单估算：覆盖率 >= 80% 的题目可能答对
    likely_correct = sum(1 for r in coverage_results if r["coverage_ratio"] >= 0.8)
    partial = sum(1 for r in coverage_results if 0.5 <= r["coverage_ratio"] < 0.8)
    
    # 部分覆盖的题目有约 50% 概率答对
    estimated_correct = likely_correct + partial * 0.5
    accuracy = round(estimated_correct / len(coverage_results), 3)
    
    return {
        "likely_correct": likely_correct,
        "partial_coverage": partial,
        "unlikely_correct": len(coverage_results) - likely_correct - partial,
        "estimated_accuracy": accuracy,
        "confidence": "medium" if partial > 0 else ("high" if likely_correct > 0 else "low"),
    }


def generate_comparison_report(v1_content: str, v2_content: str, question_data: dict):
    """生成完整的对比报告"""
    
    # === 1. Token 分析 ===
    v1_chars = len(v1_content)
    v2_chars = len(v2_content)
    v1_tokens = estimate_tokens(v1_content)
    v2_tokens = estimate_tokens(v2_content)
    token_saving_pct = round((v1_tokens - v2_tokens) / v1_tokens * 100, 1) if v1_tokens > 0 else 0
    
    # === 2. 结构分析 ===
    v1_structure = analyze_structure(v1_content)
    v2_structure = analyze_structure(v2_content)
    
    # === 3. 覆盖度分析 ===
    v1_coverage = check_keyword_coverage(v1_content, question_data)
    v2_coverage = check_keyword_coverage(v2_content, question_data)
    
    v1_coverage_summary = calculate_effective_coverage(v1_coverage)
    v2_coverage_summary = calculate_effective_coverage(v2_coverage)
    
    # === 4. 预估准确率 ===
    v1_accuracy_est = estimate_accuracy(v1_coverage)
    v2_accuracy_est = estimate_accuracy(v2_coverage)
    
    # === 5. 生成报告 ===
    report = {
        "generated_at": datetime.now().isoformat(),
        "optimization_target": "token_reduction",
        "comparison": {
            "token_analysis": {
                "v1_chars": v1_chars,
                "v2_chars": v2_chars,
                "char_reduction_pct": round((v1_chars - v2_chars) / v1_chars * 100, 1),
                "v1_estimated_tokens": v1_tokens,
                "v2_estimated_tokens": v2_tokens,
                "token_saved": v1_tokens - v2_tokens,
                "token_saving_pct": token_saving_pct,
                "note": "Token 为基于字符类型的估算值，实际值取决于分词器"
            },
            "structure_analysis": {
                "v1": v1_structure,
                "v2": v2_structure,
                "improvements": [
                    f"表格使用: {v2_structure['table_rows']} 行表格（v1: {v1_structure['table_rows']} 行）",
                    f"列表项: {v2_structure['list_items']} 项（v1: {v1_structure['list_items']} 项）",
                    f"文档行数: {v1_structure['total_lines']} → {v2_structure['total_lines']} (减少 {v1_structure['total_lines'] - v2_structure['total_lines']} 行)",
                ]
            },
            "coverage_analysis": {
                "v1": v1_coverage_summary,
                "v2": v2_coverage_summary,
                "per_question_comparison": []
            },
            "estimated_accuracy": {
                "v1": v1_accuracy_est,
                "v2": v2_accuracy_est,
                "note": "基于关键词覆盖度的理论估算，实际准确率需 LLM 测试"
            }
        },
        "question_details": []
    }
    
    # 添加逐题对比
    for i, (c1, c2) in enumerate(zip(v1_coverage, v2_coverage)):
        detail = {
            "id": c1["id"],
            "question": c1["question"],
            "v1_coverage": {
                "ratio": c1["coverage_ratio"],
                "missing": c1["required_missing"]
            },
            "v2_coverage": {
                "ratio": c2["coverage_ratio"],
                "missing": c2["required_missing"]
            },
            "improved": c2["coverage_ratio"] > c1["coverage_ratio"],
            "regressed": c2["coverage_ratio"] < c1["coverage_ratio"],
            "note": c2["required_missing"][:50] if c2["required_missing"] else "全覆盖"
        }
        report["question_details"].append(detail)
        report["comparison"]["coverage_analysis"]["per_question_comparison"].append(detail)
    
    return report


def print_report(report: dict):
    """打印格式化的对比报告"""
    comp = report["comparison"]
    ta = comp["token_analysis"]
    sa = comp["structure_analysis"]
    ca = comp["coverage_analysis"]
    ea = comp["estimated_accuracy"]
    
    print(f"\n{'='*70}")
    print(f"  Skill 优化效果 - 静态对比分析报告")
    print(f"{'='*70}")
    
    # === Token 对比 ===
    print(f"\n{'─'*70}")
    print(f"  1. Token 消耗分析（估算）")
    print(f"{'─'*70}")
    print(f"  {'指标':<25} {'优化前 (v1)':<20} {'优化后 (v2)':<20} {'变化':<15}")
    print(f"  {'─'*70}")
    print(f"  {'字符数':<23} {ta['v1_chars']:<20} {ta['v2_chars']:<20} ↓{ta['char_reduction_pct']}%")
    print(f"  {'估算 token':<23} {ta['v1_estimated_tokens']:<20} {ta['v2_estimated_tokens']:<20} ↓{ta['token_saving_pct']}%")
    print(f"  {'节省 token':<23} {'':<20} {ta['token_saved']:<20} tokens")
    print(f"\n  注：{ta['note']}")
    
    # === 结构对比 ===
    print(f"\n{'─'*70}")
    print(f"  2. 文档结构对比")
    print(f"{'─'*70}")
    print(f"  {'指标':<25} {'v1':<20} {'v2':<20}")
    print(f"  {'─'*65}")
    print(f"  {'总行数':<23} {sa['v1']['total_lines']:<20} {sa['v2']['total_lines']:<20}")
    print(f"  {'标题数':<23} {sa['v1']['header_count']:<20} {sa['v2']['header_count']:<20}")
    print(f"  {'表格行数':<23} {sa['v1']['table_rows']:<20} {sa['v2']['table_rows']:<20}")
    print(f"  {'列表项数':<23} {sa['v1']['list_items']:<20} {sa['v2']['list_items']:<20}")
    
    print(f"\n  结构改进:")
    for imp in sa["improvements"]:
        print(f"    • {imp}")
    
    # === 覆盖度分析 ===
    print(f"\n{'─'*70}")
    print(f"  3. 关键信息覆盖度分析")
    print(f"{'─'*70}")
    
    v1c = ca["v1"]
    v2c = ca["v2"]
    
    print(f"  {'指标':<25} {'v1':<20} {'v2':<20}")
    print(f"  {'─'*65}")
    print(f"  {'完全覆盖题数':<23} {v1c['fully_covered']:<20} {v2c['fully_covered']:<20}")
    print(f"  {'部分覆盖题数':<23} {v1c['partially_covered']:<20} {v2c['partially_covered']:<20}")
    print(f"  {'未覆盖题数':<23} {v1c['not_covered']:<20} {v2c['not_covered']:<20}")
    v1_avg_cov = f"{v1c['avg_coverage_per_q']:.1%}"
    v2_avg_cov = f"{v2c['avg_coverage_per_q']:.1%}"
    v1_score = f"{v1c['coverage_score']:.1%}"
    v2_score = f"{v2c['coverage_score']:.1%}"
    print(f"  {'平均覆盖率':<23} {v1_avg_cov:<20} {v2_avg_cov:<20}")
    print(f"  {'综合覆盖分':<23} {v1_score:<20} {v2_score:<20}")
    
    # === 预估准确率 ===
    print(f"\n{'─'*70}")
    print(f"  4. 理论准确率估算（基于关键词覆盖度）")
    print(f"{'─'*70}")
    print(f"  {'指标':<25} {'v1':<20} {'v2':<20}")
    print(f"  {'─'*65}")
    print(f"  {'高概率正确(≥80%覆盖)':<23} {ea['v1']['likely_correct']:<20} {ea['v2']['likely_correct']:<20}")
    print(f"  {'部分正确(50-80%覆盖)':<23} {ea['v1']['partial_coverage']:<20} {ea['v2']['partial_coverage']:<20}")
    v1_acc = f"{ea['v1']['estimated_accuracy']:.1%}"
    v2_acc = f"{ea['v2']['estimated_accuracy']:.1%}"
    print(f"  {'理论准确率':<23} {v1_acc:<20} {v2_acc:<20}")
    print(f"\n  注：{ea['note']}")
    
    # === 逐题对比 ===
    print(f"\n{'─'*70}")
    print(f"  5. 逐题覆盖度对比")
    print(f"{'─'*70}")
    print(f"  {'ID':<5} {'题目':<30} {'v1覆盖':<10} {'v2覆盖':<10} {'变化':<10} {'缺失关键词'}")
    print(f"  {'─'*110}")
    
    for detail in report["question_details"]:
        v1r = f"{detail['v1_coverage']['ratio']:.0%}"
        v2r = f"{detail['v2_coverage']['ratio']:.0%}"
        if detail["improved"]:
            change = "↑ 改善"
        elif detail["regressed"]:
            change = "↓ 下降"
        else:
            change = "→ 持平"
        missing = detail["v2_coverage"]["missing"][:40] if detail["v2_coverage"]["missing"] else "无"
        print(f"  Q{detail['id']:02d}  {detail['question'][:27]:<30} {v1r:<10} {v2r:<10} {change:<10} {missing}")
    
    # === 总结 ===
    print(f"\n{'─'*70}")
    print(f"  6. 优化效果总结")
    print(f"{'─'*70}")
    
    token_saved = ta['token_saved']
    coverage_change = round((v2c['coverage_score'] - v1c['coverage_score']) * 100, 1)
    accuracy_change = round((ea['v2']['estimated_accuracy'] - ea['v1']['estimated_accuracy']) * 100, 1)
    
    print(f"  ✓ Token 节省: {token_saved} tokens ({ta['token_saving_pct']}%)")
    print(f"  ✓ 字符压缩: {ta['v1_chars']} → {ta['v2_chars']} (↓{ta['char_reduction_pct']}%)")
    print(f"  ✓ 覆盖度变化: {coverage_change:+.1f}%")
    print(f"  ✓ 理论准确率变化: {accuracy_change:+.1f}%")
    
    # 评价
    if coverage_change >= 0 and ta['token_saving_pct'] > 20:
        print(f"\n  ⭐ 优化优秀！在保持信息完整的同时，大幅减少了 token 消耗")
    elif coverage_change >= -5 and ta['token_saving_pct'] > 10:
        print(f"\n  ✓ 优化良好！信息覆盖基本完整，有效减少了 token 消耗")
    elif coverage_change < -10 and ta['token_saving_pct'] > 20:
        print(f"\n  ⚠ 权衡取舍：大幅节省 token，但损失了部分信息覆盖度")
    else:
        print(f"\n  ⚡ 优化效果一般，建议进一步调整")


def main():
    print("=" * 60)
    print("  Skill 优化效果静态对比分析")
    print("=" * 60)
    
    # 检查备份是否存在
    if not BACKUP_DIR.exists():
        print("错误: 初始 Skill 备份不存在！")
        print("请先运行主实验脚本或手动创建备份")
        return
    
    # 读取 v1 内容（从备份）
    v1_content = get_skill_content("logistics", "v1")
    
    # 读取 v2 内容（当前）
    v2_content = get_skill_content("logistics", "current")
    
    print(f"\n已加载:")
    print(f"  v1 (初始版): {len(v1_content)} 字符")
    print(f"  v2 (优化版): {len(v2_content)} 字符")
    
    # 加载评估集
    eval_data = json.loads((ROOT / EVAL_SET).read_text(encoding="utf-8"))
    question_data = {q["id"]: q for q in eval_data["questions"]}
    
    # 生成报告
    print("\n正在分析...")
    report = generate_comparison_report(v1_content, v2_content, question_data)
    
    # 打印报告
    print_report(report)
    
    # 保存报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"static_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 完整报告已保存: {output_file}")
    
    # 同时保存两份 skill 内容便于查看
    v1_save = OUTPUT_DIR / "logistics_v1_SKILL.md"
    v2_save = OUTPUT_DIR / "logistics_v2_SKILL.md"
    v1_save.write_text(v1_content, encoding="utf-8")
    v2_save.write_text(v2_content, encoding="utf-8")
    print(f"✓ Skill 内容已保存:")
    print(f"    v1: {v1_save}")
    print(f"    v2: {v2_save}")


if __name__ == "__main__":
    main()
