"""
分析汇总工具
整合所有分析结果并生成结构化报告
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _unwrap_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    解包 SubAgent 返回的结果

    SubAgent 返回格式：{"task": ..., "status": ..., "result": ...}
    直接工具返回格式：{"total_samples": ..., "distribution": ..., ...}
    """
    if "result" in result and isinstance(result["result"], dict):
        return result["result"]
    return result


def generate_summary(
    analysis_results: Dict[str, Dict[str, Any]],
    dataset_name: str = "数据集",
    output_format: str = "markdown"
) -> str:
    """
    生成综合分析汇总报告

    Args:
        analysis_results: 包含所有分析结果的字典
            - label_distribution: 标签分布分析结果
            - text_length: 文本长度分析结果
            - quality_check: 数据质量检查结果
        dataset_name: 数据集名称
        output_format: 输出格式 ('markdown' 或 'html')

    Returns:
        格式化的报告文本
    """
    # 解包可能的 SubAgent 包装格式
    unwrapped_results = {}
    for key, value in analysis_results.items():
        if isinstance(value, dict):
            unwrapped_results[key] = _unwrap_result(value)
        else:
            unwrapped_results[key] = value

    analysis_results = unwrapped_results

    # 生成报告头部
    report_lines = [
        f"# {dataset_name} 数据分析报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # 1. 数据集基本信息
    report_lines.extend(_generate_dataset_info_section(analysis_results, dataset_name))

    # 2. 标签分布分析
    if "label_distribution" in analysis_results:
        report_lines.extend(_generate_label_section(analysis_results["label_distribution"]))

    # 3. 文本长度分析
    if "text_length" in analysis_results:
        report_lines.extend(_generate_text_length_section(analysis_results["text_length"]))

    # 4. 数据质量检查
    if "quality_check" in analysis_results:
        report_lines.extend(_generate_quality_section(analysis_results["quality_check"]))

    # 5. 综合评估与建议
    report_lines.extend(_generate_summary_section(analysis_results))

    if output_format == "html":
        return _convert_to_html("\n".join(report_lines))

    return "\n".join(report_lines)


def _generate_dataset_info_section(analysis_results: Dict[str, Any], dataset_name: str) -> List[str]:
    """生成数据集基本信息部分"""
    lines = ["## 1. 数据集基本信息", ""]

    # 从各个分析结果中提取信息
    total_samples = 0
    if "label_distribution" in analysis_results:
        total_samples = analysis_results["label_distribution"].get("total_samples", 0)
    elif "text_length" in analysis_results:
        total_samples = analysis_results["text_length"].get("total_samples", 0)
    elif "quality_check" in analysis_results:
        total_samples = analysis_results["quality_check"].get("total_samples", 0)

    num_classes = analysis_results.get("label_distribution", {}).get("num_classes", 0)

    lines.append(f"- **数据集名称**: {dataset_name}")
    lines.append(f"- **总样本数**: {total_samples}")
    lines.append(f"- **类别数**: {num_classes}")
    lines.append("")

    return lines


def _generate_label_section(label_result: Dict[str, Any]) -> List[str]:
    """生成标签分布分析部分"""
    if "error" in label_result:
        return ["## 2. 标签分布分析", "", f"分析失败：{label_result['error']}", ""]

    lines = [
        "## 2. 标签分布分析",
        "",
        f"**总样本数**: {label_result['total_samples']}",
        f"**类别数**: {label_result['num_classes']}",
        "",
        "### 2.1 分布详情",
        "",
        "| 类别 | 样本数 | 占比 |",
        "|------|--------|------|",
    ]

    for item in label_result.get("distribution", []):
        lines.append(f"| {item['name']} | {item['count']} | {item['percentage']}% |")

    lines.extend([
        "",
        "### 2.2 均衡性指标",
        "",
        f"- **最大/最小比值**: {label_result['balance_metrics']['imbalance_ratio']:.2f}",
        f"- **熵值**: {label_result['balance_metrics']['entropy']:.4f}",
        f"- **归一化熵",
        f"- **基尼系数**: {label_result['balance_metrics']['gini_coefficient']:.4f}",
        "",
        f"### 2.3 评估结果：{label_result['assessment']}",
        "",
        f"{label_result['assessment_detail']}",
        "",
    ])

    if label_result.get("suggestions"):
        lines.append("### 2.4 优化建议")
        lines.append("")
        for i, suggestion in enumerate(label_result["suggestions"], 1):
            lines.append(f"{i}. {suggestion}")
        lines.append("")

    return lines


def _generate_text_length_section(text_result: Dict[str, Any]) -> List[str]:
    """生成文本长度分析部分"""
    if "error" in text_result:
        return ["## 3. 文本长度分析", "", f"分析失败：{text_result['error']}", ""]

    lines = [
        "## 3. 文本长度分析",
        "",
        f"**总样本数**: {text_result['total_samples']}",
        f"**长度单位**: {'字符' if text_result['length_unit'] == 'char' else '词'}",
        "",
        "### 3.1 描述性统计",
        "",
        f"- **平均值**: {text_result['length_stats']['mean']:.2f}",
        f"- **标准差",
        f"- **中位数**: {text_result['length_stats']['median']:.2f}",
        f"- **最小值**: {text_result['length_stats']['min']}",
        f"- **最大值**: {text_result['length_stats']['max']}",
        "",
        "### 3.2 分位数",
        "",
        f"- **P10**: {text_result['percentiles']['p10']:.0f}",
        f"- **P25**: {text_result['percentiles']['p25']:.0f}",
        f"- **P50**: {text_result['percentiles']['p50']:.0f}",
        f"- **P75**: {text_result['percentiles']['p75']:.0f}",
        f"- **P90**: {text_result['percentiles']['p90']:.0f}",
        f"- **P99**: {text_result['percentiles']['p99']:.0f}",
        "",
    ]

    if text_result.get("suggestions"):
        lines.append("### 3.3 优化建议")
        lines.append("")
        for i, suggestion in enumerate(text_result["suggestions"], 1):
            lines.append(f"{i}. {suggestion}")
        lines.append("")

    return lines


def _generate_quality_section(quality_result: Dict[str, Any]) -> List[str]:
    """生成数据质量检查部分"""
    if "error" in quality_result:
        return ["## 4. 数据质量检查", "", f"分析失败：{quality_result['error']}", ""]

    lines = [
        "## 4. 数据质量检查",
        "",
        f"**总样本数**: {quality_result['total_samples']}",
        f"**质量评分**: {quality_result['quality_score']}/100",
        "",
        "### 4.1 缺失值分析",
        "",
        f"- **总体缺失率**: {quality_result['missing_values']['missing_rate']:.2f}%",
        "",
        "### 4.2 重复值分析",
        "",
        f"- **重复样本数**: {quality_result['duplicates']['duplicate_count']}",
        f"- **重复率**: {quality_result['duplicates']['duplicate_rate']:.2f}%",
        "",
        "### 4.3 一致性检查",
        "",
        f"- **不一致样本数**: {quality_result['consistency']['inconsistency_count']}",
        f"- **不一致率**: {quality_result['consistency']['inconsistency_rate']:.2f}%",
        "",
        "### 4.4 异常检测",
        "",
        f"- **空文本数**: {quality_result['anomalies']['empty_text_count']}",
        f"- **空标签数**: {quality_result['anomalies']['empty_label_count']}",
        f"- **异常长文本数**: {quality_result['anomalies']['unusual_length_count']}",
        f"- **含特殊字符数**: {quality_result['anomalies']['special_char_count']}",
        "",
    ]

    if quality_result.get("suggestions"):
        lines.append("### 4.5 优化建议")
        lines.append("")
        for i, suggestion in enumerate(quality_result["suggestions"], 1):
            lines.append(f"{i}. {suggestion}")
        lines.append("")

    return lines


def _generate_summary_section(analysis_results: Dict[str, Any]) -> List[str]:
    """生成综合评估与建议部分"""
    lines = [
        "## 5. 综合评估与建议",
        "",
    ]

    # 综合质量评估
    overall_assessment = "良好"
    concerns = []

    # 检查标签分布
    if "label_distribution" in analysis_results:
        label_result = analysis_results["label_distribution"]
        if label_result.get("assessment") != "均衡":
            concerns.append(f"标签分布{label_result.get('assessment', '不均衡')}")

    # 检查数据质量
    if "quality_check" in analysis_results:
        quality_result = analysis_results["quality_check"]
        score = quality_result.get("quality_score", 0)
        if score < 60:
            concerns.append(f"数据质量评分较低 ({score}分)")
        elif score < 80:
            concerns.append(f"数据质量有待提升 ({score}分)")

    if concerns:
        overall_assessment = "需要关注"
        lines.append(f"**整体评估**: {overall_assessment}")
        lines.append("")
        lines.append("**关注点**:")
        for concern in concerns:
            lines.append(f"- {concern}")
    else:
        lines.append(f"**整体评估**: {overall_assessment}")

    lines.append("")

    # 综合建议
    lines.append("### 5.1 综合建议")
    lines.append("")

    all_suggestions = set()

    # 收集所有建议
    for key, result in analysis_results.items():
        if isinstance(result, dict) and "suggestions" in result:
            for suggestion in result["suggestions"]:
                all_suggestions.add(suggestion)

    if all_suggestions:
        for i, suggestion in enumerate(sorted(all_suggestions), 1):
            lines.append(f"{i}. {suggestion}")
    else:
        lines.append("数据集质量良好，可直接用于后续分析或建模。")

    lines.append("")

    return lines


def _convert_to_html(markdown_text: str) -> str:
    """将 Markdown 转换为简单 HTML"""
    # 简单的 Markdown 到 HTML 转换
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <title>数据分析报告</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 40px; }",
        "        h1 { color: #333; }",
        "        h2 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }",
        "        h3 { color: #888; }",
        "        table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        ul { line-height: 1.8; }",
        "    </style>",
        "</head>",
        "<body>",
    ]

    for line in markdown_text.split("\n"):
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("**") and line.endswith("**"):
            html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            # 表格行
            cells = line.strip("|").split("|")
            if all(c.strip().startswith("-") and c.strip().endswith("-") for c in cells):
                continue  # 跳过表格分隔行
            html_lines.append("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in cells) + "</tr>")
        elif line:
            html_lines.append(f"<p>{line}</p>")

    html_lines.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(html_lines)


def generate_optimization_suggestions(analysis_results: Dict[str, Any]) -> List[str]:
    """
    基于分析结果生成优化建议

    Args:
        analysis_results: 包含所有分析结果的字典

    Returns:
        建议列表
    """
    # 解包可能的 SubAgent 包装格式
    unwrapped_results = {}
    for key, value in analysis_results.items():
        if isinstance(value, dict):
            unwrapped_results[key] = _unwrap_result(value)
        else:
            unwrapped_results[key] = value

    analysis_results = unwrapped_results

    suggestions = []

    # 基于标签分布的建议
    if "label_distribution" in analysis_results:
        label_result = analysis_results["label_distribution"]
        if label_result.get("assessment") == "严重不均衡":
            suggestions.append("【标签分布】数据严重不均衡，建议：1) 对少数类进行过采样；2) 使用类别权重；3) 收集更多少数类样本")
        elif label_result.get("assessment") == "中度不均衡":
            suggestions.append("【标签分布】数据中度不均衡，建议在模型训练时考虑类别权重")

    # 基于文本长度的建议
    if "text_length" in analysis_results:
        text_result = analysis_results["text_length"]
        if text_result.get("empty_count", 0) > 0:
            suggestions.append(f"【文本质量】发现{text_result['empty_count']}个空文本，建议清理或补充")
        if text_result.get("outliers", {}).get("outlier_percentage", 0) > 5:
            suggestions.append("【文本质量】异常长度文本比例较高，建议检查数据质量")

    # 基于数据质量的建议
    if "quality_check" in analysis_results:
        quality_result = analysis_results["quality_check"]
        if quality_result.get("quality_score", 100) < 80:
            suggestions.append(f"【整体质量】质量评分为{quality_result['quality_score']}，建议进行数据清洗")
        if quality_result.get("duplicates", {}).get("duplicate_rate", 0) > 1:
            suggestions.append("【重复数据】发现重复样本，建议去重处理")

    return suggestions
