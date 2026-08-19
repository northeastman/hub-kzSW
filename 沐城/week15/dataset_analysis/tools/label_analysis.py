"""
标签分布分析工具
统计数据集的标签分布情况，包括均衡性分析
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def analyze_label_distribution(
    labels: List[Any],
    label_names: Optional[Dict[Any, str]] = None,
    dataset_name: str = "数据集"
) -> Dict[str, Any]:
    """
    分析标签分布情况

    Args:
        labels: 标签列表
        label_names: 标签编码到中文名称的映射（可选）
        dataset_name: 数据集名称

    Returns:
        分析结果字典，包含:
        - total_samples: 总样本数
        - num_classes: 类别数
        - distribution: 各类别分布详情
        - balance_metrics: 均衡性指标
        - imbalance_assessment: 不均衡程度评估
        - suggestions: 优化建议
    """
    if not labels:
        return {"error": "标签列表为空"}

    total_samples = len(labels)
    label_counts = Counter(labels)
    num_classes = len(label_counts)

    # 计算分布详情
    distribution = []
    counts = list(label_counts.values())

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        name = label_names.get(label, str(label)) if label_names else str(label)
        percentage = (count / total_samples) * 100
        distribution.append({
            "label": label,
            "name": name,
            "count": count,
            "percentage": round(percentage, 2),
        })

    # 计算均衡性指标
    max_count = max(counts)
    min_count = min(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)

    # 不均衡比率
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    # 熵值计算（衡量分布的混乱程度）
    probabilities = [c / total_samples for c in counts]
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    max_entropy = np.log2(num_classes) if num_classes > 1 else 0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1

    # 基尼系数（衡量不均衡程度）
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * sum((i + 1) * c for i, c in enumerate(sorted_counts)) - (n + 1) * sum(counts)) / (n * sum(counts))

    # 不均衡程度评估
    if imbalance_ratio < 1.5:
        assessment = "均衡"
        assessment_detail = f"最大/最小比值为 {imbalance_ratio:.2f}，分布较为均衡"
    elif imbalance_ratio < 5:
        assessment = "轻度不均衡"
        assessment_detail = f"最大/最小比值为 {imbalance_ratio:.2f}，存在轻度不均衡"
    elif imbalance_ratio < 10:
        assessment = "中度不均衡"
        assessment_detail = f"最大/最小比值为 {imbalance_ratio:.2f}，存在中度不均衡"
    else:
        assessment = "严重不均衡"
        assessment_detail = f"最大/最小比值为 {imbalance_ratio:.2f}，分布严重不均衡"

    # 生成优化建议
    suggestions = []
    if assessment != "均衡":
        suggestions.append(f"考虑对少数类别进行过采样（如 SMOTE）")
        suggestions.append(f"或考虑对多数类别进行欠采样")
        if imbalance_ratio > 10:
            suggestions.append("严重不均衡可能影响模型训练效果，建议优先处理")

    if num_classes < 2:
        suggestions.append("单类别数据集，无法进行分布分析")

    result = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "distribution": distribution,
        "balance_metrics": {
            "max_count": max_count,
            "min_count": min_count,
            "mean_count": round(mean_count, 2),
            "std_count": round(std_count, 2),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "gini_coefficient": round(gini, 4),
        },
        "assessment": assessment,
        "assessment_detail": assessment_detail,
        "suggestions": suggestions,
    }

    logger.info(f"标签分布分析完成：{total_samples} 样本，{num_classes} 类别，{assessment}")
    return result


def generate_label_distribution_report(analysis_result: Dict[str, Any]) -> str:
    """
    生成标签分布分析报告文本

    Args:
        analysis_result: analyze_label_distribution 返回的结果

    Returns:
        格式化的报告文本
    """
    if "error" in analysis_result:
        return f"分析失败：{analysis_result['error']}"

    lines = [
        f"=== {analysis_result['dataset_name']} 标签分布分析报告 ===",
        "",
        f"总样本数：{analysis_result['total_samples']}",
        f"类别数：{analysis_result['num_classes']}",
        "",
        "--- 分布详情 ---",
    ]

    for item in analysis_result["distribution"]:
        lines.append(f"  {item['name']}: {item['count']} ({item['percentage']}%)")

    lines.extend([
        "",
        "--- 均衡性指标 ---",
        f"  最大类别样本数：{analysis_result['balance_metrics']['max_count']}",
        f"  最小类别样本数：{analysis_result['balance_metrics']['min_count']}",
        f"  平均值：{analysis_result['balance_metrics']['mean_count']}",
        f"  标准差：{analysis_result['balance_metrics']['std_count']}",
        f"  不均衡比率：{analysis_result['balance_metrics']['imbalance_ratio']}",
        f"  熵值：{analysis_result['balance_metrics']['entropy']}",
        f"  基尼吉尼系数：{analysis_result['balance_metrics']['gini_coefficient']}",
        "",
        f"--- 评估结果：{analysis_result['assessment']}",
        f"  {analysis_result['assessment_detail']}",
    ])

    if analysis_result["suggestions"]:
        lines.extend([
            "",
            "--- 优化建议 ---",
        ])
        for i, suggestion in enumerate(analysis_result["suggestions"], 1):
            lines.append(f"  {i}. {suggestion}")

    return "\n".join(lines)
