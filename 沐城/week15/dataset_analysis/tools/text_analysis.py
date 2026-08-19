"""
文本长度分析工具
统计数据集的文本长度分布情况
"""
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def analyze_text_length(
    texts: List[str],
    dataset_name: str = "数据集",
    length_unit: str = "char"
) -> Dict[str, Any]:
    """
    分析文本长度分布

    Args:
        texts: 文本列表
        dataset_name: 数据集名称
        length_unit: 长度单位，'char' 为字符数，'word' 为词数

    Returns:
        分析结果字典，包含:
        - total_samples: 总样本数
        - length_stats: 长度统计指标
        - percentiles: 分位数
        - distribution: 分布详情
        - outliers: 异常值分析
        - suggestions: 优化建议
    """
    if not texts:
        return {"error": "文本列表为空"}

    # 计算每个文本的长度
    lengths = []
    empty_count = 0
    for text in texts:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if length_unit == "word":
            # 按词数计算（简单分词）
            length = len(text.split())
        else:
            # 按字符数计算
            length = len(text)

        lengths.append(length)
        if length == 0:
            empty_count += 1

    lengths = np.array(lengths)
    total_samples = len(lengths)

    # 描述性统计
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    median_length = np.median(lengths)

    # 分位数
    percentiles = {
        "p10": float(np.percentile(lengths, 10)),
        "p25": float(np.percentile(lengths, 25)),
        "p50": float(np.percentile(lengths, 50)),  # 中位数
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }

    # 分布分布详情（用于直方图）
    # 使用 Freedman-Diaconis 规则确定 bin 数量
    q1, q3 = percentiles["p25"], percentiles["p75"]
    iqr = q3 - q1
    if iqr > 0:
        bin_width = 2 * iqr / (total_samples ** (1/3))
        num_bins = int((max_length - min_length) / bin_width) if bin_width > 0 else 10
    else:
        num_bins = 10

    num_bins = max(5, min(num_bins, 50))  # 限制在 5-50 之间
    hist, bin_edges = np.histogram(lengths, bins=num_bins)

    distribution_info = []
    for i in range(len(hist)):
        distribution_info.append({
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "count": int(hist[i]),
            "percentage": round(float(hist[i] / total_samples * 100), 2),
        })

    # 异常值检测（使用 IQR 方法）
    lower_bound = q1 - 1.5 * iqr if iqr > 0 else min_length
    upper_bound = q3 + 1.5 * iqr if iqr > 0 else max_length

    outliers = {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(np.sum((lengths < lower_bound) | (lengths > upper_bound))),
        "outlier_percentage": round(float(np.sum((lengths < lower_bound) | (lengths > upper_bound)) / total_samples * 100), 2),
        "min_outliers": [i for i, l in enumerate(lengths) if l < lower_bound][:10],  # 最多返回 10 个
        "max_outliers": [i for i, l in enumerate(lengths) if l > upper_bound][:10],
    }

    # 生成建议
    suggestions = []

    if empty_count > 0:
        empty_pct = round(empty_count / total_samples * 100, 2)
        suggestions.append(f"发现 {empty_count} 个空文本样本 ({empty_pct}%)，建议检查数据质量")

    if percentiles["p99"] > percentiles["p50"] * 10:
        suggestions.append("存在极长文本，建议检查是否有异常数据混入")

    if outliers["outlier_percentage"] > 5:
        suggestions.append(f"异常值比例较高 ({outliers['outlier_percentage']}%)，建议进行数据清洗")

    if std_length > mean_length:
        suggestions.append("文本长度方差大于均值，分布较为分散")

    if not suggestions:
        suggestions.append("文本长度分布正常，无明显异常")

    result = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "empty_count": empty_count,
        "length_unit": length_unit,
        "length_stats": {
            "mean": round(float(mean_length), 2),
            "std": round(float(std_length), 2),
            "min": int(min_length),
            "max": int(max_length),
            "median": float(median_length),
        },
        "percentiles": percentiles,
        "distribution": distribution_info,
        "outliers": outliers,
        "suggestions": suggestions,
    }

    logger.info(f"文本长度分析完成：{total_samples} 样本，平均长度 {mean_length:.1f} {length_unit}")
    return result


def generate_text_length_report(analysis_result: Dict[str, Any]) -> str:
    """
    生成文本长度分析报告文本

    Args:
        analysis_result: analyze_text_length 返回的结果

    Returns:
        格式化的报告文本
    """
    if "error" in analysis_result:
        return f"分析失败：{analysis_result['error']}"

    lines = [
        f"=== {analysis_result['dataset_name']} 文本长度分析报告 ===",
        "",
        f"总样本数：{analysis_result['total_samples']}",
        f"空文本数：{analysis_result['empty_count']}",
        f"长度单位：{'字符' if analysis_result['length_unit'] == 'char' else '词'}",
        "",
        "--- 长度统计 ---",
        f"  平均值：{analysis_result['length_stats']['mean']}",
        f"  标准差：{analysis_result['length_stats']['std']}",
        f"  最小值：{analysis_result['length_stats']['min']}",
        f"  最大值：{analysis_result['length_stats']['max']}",
        f"  中位数：{analysis_result['length_stats']['median']}",
        "",
        "--- 分位数 ---",
        f"  P10: {analysis_result['percentiles']['p10']}",
        f"  P25: {analysis_result['percentiles']['p25']}",
        f"  P50 (中位数): {analysis_result['percentiles']['p50']}",
        f"  P75: {analysis_result['percentiles']['p75']}",
        f"  P90: {analysis_result['percentiles']['p90']}",
        f"  P99: {analysis_result['percentiles']['p99']}",
        "",
        "--- 异常值检测 ---",
        f"  异常值数量：{analysis_result['outliers']['outlier_count']}",
        f"  异常值比例：{analysis_result['outliers']['outlier_percentage']}%",
        f"  异常值范围：< {analysis_result['outliers']['lower_bound']} 或 > {analysis_result['outliers']['upper_bound']}",
        "",
        "--- 优化建议 ---",
    ]

    for i, suggestion in enumerate(analysis_result["suggestions"], 1):
        lines.append(f"  {i}. {suggestion}")

    return "\n".join(lines)
