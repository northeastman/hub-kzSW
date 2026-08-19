"""
数据质量检查工具
检查数据集的质量问题，包括缺失值、重复值、异常值等
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def check_data_quality(
    data: Union[List[Dict[str, Any]], Dict[str, List[Any]]],
    text_column: str = "text",
    label_column: str = "label",
    dataset_name: str = "数据集"
) -> Dict[str, Any]:
    """
    检查数据质量

    Args:
        data: 数据集，可以是是列表（字典列表）或字典（列式数据）
        text_column: 文本列名
        label_column: 标签列名
        dataset_name: 数据集名称

    Returns:
        分析结果字典，包含:
        - total_samples: 总样本数
        - missing_values: 缺失值分析
        - duplicates: 重复值分析
        - consistency: 一致性检查
        - anomalies: 异常检测
        - quality_score: 质量评分
        - suggestions: 优化建议
    """
    # 统一转换为列式数据
    if isinstance(data, list):
        # 列表格式转换为字典格式
        if not data:
            return {"error": "数据集为空"}

        column_names = set()
        for item in data:
            column_names.update(item.keys())

        column_data = {col: [] for col in column_names}
        for item in data:
            for col in column_names:
                column_data[col].append(item.get(col))

        data = column_data

    total_samples = len(data.get(text_column, data.get(label_column, [])))

    if total_samples == 0:
        return {"error": "无法获取有效数据"}

    # 1. 缺失值分析
    missing_analysis = _analyze_missing_values(data, total_samples)

    # 2. 重复值分析
    duplicate_analysis = _analyze_duplicates(data, text_column, label_column)

    # 3. 一致性检查（文本 - 标签匹配）
    consistency_check = _check_consistency(data, text_column, label_column)

    # 4. 异常检测
    anomaly_detection = _detect_anomalies(data, text_column, label_column)

    # 5. 计算质量评分
    quality_score = _calculate_quality_score(
        missing_analysis, duplicate_analysis, consistency_check, anomaly_detection, total_samples
    )

    # 6. 生成建议
    suggestions = _generate_quality_suggestions(
        missing_analysis, duplicate_analysis, consistency_check, anomaly_detection
    )

    result = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "missing_values": missing_analysis,
        "duplicates": duplicate_analysis,
        "consistency": consistency_check,
        "anomalies": anomaly_detection,
        "quality_score": quality_score,
        "suggestions": suggestions,
    }

    logger.info(f"数据质量检查完成：{total_samples} 样本，质量评分 {quality_score:.2f}")
    return result


def _analyze_missing_values(data: Dict[str, List[Any]], total_samples: int) -> Dict[str, Any]:
    """分析缺失值"""
    missing_info = {}
    total_missing = 0

    for col, values in data.items():
        missing_count = sum(1 for v in values if v is None or (isinstance(v, str) and v.strip() == ""))
        missing_pct = round(missing_count / total_samples * 100, 2) if total_samples > 0 else 0
        missing_info[col] = {
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
        }
        total_missing += missing_count

    return {
        "by_column": missing_info,
        "total_missing_samples": total_missing,
        "missing_rate": round(total_missing / (total_samples * len(data)) * 100, 2) if len(data) > 0 else 0,
    }


def _analyze_duplicates(data: Dict[str, List[Any]], text_column: str, label_column: str) -> Dict[str, Any]:
    """分析重复值"""
    # 检查完全重复的行
    seen = set()
    duplicate_indices = []

    for i in range(len(data.get(text_column, []))):
        text = data.get(text_column, [])[i] if i < len(data.get(text_column, [])) else None
        label = data.get(label_column, [])[i] if i < len(data.get(label_column, [])) else None

        key = (str(text), str(label))
        if key in seen:
            duplicate_indices.append(i)
        else:
            seen.add(key)

    total_duplicates = len(duplicate_indices)
    duplicate_rate = round(total_duplicates / len(data.get(text_column, [])) * 100, 2) if data.get(text_column) else 0

    return {
        "duplicate_count": total_duplicates,
        "duplicate_rate": duplicate_rate,
        "duplicate_indices": duplicate_indices[:20],  # 只返回前 20 个
    }


def _check_consistency(data: Dict[str, List[Any]], text_column: str, label_column: str) -> Dict[str, Any]:
    """检查文本 - 标签一致性"""
    inconsistencies = []

    texts = data.get(text_column, [])
    labels = data.get(label_column, [])

    # 检查是否有文本但无标签，或有标签但无文本
    for i in range(min(len(texts), len(labels))):
        text_empty = texts[i] is None or (isinstance(texts[i], str) and texts[i].strip() == "")
        label_empty = labels[i] is None or (isinstance(labels[i], str) and labels[i].strip() == "")

        if text_empty and not label_empty:
            inconsistencies.append({"index": i, "issue": "有标签无文本"})
        elif label_empty and not text_empty:
            inconsistencies.append({"index": i, "issue": "有文本无标签"})

    return {
        "inconsistency_count": len(inconsistencies),
        "inconsistency_rate": round(len(inconsistencies) / min(len(texts), len(labels)) * 100, 2) if min(len(texts), len(labels)) > 0 else 0,
        "inconsistencies": inconsistencies[:20],
    }


def _detect_anomalies(data: Dict[str, List[Any]], text_column: str, label_column: str) -> Dict[str, Any]:
    """检测异常值"""
    anomalies = {
        "empty_texts": [],
        "empty_labels": [],
        "unusual_lengths": [],
        "special_characters": [],
    }

    texts = data.get(text_column, [])

    for i, text in enumerate(texts):
        if text is None:
            anomalies["empty_texts"].append(i)
            continue

        if not isinstance(text, str):
            text = str(text)

        # 空文本
        if text.strip() == "":
            anomalies["empty_texts"].append(i)

        # 异常长度（过长）
        if len(text) > 10000:
            anomalies["unusual_lengths"].append({"index": i, "length": len(text)})

        # 特殊字符检测
        if any(c in text for c in ["\x00", "\ufffd"]):
            anomalies["special_characters"].append(i)

    # 检查标签
    labels = data.get(label_column, [])
    for i, label in enumerate(labels):
        if label is None or (isinstance(label, str) and label.strip() == ""):
            anomalies["empty_labels"].append(i)

    return {
        "empty_text_count": len(anomalies["empty_texts"]),
        "empty_label_count": len(anomalies["empty_labels"]),
        "unusual_length_count": len(anomalies["unusual_lengths"]),
        "special_char_count": len(anomalies["special_characters"]),
        "details": anomalies,
    }


def _calculate_quality_score(missing: Dict, duplicates: Dict, consistency: Dict, anomalies: Dict, total: int) -> float:
    """计算整体质量评分（0-100）"""
    if total == 0:
        return 0.0

    # 各维度权重
    weights = {
        "missing": 0.3,
        "duplicates": 0.2,
        "consistency": 0.3,
        "anomalies": 0.2,
    }

    # 各维度得分
    missing_score = 100 - missing.get("missing_rate", 0)
    duplicate_score = 100 - duplicates.get("duplicate_rate", 0)
    consistency_score = 100 - consistency.get("inconsistency_rate", 0)

    anomaly_penalty = min(50, (anomalies.get("empty_text_count", 0) +
                               anomalies.get("empty_label_count", 0) +
                               anomalies.get("unusual_length_count", 0) +
                               anomalies.get("special_char_count", 0)) / total * 1000)
    anomaly_score = 100 - anomaly_penalty

    # 加权平均
    total_score = (
        weights["missing"] * missing_score +
        weights["duplicates"] * duplicate_score +
        weights["consistency"] * consistency_score +
        weights["anomalies"] * anomaly_score
    )

    return round(max(0, min(100, total_score)), 2)


def _generate_quality_suggestions(missing: Dict, duplicates: Dict, consistency: Dict, anomalies: Dict) -> List[str]:
    """生成质量优化建议"""
    suggestions = []

    # 缺失值建议
    if missing.get("missing_rate", 0) > 5:
        suggestions.append(f"缺失值比例较高 ({missing['missing_rate']}%)，建议进行缺失值处理或删除相关字段")

    # 重复值建议
    if duplicates.get("duplicate_rate", 0) > 1:
        suggestions.append(f"发现 {duplicates['duplicate_count']} 个重复样本，建议去重处理")

    # 一致性建议
    if consistency.get("inconsistency_count", 0) > 0:
        suggestions.append(f"发现 {consistency['inconsistency_count']} 个文本 - 标签不一致的样本，建议检查数据标注质量")

    # 异常值建议
    if anomalies.get("empty_text_count", 0) > 0:
        suggestions.append(f"发现 {anomalies['empty_text_count']} 个空文本，建议删除或补充")

    if anomalies.get("empty_label_count", 0) > 0:
        suggestions.append(f"发现 {anomalies['empty_label_count']} 个空标签，建议删除或补充")

    if anomalies.get("unusual_length_count", 0) > 0:
        suggestions.append(f"发现 {anomalies['unusual_length_count']} 个异常长文本，建议检查是否为数据错误")

    if not suggestions:
        suggestions.append("数据质量良好，无明显问题")

    return suggestions


def generate_quality_report(quality_result: Dict[str, Any]) -> str:
    """
    生成数据质量报告文本

    Args:
        quality_result: check_data_quality 返回的结果

    Returns:
        格式化的报告文本
    """
    if "error" in quality_result:
        return f"分析失败：{quality_result['error']}"

    lines = [
        f"=== {quality_result['dataset_name']} 数据质量检查报告 ===",
        "",
        f"总样本数：{quality_result['total_samples']}",
        f"质量评分：{quality_result['quality_score']}/100",
        "",
        "--- 缺失值分析 ---",
        f"  总体缺失率：{quality_result['missing_values']['missing_rate']}%",
    ]

    for col, info in quality_result["missing_values"]["by_column"].items():
        lines.append(f"  {col}: {info['missing_count']} 缺失 ({info['missing_percentage']}%)")

    lines.extend([
        "",
        "--- 重复值分析 ---",
        f"  重复样本数：{quality_result['duplicates']['duplicate_count']}",
        f"  重复率：{quality_result['duplicates']['duplicate_rate']}%",
        "",
        "--- 一致性检查 ---",
        f"  不一致样本数：{quality_result['consistency']['inconsistency_count']}",
        f"  不一致率：{quality_result['consistency']['inconsistency_rate']}%",
        "",
        "--- 异常检测 ---",
        f"  空文本数：{quality_result['anomalies']['empty_text_count']}",
        f"  空标签数：{quality_result['anomalies']['empty_label_count']}",
        f"  异常长文本数：{quality_result['anomalies']['unusual_length_count']}",
        f"  含特殊字符数：{quality_result['anomalies']['special_char_count']}",
        "",
        "--- 优化建议 ---",
    ])

    for i, suggestion in enumerate(quality_result["suggestions"], 1):
        lines.append(f"  {i}. {suggestion}")

    return "\n".join(lines)
