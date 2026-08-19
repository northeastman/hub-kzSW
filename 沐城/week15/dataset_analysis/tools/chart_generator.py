"""
统计图表生成工具
生成各类数据分析可视化图表
"""
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 延迟导入 matplotlib，避免不必要的依赖加载
plt = None
sns = None


def _init_matplotlib():
    """延迟初始化 matplotlib"""
    global plt, sns
    if plt is None:
        try:
            import matplotlib.pyplot as plt_module
            import seaborn as sns_module
            plt = plt_module
            sns = sns_module
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError as e:
            logger.exception(f"无法导入 matplotlib/seaborn: {e}")
            raise


def generate_chart(
    chart_type: str,
    data: Dict[str, Any],
    output_path: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (10, 6),
    **kwargs
) -> str:
    """
    生成统计图表

    Args:
        chart_type: 图表类型 ('bar', 'histogram', 'pie', 'box', 'line')
        data: 数据字典
        output_path: 输出文件路径
        title: 图表标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        figsize: 图表大小
        **kwargs: 其他参数

    Returns:
        输出文件路径
    """
    _init_matplotlib()

    try:
        fig, ax = plt.subplots(figsize=figsize)

        if chart_type == "bar":
            _generate_bar_chart(ax, data, **kwargs)
        elif chart_type == "histogram":
            _generate_histogram(ax, data, **kwargs)
        elif chart_type == "pie":
            _generate_pie_chart(ax, data, **kwargs)
        elif chart_type == "box":
            _generate_box_plot(ax, data, **kwargs)
        elif chart_type == "line":
            _generate_line_chart(ax, data, **kwargs)
        else:
            raise ValueError(f"不支持的图表类型：{chart_type}")

        # 设置标题和标签
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # 添加图例（如果有）
        if kwargs.get("labels"):
            ax.legend()

        # 自动调整布局
        plt.tight_layout()

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存图表
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"图表已保存到：{output_path}")
        return output_path

    except Exception as e:
        logger.error(f"生成图表失败：{e}")
        raise


def _generate_bar_chart(ax, data: Dict, **kwargs):
    """生成柱状图（用于标签分布）"""
    labels = data.get("labels", [])
    values = data.get("values", [])
    colors = kwargs.get("colors", None)

    x_pos = range(len(labels))
    ax.bar(x_pos, values, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 在柱子上方添加数值标签
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom', fontsize=9)


def _generate_histogram(ax, data: Dict, **kwargs):
    """生成直方图（用于文本长度分布）"""
    values = data.get("values", [])
    bins = kwargs.get("bins", 30)
    color = kwargs.get("color", "skyblue")
    edge_color = kwargs.get("edge_color", "black")

    ax.hist(values, bins=bins, color=color, edgecolor=edge_color, alpha=0.7)


def _generate_pie_chart(ax, data: Dict, **kwargs):
    """生成饼图（用于类别占比）"""
    labels = data.get("labels", [])
    values = data.get("values", [])
    colors = kwargs.get("colors", None)

    if colors is None:
        colors = plt.cm.Set3(range(len(labels)))

    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)


def _generate_box_plot(ax, data: Dict, **kwargs):
    """生成箱线图（用于异常值检测）"""
    values = data.get("values", [])
    labels = data.get("labels", ["数据"])

    bp = ax.boxplot(values, labels=labels, patch_artist=True)

    # 设置颜色
    colors = kwargs.get("colors", ["lightblue"])
    for patch, color in zip(bp['boxes'], colors * len(values)):
        patch.set_facecolor(color)


def _generate_line_chart(ax, data: Dict, **kwargs):
    """生成折线图"""
    x_values = data.get("x", [])
    y_values = data.get("y", [])
    label = kwargs.get("label", "")
    color = kwargs.get("color", "blue")
    marker = kwargs.get("marker", "o")
    linestyle = kwargs.get("linestyle", "-")

    ax.plot(x_values, y_values, label=label, color=color, marker=marker, linestyle=linestyle)


def generate_label_distribution_chart(
    analysis_result: Dict[str, Any],
    output_path: str,
    top_n: int = 10
) -> str:
    """
    生成标签分布柱状图

    Args:
        analysis_result: analyze_label_distribution 返回的结果
        output_path: 输出文件路径
        top_n: 显示前 N 个类别

    Returns:
        输出文件路径
    """
    if "error" in analysis_result:
        raise ValueError(analysis_result["error"])

    distribution = analysis_result["distribution"][:top_n]
    labels = [item["name"] for item in distribution]
    values = [item["count"] for item in distribution]

    data = {"labels": labels, "values": values}

    return generate_chart(
        chart_type="bar",
        data=data,
        output_path=output_path,
        title=f"{analysis_result['dataset_name']} - 标签分布 (Top {top_n})",
        xlabel="类别",
        ylabel="样本数",
        figsize=(max(10, len(labels) * 0.8), 6),
        colors=plt.cm.Set3(range(len(labels))) if plt else None,
    )


def generate_text_length_histogram(
    analysis_result: Dict[str, Any],
    output_path: str,
) -> str:
    """
    生成文本长度分布直方图

    Args:
        analysis_result: analyze_text_length 返回的结果
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    if "error" in analysis_result:
        raise ValueError(analysis_result["error"])

    distribution = analysis_result.get("distribution", [])
    if not distribution:
        # 如果没有分布数据，使用分位数生成简化的直方图
        percentiles = analysis_result.get("percentiles", {})
        data = {"values": [percentiles.get(f"p{p}", 0) for p in [10, 25, 50, 75, 90, 95, 99]]}
        return generate_chart(
            chart_type="bar",
            data=data,
            output_path=output_path,
            title=f"{analysis_result['dataset_name']} - 文本长度分位数",
            xlabel="分位数",
            ylabel="长度",
        )

    # 使用分布数据生成直方图
    x_values = [d["bin_start"] for d in distribution]
    y_values = [d["count"] for d in distribution]

    data = {"values": y_values}

    return generate_chart(
        chart_type="histogram",
        data=data,
        output_path=output_path,
        title=f"{analysis_result['dataset_name']} - 文本长度分布",
        xlabel="长度",
        ylabel="频数",
        bins=len(distribution),
    )


def generate_quality_boxplot(
    quality_result: Dict[str, Any],
    output_path: str,
) -> str:
    """
    生成数据质量箱线图

    Args:
        quality_result: check_data_quality 返回的结果
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    if "error" in quality_result:
        raise ValueError(quality_result["error"])

    anomalies = quality_result.get("anomalies", {})
    details = anomalies.get("details", {})

    # 准备箱线图数据
    values = [
        len(details.get("empty_texts", [])),
        len(details.get("empty_labels", [])),
        len(details.get("unusual_lengths", [])),
        len(details.get("special_characters", [])),
    ]
    labels = ["空文本", "空标签", "异常长度", "特殊字符"]

    data = {"values": [values], "labels": labels}

    return generate_chart(
        chart_type="box",
        data=data,
        output_path=output_path,
        title=f"{quality_result['dataset_name']} - 数据质量异常分布",
        ylabel="样本数",
    )


def generate_summary_chart(
    analysis_results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> str:
    """
    生成综合分析摘要图表

    Args:
        analysis_results: 包含所有分析结果的字典
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    _init_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 标签分布饼图
    if "label_distribution" in analysis_results:
        label_result = analysis_results["label_distribution"]
        distribution = label_result.get("distribution", [])[:8]  # 最多 8 个类别

        if distribution:
            labels = [item["name"] for item in distribution]
            values = [item["percentage"] for item in distribution]

            axes[0, 0].pie(values, labels=labels, autopct='%1.1f%%')
            axes[0, 0].set_title("标签分布占比")

    # 2. 文本长度直方图
    if "text_length" in analysis_results:
        text_result = analysis_results["text_length"]
        distribution = text_result.get("distribution", [])

        if distribution:
            bin_starts = [d["bin_start"] for d in distribution]
            counts = [d["count"] for d in distribution]

            axes[0, 1].bar(range(len(counts)), counts)
            axes[0, 1].set_title("文本长度分布")
            axes[0, 1].set_xlabel("长度区间")
            axes[0, 1].set_ylabel("样本数")

    # 3. 质量评分仪表盘
    if "quality_check" in analysis_results:
        quality_result = analysis_results["quality_check"]
        score = quality_result.get("quality_score", 0)

        # 简单的仪表盘表示
        categories = ['缺失值', '重复值', '一致性', '异常值']
        scores = [
            100 - quality_result.get("missing_values", {}).get("missing_rate", 0),
            100 - quality_result.get("duplicates", {}).get("duplicate_rate", 0),
            100 - quality_result.get("consistency", {}).get("inconsistency_rate", 0),
            score,
        ]

        axes[1, 0].barh(categories, scores, color=['green', 'blue', 'orange', 'red'])
        axes[1, 0].set_title("数据质量评分")
        axes[1, 0].set_xlim(0, 100)

    # 4. 综合指标雷达图
    axes[1, 1].text(0.5, 0.5, f"综合质量评分\n{score:.1f}/100",
                   ha='center', va='center', fontsize=20, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"综合摘要图表已保存到：{output_path}")
    return output_path
