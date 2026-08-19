"""
报告生成器
负责生成完整的分析报告，包括统计图表和文本报告
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReportWriter:
    """
    报告生成器

    功能:
    1. 生成 Markdown 格式分析报告
    2. 生成 HTML 格式分析报告
    3. 生成统计图表
    4. 整合所有分析结果
    """

    def __init__(self, output_dir: str = "output/analysis_report"):
        """
        初始化报告生成器

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录：{self.output_dir}")

    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        dataset_name: str,
        output_format: str = "markdown",
        include_charts: bool = True,
    ) -> str:
        """
        生成完整分析报告

        Args:
            analysis_results: 分析结果字典
            dataset_name: 数据集名称
            output_format: 输出格式 ('markdown' 或 'html')
            include_charts: 是否包含图表

        Returns:
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成报告内容
        if output_format == "markdown":
            content = self._generate_markdown_report(analysis_results, dataset_name)
            filename = f"{dataset_name}_analysis_{timestamp}.md"
        else:
            content = self._generate_html_report(analysis_results, dataset_name, include_charts)
            filename = f"{dataset_name}_analysis_{timestamp}.html"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"报告已保存到：{filepath}")
        return filepath

    def _generate_markdown_report(
        self,
        analysis_results: Dict[str, Any],
        dataset_name: str
    ) -> str:
        """生成 Markdown 格式报告"""
        from ..tools.summary_generator import generate_summary

        return generate_summary(analysis_results, dataset_name, output_format="markdown")

    def _generate_html_report(
        self,
        analysis_results: Dict[str, Any],
        dataset_name: str,
        include_charts: bool = True,
    ) -> str:
        """生成 HTML 格式报告"""
        from ..tools.summary_generator import generate_summary

        markdown_content = generate_summary(analysis_results, dataset_name, output_format="markdown")

        # 简单的 Markdown 到 HTML 转换
        html_content = self._markdown_to_html(markdown_content, dataset_name)

        if include_charts:
            # 生成图表并添加到 HTML
            chart_html = self._generate_chart_section(analysis_results, dataset_name)
            html_content = html_content.replace("</body>", f"{chart_html}</body>")

        return html_content

    def _markdown_to_html(self, markdown_text: str, dataset_name: str = "数据集") -> str:
        """将 Markdown 转换为 HTML"""
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"    <title>{dataset_name} - 数据分析报告</title>",
            "    <style>",
            "        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }",
            "        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
            "        h2 { color: #34495e; margin-top: 30px; }",
            "        h3 { color: #7f8c8d; }",
            "        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }",
            "        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
            "        th { background-color: #3498db; color: white; }",
            "        tr:nth-child(even) { background-color: #f2f2f2; }",
            "        ul { line-height: 1.8; }",
            "        .summary { background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }",
            "        .warning { background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }",
            "        .success { background: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745; }",
            "        .chart-container { margin: 20px 0; text-align: center; }",
            "    </style>",
            "</head>",
            "<body>",
        ]

        in_table = False
        for line in markdown_text.split("\n"):
            line = line.strip()

            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("**") and line.endswith("**"):
                html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
            elif line.startswith("- ["):
                # 列表项
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("|") and not line.startswith("|---"):
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True
                cells = line.strip("|").split("|")
                if all(c.strip().startswith("-") or c.strip().isalpha() for c in cells):
                    html_lines.append("<tr>" + "".join(f"<th>{c.strip()}</th>" for c in cells) + "</tr>")
                else:
                    html_lines.append("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in cells) + "</tr>")
            elif line.startswith("|---"):
                continue  # 跳过表格分隔行
            elif in_table and line.startswith("|"):
                continue
            elif in_table:
                html_lines.append("</table>")
                in_table = False
                if line:
                    html_lines.append(f"<p>{line}</p>")
            elif line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                html_lines.append(f"<li>{line}</li>")
            elif line:
                html_lines.append(f"<p>{line}</p>")

        if in_table:
            html_lines.append("</table>")

        html_lines.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(html_lines)

    def _generate_chart_section(
        self,
        analysis_results: Dict[str, Any],
        dataset_name: str,
    ) -> str:
        """生成图表部分 HTML"""
        html_lines = ["<div class='chart-container'>", "<h2>分析图表</h2>"]

        # 生成各类图表
        chart_files = []

        # 解包 SubAgent 返回的结果格式
        unwrapped_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict) and "result" in value:
                unwrapped_results[key] = value["result"]
            else:
                unwrapped_results[key] = value

        try:
            from ..tools.chart_generator import (
                generate_label_distribution_chart,
                generate_text_length_histogram,
                generate_quality_boxplot,
                generate_summary_chart,
            )

            # 标签分布图
            if "label_distribution" in unwrapped_results:
                try:
                    chart_path = generate_label_distribution_chart(
                        unwrapped_results["label_distribution"],
                        os.path.join(self.output_dir, f"{dataset_name}_label_dist.png"),
                    )
                    chart_files.append(("标签分布", chart_path))
                except Exception as e:
                    logger.warning(f"生成标签分布图失败：{e}")

            # 文本长度直方图
            if "text_length" in unwrapped_results:
                try:
                    chart_path = generate_text_length_histogram(
                        unwrapped_results["text_length"],
                        os.path.join(self.output_dir, f"{dataset_name}_text_len.png"),
                    )
                    chart_files.append(("文本长度分布", chart_path))
                except Exception as e:
                    logger.warning(f"生成文本长度图失败：{e}")

            # 质量评分图
            if "quality_check" in unwrapped_results:
                try:
                    chart_path = generate_summary_chart(
                        unwrapped_results,
                        os.path.join(self.output_dir, f"{dataset_name}_summary.png"),
                    )
                    chart_files.append(("综合分析", chart_path))
                except Exception as e:
                    logger.warning(f"生成综合图失败：{e}")

        except ImportError as e:
            logger.warning(f"无法导入图表生成模块：{e}")

        # 添加图表到 HTML
        for title, path in chart_files:
            if os.path.exists(path):
                html_lines.append(f"<h3>{title}</h3>")
                html_lines.append(f"<img src='{path}' alt='{title}' style='max-width: 100%;'>")

        html_lines.append("</div>")

        return "\n".join(html_lines)

    def generate_optimization_suggestions(
        self,
        analysis_results: Dict[str, Any],
    ) -> List[str]:
        """
        基于分析结果生成优化建议

        Args:
            analysis_results: 分析结果字典

        Returns:
            建议列表
        """
        suggestions = []

        # 基于标签分布的建议
        if "label_distribution" in analysis_results:
            label_result = analysis_results["label_distribution"]
            assessment = label_result.get("assessment", "均衡")

            if assessment == "严重不均衡":
                suggestions.append("【标签分布】数据严重不均衡，建议：1) 对少数类进行过采样；2) 使用类别权重；3) 收集更多少数类样本")
            elif assessment == "中度不均衡":
                suggestions.append("【标签分布】数据中度不均衡，建议在模型训练时考虑类别权重")

        # 基于文本长度的建议
        if "text_length" in analysis_results:
            text_result = analysis_results["text_length"]
            empty_count = text_result.get("empty_count", 0)

            if empty_count > 0:
                suggestions.append(f"【文本质量】发现{empty_count}个空文本，建议清理或补充")

            outlier_pct = text_result.get("outliers", {}).get("outlier_percentage", 0)
            if outlier_pct > 5:
                suggestions.append("【文本质量】异常长度文本比例较高，建议检查数据质量")

        # 基于数据质量的建议
        if "quality_check" in analysis_results:
            quality_result = analysis_results["quality_check"]
            quality_score = quality_result.get("quality_score", 100)

            if quality_score < 60:
                suggestions.append(f"【整体质量】质量评分较低 ({quality_score}分)，建议进行全面数据清洗")
            elif quality_score < 80:
                suggestions.append(f"【整体质量】质量评分中等 ({quality_score}分)，建议进行部分数据清洗")

            duplicate_rate = quality_result.get("duplicates", {}).get("duplicate_rate", 0)
            if duplicate_rate > 1:
                suggestions.append(f"【重复数据】发现重复样本 (重复率{duplicate_rate}%)，建议去重处理")

        return suggestions
