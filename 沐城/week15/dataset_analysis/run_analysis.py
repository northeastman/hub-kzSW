#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集分析 Agent 示例脚本

使用示例:
    python -m dataset_analysis.run_analysis --dataset my_data.xlsx

功能:
1. 加载数据集
2. 使用主 Agent + SubAgents 架构并行执行分析
3. 生成分析报告和统计图表
"""

import argparse
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset_from_excel(file_path: str) -> dict:
    """从 Excel 文件加载数据集"""
    try:
        import pandas as pd

        df = pd.read_excel(file_path)
        logger.info(f"成功加载数据集：{file_path}, 共 {len(df)} 条记录")

        # 假设列名为 'text' 和 'label'
        texts = df['text'].fillna('').astype(str).tolist()
        labels = df['label'].fillna('').astype(str).tolist()

        return {
            "dataset_name": Path(file_path).stem,
            "texts": texts,
            "labels": labels,
            "data": df.to_dict('records'),
        }
    except Exception as e:
        logger.error(f"加载数据集失败：{e}")
        raise


def load_dataset_from_json(file_path: str) -> dict:
    """从 JSON 文件加载数据集"""
    import json

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"成功加载数据集：{file_path}")

    if isinstance(data, list):
        texts = [item.get('text', '') for item in data]
        labels = [item.get('label', '') for item in data]
        return {
            "dataset_name": Path(file_path).stem,
            "texts": texts,
            "labels": labels,
            "data": data,
        }
    elif isinstance(data, dict):
        return {
            "dataset_name": Path(file_path).stem,
            "texts": data.get("texts", []),
            "labels": data.get("labels", []),
            "data": data,
        }
    else:
        raise ValueError("不支持的数据格式")


def run_analysis(dataset: dict, verbose: bool = True):
    """执行数据集分析"""
    from dataset_analysis.llm import init_llm
    from dataset_analysis.agents.coordinator import CoordinatorAgent
    from dataset_analysis.report.report_writer import ReportWriter

    # 初始化 LLM
    logger.info("正在初始化 LLM 客户端...")
    llm = init_llm()
    if llm is None:
        logger.error("LLM 初始化失败，请检查配置")
        return None

    # 创建主 Agent
    coordinator = CoordinatorAgent(
        llm_client=llm,
        max_steps=8,
        max_workers=3,
        verbose=verbose,
    )

    # 执行分析
    logger.info("开始执行数据集分析...")
    results = coordinator.execute(dataset)

    if results["status"] != "completed":
        logger.error("分析任务未能完成")
        return None

    # 生成报告
    logger.info("正在生成分析报告...")
    writer = ReportWriter(output_dir="output/analysis_report")

    # 生成 Markdown 报告
    md_report = writer.generate_report(
        analysis_results=results["results"],
        dataset_name=dataset["dataset_name"],
        output_format="markdown",
        include_charts=True,
    )
    logger.info(f"Markdown 报告已生成：{md_report}")

    # 生成 HTML 报告
    html_report = writer.generate_report(
        analysis_results=results["results"],
        dataset_name=dataset["dataset_name"],
        output_format="html",
        include_charts=True,
    )
    logger.info(f"HTML 报告已生成：{html_report}")

    # 输出优化建议
    suggestions = writer.generate_optimization_suggestions(results["results"])
    if suggestions:
        logger.info("\n=== 优化建议 ===")
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"{i}. {suggestion}")

    return results


def main():
    parser = argparse.ArgumentParser(description="数据集分析 Agent 示例脚本")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集文件路径（支持 .xlsx, .json 格式）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["excel", "json"],
        default="excel",
        help="数据集格式",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不输出详细日志",
    )

    args = parser.parse_args()

    # 加载数据集
    if args.format == "excel":
        dataset = load_dataset_from_excel(args.dataset)
    else:
        dataset = load_dataset_from_json(args.dataset)

    # 执行分析
    results = run_analysis(dataset, verbose=not args.quiet)

    if results:
        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        print(f"数据集：{dataset['dataset_name']}")
        print(f"样本数：{len(dataset.get('texts', dataset.get('labels', [])))}")
        print(f"报告输出目录：output/analysis_report/")
    else:
        print("\n分析失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
