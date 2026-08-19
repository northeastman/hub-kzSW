"""
集成测试 - 测试主 Agent 和 SubAgents 的协作
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset_analysis.agents.coordinator import CoordinatorAgent
from dataset_analysis.agents.subagents import LabelAnalysisAgent, TextLengthAgent, DataQualityAgent
from dataset_analysis.tools.summary_generator import generate_summary, generate_optimization_suggestions


class MockLLMClient:
    """模拟 LLM 客户端用于测试"""
    def invoke(self, prompt, content):
        class Response:
            content = "Thought: 分析完成\nFinal Answer: 分析结果正常"
        return Response()


class TestCoordinatorAgent(unittest.TestCase):
    """测试主 Agent 协调器"""

    def test_task_decomposition(self):
        """测试任务分解功能"""
        llm = MockLLMClient()
        coordinator = CoordinatorAgent(llm, verbose=False)

        tasks = coordinator.decompose_task("分析数据集")

        self.assertEqual(len(tasks), 3)
        self.assertIn(tasks[0]["name"], ["label_distribution", "text_length", "quality_check"])

    def test_subagents_execution(self):
        """测试 SubAgents 执行功能"""
        llm = MockLLMClient()

        # 测试标签分析 SubAgent
        label_agent = LabelAnalysisAgent(llm, verbose=False)
        result = label_agent.execute({
            "labels": ["A"] * 50 + ["B"] * 50,
            "dataset_name": "测试集",
        })
        self.assertEqual(result["status"], "success")

        # 测试文本长度 SubAgent
        text_agent = TextLengthAgent(llm, verbose=False)
        result = text_agent.execute({
            "texts": ["测试文本"] * 100,
            "dataset_name": "测试集",
        })
        self.assertEqual(result["status"], "success")

        # 测试数据质量 SubAgent
        quality_agent = DataQualityAgent(llm, verbose=False)
        result = quality_agent.execute({
            "data": [{"text": f"文本{i}", "label": "A"} for i in range(100)],
            "dataset_name": "测试集",
        })
        self.assertEqual(result["status"], "success")

    def test_report_generation(self):
        """测试报告生成功能"""
        # 准备模拟数据
        labels = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
        texts = ["测试文本"] * 100

        # 直接调用工具函数（绕过 LLM）
        from dataset_analysis.tools.label_analysis import analyze_label_distribution
        from dataset_analysis.tools.text_analysis import analyze_text_length
        from dataset_analysis.tools.quality_check import check_data_quality

        analysis_results = {
            "label_distribution": analyze_label_distribution(labels, dataset_name="测试集"),
            "text_length": analyze_text_length(texts, dataset_name="测试集"),
            "quality_check": check_data_quality(
                [{"text": t, "label": l} for t, l in zip(texts, labels)],
                dataset_name="测试集"
            ),
        }

        # 测试报告生成
        report = generate_summary(analysis_results, "测试集")
        self.assertIn("数据分析报告", report)
        self.assertIn("标签分布", report)
        self.assertIn("文本长度", report)
        self.assertIn("数据质量", report)

    def test_optimization_suggestions(self):
        """测试优化建议生成"""
        from dataset_analysis.tools.label_analysis import analyze_label_distribution

               # 创建不均衡数据
        labels = ["A"] * 90 + ["B"] * 10
        result = analyze_label_distribution(labels, dataset_name="测试集")

        analysis_results = {
            "label_distribution": result,
        }

        suggestions = generate_optimization_suggestions(analysis_results)
        self.assertTrue(len(suggestions) > 0)
        self.assertIn("不均衡", suggestions[0])


if __name__ == "__main__":
    unittest.main()
