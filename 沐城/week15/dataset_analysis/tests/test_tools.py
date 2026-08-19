"""
数据分析工具单元测试
"""
import unittest
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset_analysis.tools.label_analysis import analyze_label_distribution
from dataset_analysis.tools.text_analysis import analyze_text_length
from dataset_analysis.tools.quality_check import check_data_quality


class TestLabelAnalysis(unittest.TestCase):
    """标签分布分析测试"""

    def test_balanced_distribution(self):
        """测试均衡分布"""
        labels = ["A"] * 50 + ["B"] * 50
        result = analyze_label_distribution(labels, dataset_name="测试集")

        self.assertEqual(result["total_samples"], 100)
        self.assertEqual(result["num_classes"], 2)
        self.assertEqual(result["assessment"], "均衡")

    def test_imbalanced_distribution(self):
        """测试不均衡分布"""
        labels = ["A"] * 90 + ["B"] * 10
        result = analyze_label_distribution(labels, dataset_name="测试集")

        self.assertGreater(result["balance_metrics"]["imbalance_ratio"], 5)
        self.assertIn(result["assessment"], ["中度不均衡", "严重不均衡"])

    def test_empty_labels(self):
        """测试空标签列表"""
        result = analyze_label_distribution([], dataset_name="测试集")
        self.assertIn("error", result)


class TestTextLength(unittest.TestCase):
    """文本长度分析测试"""

    def test_normal_distribution(self):
        """测试正常分布"""
        texts = ["这是一个测试文本"] * 100
        result = analyze_text_length(texts, dataset_name="测试集")

        self.assertEqual(result["total_samples"], 100)
        self.assertEqual(result["length_stats"]["mean"], 8)  # 8 个字符

    def test_with_empty_texts(self):
        """测试包含空文本"""
        texts = ["测试文本", "", "另一个文本", None]
        result = analyze_text_length(texts, dataset_name="测试集")

        self.assertEqual(result["empty_count"], 2)

    def test_outliers(self):
        """测试异常值检测"""
        # 创建包含明显异常值的数据集
        texts = ["测试文本"] * 95 + ["这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的异常文本"] * 5
        result = analyze_text_length(texts, dataset_name="测试集")

        # 异常值检测基于 IQR，这里只验证功能可以正常运行
        self.assertIn("outlier_count", result["outliers"])


class TestQualityCheck(unittest.TestCase):
    """数据质量检查测试"""

    def test_good_quality_data(self):
        """测试高质量数据"""
        data = [
            {"text": f"文本{i}", "label": "A" if i < 50 else "B"}
            for i in range(100)
        ]
        result = check_data_quality(data, dataset_name="测试集")

        self.assertGreater(result["quality_score"], 90)
        self.assertEqual(result["duplicates"]["duplicate_count"], 0)

    def test_with_missing_values(self):
        """测试包含缺失值"""
        data = [
            {"text": "测试文本", "label": "A"},
            {"text": "", "label": "B"},
            {"text": None, "label": "A"},
        ]
        result = check_data_quality(data, dataset_name="测试集")

        self.assertGreater(result["missing_values"]["total_missing_samples"], 0)

    def test_with_duplicates(self):
        """测试包含重复值"""
        data = [
            {"text": "相同文本", "label": "A"},
            {"text": "相同文本", "label": "A"},
            {"text": "不同文本", "label": "B"},
        ]
        result = check_data_quality(data, dataset_name="测试集")

        self.assertEqual(result["duplicates"]["duplicate_count"], 1)


if __name__ == "__main__":
    unittest.main()
