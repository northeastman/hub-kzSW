"""
SubAgents 实现
每个 SubAgent 负责特定的分析任务，使用 OTAT 循环执行
"""
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from ..core.otat_loop import OTATLoop
from ..core.agent_state import AgentState

logger = logging.getLogger(__name__)


class BaseSubAgent(ABC):
    """SubAgent 基类"""

    def __init__(self, llm_client: Any, task_name: str, max_steps: int = 10, verbose: bool = False):
        """
        初始化 SubAgent

        Args:
            llm_client: LLM 客户端实例
            task_name: 任务名称
            max_steps: 最大执行步数
            verbose: 是否输出详细日志
        """
        self.llm_client = llm_client
        self.task_name = task_name
        self.max_steps = max_steps
        self.verbose = verbose
        self.system_prompt = self._build_system_prompt()

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        pass

    @abstractmethod
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分析任务

        Args:
            data: 输入数据

        Returns:
            分析结果
        """
        pass

    def _create_otat_loop(self, task_description: str) -> OTATLoop:
        """创建 OTAT 循环实例"""
        return OTATLoop(
            llm_client=self.llm_client,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            verbose=self.verbose,
        )


class LabelAnalysisAgent(BaseSubAgent):
    """标签分布分析 SubAgent"""

    def __init__(self, llm_client: Any, max_steps: int = 8, verbose: bool = False):
        super().__init__(llm_client, "标签分布分析", max_steps, verbose)

    def _build_system_prompt(self) -> str:
        return """你是一个数据分析助手，专门负责标签分布分析。

可用工具：
- analyze_label_distribution: 分析标签分布情况

工作流程：
1. 观察输入数据中的标签列
2. 调用 analyze_label_distribution 工具进行分析
3. 检查分析结果是否完整
4. 输出标签分布报告

请严格按照 OTAT 循环格式输出：
Thought: [你的推理]
Action: analyze_label_distribution
Action Input: {"labels": [...], "dataset_name": "..."}
Check: [验证结果]
"""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行标签分布分析"""
        from ..tools.label_analysis import analyze_label_distribution

        labels = data.get("labels", [])
        label_names = data.get("label_names", {})
        dataset_name = data.get("dataset_name", "数据集")

        if not labels:
            return {"error": "未提供标签数据"}

        try:
            result = analyze_label_distribution(
                labels=labels,
                label_names=label_names,
                dataset_name=dataset_name,
            )
            return {
                "task": "label_distribution",
                "status": "success",
                "result": result,
            }
        except Exception as e:
            logger.error(f"标签分布分析失败：{e}")
            return {
                "task": "label_distribution",
                "status": "error",
                "error": str(e),
            }


class TextLengthAgent(BaseSubAgent):
    """文本长度分析 SubAgent"""

    def __init__(self, llm_client: Any, max_steps: int = 8, verbose: bool = False):
        super().__init__(llm_client, "文本长度分析", max_steps, verbose)

    def _build_system_prompt(self) -> str:
        return """你是一个数据分析助手，专门负责文本长度分析。

可用工具：
- analyze_text_length: 分析文本长度分布

工作流程：
1. 观察输入数据中的文本列
2. 调用 analyze_text_length 工具进行分析
3. 检查分析结果是否完整
4. 输出文本长度分布报告

请严格按照 OTAT 循环格式输出：
Thought: [你的推理]
Action: analyze_text_length
Action Input: {"texts": [...], "dataset_name": "..."}
Check: [验证结果]
"""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行文本长度分析"""
        from ..tools.text_analysis import analyze_text_length

        texts = data.get("texts", [])
        dataset_name = data.get("dataset_name", "数据集")
        length_unit = data.get("length_unit", "char")

        if not texts:
            return {"error": "未提供文本数据"}

        try:
            result = analyze_text_length(
                texts=texts,
                dataset_name=dataset_name,
                length_unit=length_unit,
            )
            return {
                "task": "text_length",
                "status": "success",
                "result": result,
            }
        except Exception as e:
            logger.error(f"文本长度分析失败：{e}")
            return {
                "task": "text_length",
                "status": "error",
                "error": str(e),
            }


class DataQualityAgent(BaseSubAgent):
    """数据质量检查 SubAgent"""

    def __init__(self, llm_client: Any, max_steps: int = 8, verbose: bool = False):
        super().__init__(llm_client, "数据质量检查", max_steps, verbose)

    def _build_system_prompt(self) -> str:
        return """你是一个数据分析助手，专门负责数据质量检查。

可用工具：
- check_data_quality: 检查数据质量

工作流程：
1. 观察输入数据的结构
2. 调用 check_data_quality 工具进行检查
3. 检查结果是否识别出所有质量问题
4. 输出数据质量报告

请严格按照 OTAT 循环格式输出：
Thought: [你的推理]
Action: check_data_quality
Action Input: {"data": [...], "dataset_name": "..."}
Check: [验证结果]
"""

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据质量检查"""
        from ..tools.quality_check import check_data_quality

        dataset = data.get("data", [])
        dataset_name = data.get("dataset_name", "数据集")
        text_column = data.get("text_column", "text")
        label_column = data.get("label_column", "label")

        if not dataset:
            return {"error": "未提供数据集数据"}

        try:
            result = check_data_quality(
                data=dataset,
                dataset_name=dataset_name,
                text_column=text_column,
                label_column=label_column,
            )
            return {
                "task": "quality_check",
                "status": "success",
                "result": result,
            }
        except Exception as e:
            logger.error(f"数据质量检查失败：{e}")
            return {
                "task": "quality_check",
                "status": "error",
                "error": str(e),
            }
