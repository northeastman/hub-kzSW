"""
主 Agent (Coordinator) 实现
负责任务分解、SubAgents 协调、结果汇总和日志输出
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from ..core.agent_state import AgentState
from .subagents import BaseSubAgent, LabelAnalysisAgent, TextLengthAgent, DataQualityAgent

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    主 Agent 协调器

    职责:
    1. 接收用户分析请求
    2. 将请求分解为多个子任务
    3. 分发任务给 SubAgents 并行执行
    4. 等待所有 SubAgents 完成
    5. 汇总结果并输出报告
    """

    def __init__(
        self,
        llm_client: Any,
        max_steps: int = 8,
        max_workers: int = 3,
        timeout: int = 300,
        verbose: bool = False,
    ):
        """
        初始化主 Agent

        Args:
            llm_client: LLM 客户端实例
            max_steps: 每个 SubAgent 的最大执行步数
            max_workers: 最大并行工作线程数
            timeout: 单个 SubAgent 执行超时时间（秒）
            verbose: 是否输出详细日志
        """
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.max_workers = max_workers
        self.timeout = timeout
        self.verbose = verbose

        # 初始化 SubAgents
        self.subagents: List[BaseSubAgent] = [
            LabelAnalysisAgent(llm_client, max_steps=max_steps, verbose=verbose),
            TextLengthAgent(llm_client, max_steps=max_steps, verbose=verbose),
            DataQualityAgent(llm_client, max_steps=max_steps, verbose=verbose),
        ]

        # 任务状态追踪
        self.task_states: Dict[str, str] = {}

    def _log(self, message: str, level: str = "info"):
        """统一日志输出"""
        if self.verbose or level == "error":
            if level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
            else:
                logger.info(message)
        print(message)  # 始终打印到控制台

    def _update_task_state(self, task_name: str, state: str):
        """更新任务状态"""
        self.task_states[task_name] = state
        self._log(f"[任务状态] {task_name}: {state}")

    def decompose_task(self, user_request: str) -> List[Dict[str, Any]]:
        """
        将用户请求分解为子任务

        Args:
            user_request: 用户分析请求

        Returns:
            子任务列表
        """
        # 默认分解为三个分析任务
        tasks = [
            {
                "name": "label_distribution",
                "description": "标签分布分析",
                "agent": self.subagents[0],
                "required_data": ["labels"],
            },
            {
                "name": "text_length",
                "description": "文本长度分析",
                "agent": self.subagents[1],
                "required_data": ["texts"],
            },
            {
                "name": "quality_check",
                "description": "数据质量检查",
                "agent": self.subagents[2],
                "required_data": ["data"],
            },
        ]

        self._log(f"任务分解完成，共 {len(tasks)} 个子任务")
        return tasks

    def _execute_single_task(
        self, task: Dict[str, Any], data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        执行单个任务

        Args:
            task: 任务定义
            data: 输入数据

        Returns:
            (任务名，结果) 元组
        """
        task_name = task["name"]
        agent = task["agent"]

        try:
            self._update_task_state(task_name, "running")

            # 准备任务数据
            task_data = {
                "dataset_name": data.get("dataset_name", "数据集"),
            }

            # 根据任务类型传递相应数据
            if task_name == "label_distribution":
                task_data["labels"] = data.get("labels", [])
                task_data["label_names"] = data.get("label_names", {})
            elif task_name == "text_length":
                task_data["texts"] = data.get("texts", [])
                task_data["length_unit"] = data.get("length_unit", "char")
            elif task_name == "quality_check":
                task_data["data"] = data.get("data", [])
                task_data["text_column"] = data.get("text_column", "text")
                task_data["label_column"] = data.get("label_column", "label")

            # 执行任务
            result = agent.execute(task_data)
            self._update_task_state(task_name, "completed")

            return task_name, result

        except Exception as e:
            self._update_task_state(task_name, "failed")
            logger.error(f"任务 {task_name} 执行失败：{e}")
            return task_name, {
                "task": task_name,
                "status": "error",
                "error": str(e),
            }

    def execute(
        self,
        data: Dict[str, Any],
        user_request: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的数据集分析流程

        Args:
            data: 数据集数据，应包含:
                - labels: 标签列表
                - texts: 文本列表
                - data: 原始数据集（字典列表或列式数据）
                - dataset_name: 数据集名称
            user_request: 用户分析请求（可选）

        Returns:
            汇总的分析结果
        """
        start_time = time.time()
        self._log("=" * 60)
        self._log(f"开始执行数据集分析任务")
        self._log(f"数据集：{data.get('dataset_name', '未知')}")
        if user_request:
            self._log(f"用户请求：{user_request}")
        self._log("=" * 60)

        # 1. 任务分解
        if not user_request:
            user_request = "分析数据集的特征信息，包括标签分布、文本长度分布和数据质量"

        tasks = self.decompose_task(user_request)

        # 2. 重置任务状态
        self.task_states = {task["name"]: "pending" for task in tasks}

        # 3. 并行执行 SubAgents
        self._log("\n[执行阶段] 开始并行执行 SubAgents...")
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._execute_single_task, task, data): task
                for task in tasks
            }

            # 收集结果
            for future in as_completed(future_to_task, timeout=self.timeout):
                task = future_to_task[future]
                try:
                    task_name, result = future.result()
                    results[task_name] = result
                    status = result.get("status", "unknown")
                    self._log(f"  - {task['description']}: {status}")
                except Exception as e:
                    self._log(f"  - {task['description']}: 执行异常 - {e}")
                    results[task["name"]] = {
                        "task": task["name"],
                        "status": "error",
                        "error": str(e),
                    }

        # 4. 汇总结果
        elapsed_time = time.time() - start_time
        self._log("\n" + "=" * 60)
        self._log(f"[完成] 所有任务执行完成，耗时 {elapsed_time:.2f} 秒")
        self._log("=" * 60)

        # 生成任务执行日志
        execution_log = self._generate_execution_log(results, elapsed_time)

        return {
            "status": "completed",
            "dataset_name": data.get("dataset_name", "数据集"),
            "elapsed_time": elapsed_time,
            "task_states": self.task_states,
            "results": results,
            "execution_log": execution_log,
        }

    def _generate_execution_log(self, results: Dict[str, Any], elapsed_time: float) -> str:
        """生成任务执行完成日志"""
        lines = [
            "=" * 60,
            "数据集分析任务执行完成日志",
            "=" * 60,
            f"总耗时：{elapsed_time:.2f}秒",
            "",
            "任务执行状态:",
        ]

        for task_name, state in self.task_states.items():
            result = results.get(task_name, {})
            status = result.get("status", "unknown")
            icon = "✓" if status == "success" else "✗"
            lines.append(f"  {icon} [{state.upper()}] {task_name}: {status}")

        # 汇总成功/失败数量
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        total_count = len(results)

        lines.extend([
            "",
            f"执行汇总：{success_count}/{total_count} 任务成功",
        ])

        if success_count < total_count:
            lines.append("注意：部分任务执行失败，请检查日志详情")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_task_status(self) -> Dict[str, str]:
        """获取当前任务状态"""
        return self.task_states.copy()

    def is_all_completed(self) -> bool:
        """检查所有任务是否完成"""
        return all(state == "completed" for state in self.task_states.values())
