"""SelfEvolvingSkillSystem - 自进化 Skill 系统主入口"""
import logging
from typing import List, Optional, Dict, Any, Callable, Tuple
from datetime import datetime

from .models import Skill, Trajectory, OptimizationResult, SkillStatus
from .nudge_engine import NudgeEngine
from .gepa_optimizer import GEPAOptimizer
from .curator import Curator

logger = logging.getLogger(__name__)


class SelfEvolvingSkillSystem:
    """
    自进化 Skill 系统

    完整闭环:
    执行 → Nudge 捕捉 → GEPA 优化 → Curator 维护 → 部署
    """

    def __init__(
            self,
            skill_dir: str = "~/.self_evolving_skills/",
            llm_provider: Optional[Callable] = None,
            memory_nudge_interval: int = 10,
            skill_nudge_interval: int = 10,
            gepa_population_size: int = 5,
            gepa_max_generations: int = 10,
            curator_stale_days: int = 7,
            curator_archive_days: int = 30,
    ):
        self.skill_dir = skill_dir

        # 初始化各组件
        self.nudge_engine = NudgeEngine(
            memory_nudge_interval=memory_nudge_interval,
            skill_nudge_interval=skill_nudge_interval,
            llm_provider=llm_provider,
            skill_saver=self.save_skill,
        )

        self.gepa_optimizer = GEPAOptimizer(
            metric=self._default_metric,
            llm_provider=llm_provider,
            population_size=gepa_population_size,
            max_generations=gepa_max_generations,
        )

        self.curator = Curator(
            skill_dir=skill_dir,
            stale_days=curator_stale_days,
            archive_days=curator_archive_days,
        )

        self._skills: Dict[str, Skill] = {}
        self._trajectories: List[Trajectory] = []
        self._optimization_history: List[OptimizationResult] = []

        # 加载已有 Skills
        self._load_skills()

    def _load_skills(self):
        """加载已有 Skills"""
        from pathlib import Path
        skill_dir = Path(self.skill_dir).expanduser()
        if skill_dir.exists():
            for skill_file in skill_dir.glob("*.md"):
                try:
                    skill = Skill.from_markdown(skill_file.read_text())
                    self._skills[skill.name] = skill
                except Exception:
                    pass

    def save_skill(self, skill: Skill):
        """保存 Skill"""
        from pathlib import Path
        skill_dir = Path(self.skill_dir).expanduser()
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_file = skill_dir / f"{skill.name}.md"
        skill_file.write_text(skill.to_markdown())
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取 Skill"""
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        """列出所有 Skills"""
        return list(self._skills.values())

    def on_task_complete(self, trajectory: Trajectory):
        """
        任务完成回调 - 触发自进化闭环

        Args:
            trajectory: 执行轨迹
        """
        logger.info("任务完成，触发自进化检查")

        # 1. 记录轨迹
        self._trajectories.append(trajectory)

        # 2. Nudge: 判断是否需要沉淀经验[reference:27]
        skill = self.nudge_engine.try_nudge(trajectory)
        if skill:
            logger.info(f"Nudge 生成新 Skill: {skill.name}")
            self.save_skill(skill)

        # 3. 更新计数器
        self.nudge_engine.record_turn()
        self.nudge_engine.record_llm_call()

    def optimize_skills(self,
                        skill_name: Optional[str] = None,
                        train_dataset: Optional[List[Dict]] = None,
                        ) -> List[OptimizationResult]:
        """
        触发 GEPA 离线批量优化

        Args:
            skill_name: 要优化的 Skill 名称，None 表示优化所有
            train_dataset: 训练数据集

        Returns:
            优化结果列表
        """
        if train_dataset is None:
            train_dataset = self._build_dataset_from_trajectories()

        if not train_dataset:
            logger.warning("无训练数据，跳过优化")
            return []

        targets = [self._skills[name] for name in self._skills
                   if skill_name is None or name == skill_name]

        results = []
        for skill in targets:
            logger.info(f"优化 Skill: {skill.name}")
            result = self.gepa_optimizer.optimize(skill, train_dataset)
            results.append(result)

            # 保存优化后的 Skill
            if result.improvement_score > 0:
                self.save_skill(result.optimized_skill)
                self._optimization_history.append(result)
                logger.info(f"✓ {skill.name} 优化完成，提升: {result.improvement_score:.3f}")

        return results

    def _build_dataset_from_trajectories(self) -> List[Dict]:
        """从执行轨迹构建训练数据集"""
        dataset = []
        for traj in self._trajectories[-50:]:  # 最多取最近50条
            dataset.append({
                "input": traj.task_description,
                "expected": traj.final_result,
                "success": traj.success,
                "trajectory": traj,
            })
        return dataset

    def _default_metric(self, example: Dict, prediction: str) -> Tuple[float, str]:
        """默认评估指标"""
        # 检查预测是否包含关键信息
        score = 0.0
        feedback = []

        expected = example.get("expected", "")
        if expected and prediction:
            # 简单相似度
            common = len(set(prediction.split()) & set(expected.split()))
            total = len(set(expected.split()))
            score = common / total if total > 0 else 0.5

        # 检查是否成功
        if example.get("success", False):
            score = max(score, 0.7)
            feedback.append("任务成功")

        feedback.append(f"预测长度: {len(prediction)}")

        return min(score, 1.0), "; ".join(feedback)

    def run_curator(self) -> Dict[str, int]:
        """运行 Curator 维护"""
        if self.curator.should_run():
            return self.curator.run()
        return {"checked": 0, "message": "未到运行时间"}

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "total_skills": len(self._skills),
            "active_skills": len([s for s in self._skills.values() if s.status == SkillStatus.ACTIVE]),
            "total_trajectories": len(self._trajectories),
            "optimization_count": len(self._optimization_history),
            "nudge_interval": self.nudge_engine.skill_nudge_interval,
        }
