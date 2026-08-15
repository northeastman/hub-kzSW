"""GEPA 离线批量优化 - 基于 DSPy GEPA 的反思式进化[reference:9][reference:10]"""
import logging
import random
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime

from .models import Skill, Trajectory, OptimizationResult, SkillSource

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """GEPA 候选变体"""
    skill: Skill
    score: float
    feedback: str
    generation: int


class GEPAOptimizer:
    """
    GEPA 优化器 - 反思式遗传算法

    核心机制[reference:11]:
    - 读取执行轨迹，理解为什么会失败（而不仅仅是知道它失败了）
    - 基于反思结果提出有针对性的改进
    - 维护 Pareto 前沿，保留在不同样例上表现最佳的 prompts

    参考: hermes-agent-self-evolution 项目
    """

    def __init__(
            self,
            metric: Callable[[Dict, str], Tuple[float, str]],
            llm_provider: Optional[Callable] = None,
            population_size: int = 5,
            max_generations: int = 10,
            mutation_rate: float = 0.3,
            use_merge: bool = True,
            reflection_minibatch_size: int = 3,
    ):
        self.metric = metric
        self.llm_provider = llm_provider
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.use_merge = use_merge
        self.reflection_minibatch_size = reflection_minibatch_size

        self.population: List[Candidate] = []
        self.pareto_front: List[Candidate] = []
        self.best_candidate: Optional[Candidate] = None

    def optimize(
            self,
            base_skill: Skill,
            train_dataset: List[Dict[str, Any]],
            val_dataset: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizationResult:
        """
        运行 GEPA 优化

        Args:
            base_skill: 待优化的 Skill
            train_dataset: 训练数据集，每个元素包含 input 和 expected
            val_dataset: 验证数据集

        Returns:
            优化结果
        """
        logger.info(f"开始 GEPA 优化: {base_skill.name}")
        logger.info(f"训练集大小: {len(train_dataset)}")

        # 初始化种群
        self.population = [
            Candidate(
                skill=self._mutate_skill(base_skill),
                score=0.0,
                feedback="",
                generation=0
            )
            for _ in range(self.population_size)
        ]

        # 评估初始种群
        self._evaluate_population(train_dataset)
        self._update_pareto_front()

        for gen in range(1, self.max_generations + 1):
            logger.info(f"GEPA Generation {gen}/{self.max_generations}")

            # 1. 反思：分析失败模式[reference:13]
            reflections = self._reflect_on_failures(train_dataset)

            # 2. 生成新变体（变异 + 合并）
            new_candidates = []
            for _ in range(self.population_size):
                if self.use_merge and random.random() < 0.3:
                    # 合并两个父代
                    parent1 = random.choice(self.pareto_front)
                    parent2 = random.choice(self.pareto_front)
                    child = self._merge_skills(parent1.skill, parent2.skill, reflections)
                else:
                    # 变异
                    parent = random.choice(self.pareto_front)
                    child = self._mutate_skill(parent.skill, reflections)
                new_candidates.append(
                    Candidate(
                        skill=child,
                        score=0.0,
                        feedback="",
                        generation=gen
                    )
                )

            # 3. 评估新变体
            self._evaluate_candidates(new_candidates, train_dataset)

            # 4. 选择：保留 Pareto 最优
            self.population.extend(new_candidates)
            self._update_pareto_front()

            # 5. 剪枝：保持种群大小
            self._prune_population()

            # 记录最佳
            if self.pareto_front:
                best = max(self.pareto_front, key=lambda c: c.score)
                if not self.best_candidate or best.score > self.best_candidate.score:
                    self.best_candidate = best
                    logger.info(f"新最佳: score={best.score:.3f}")

        # 返回最佳变体
        best = self.best_candidate or self.pareto_front[0]

        return OptimizationResult(
            original_skill=base_skill,
            optimized_skill=best.skill,
            improvement_score=best.score,
            metrics={
                "generations": self.max_generations,
                "population_size": len(self.population),
                "pareto_front_size": len(self.pareto_front),
                "best_score": best.score,
                "best_feedback": best.feedback,
            },
            generation=self.max_generations,
        )

    def _evaluate_population(self, dataset: List[Dict]):
        """评估整个种群"""
        self._evaluate_candidates(self.population, dataset)

    def _evaluate_candidates(self, candidates: List[Candidate], dataset: List[Dict]):
        """评估候选变体"""
        for candidate in candidates:
            scores = []
            feedbacks = []
            # 使用 minibatch 评估
            batch = random.sample(dataset, min(len(dataset), self.reflection_minibatch_size))
            for example in batch:
                # 使用 Skill 内容作为指令
                result = self._run_skill(candidate.skill, example)
                score, feedback = self.metric(example, result)
                scores.append(score)
                feedbacks.append(feedback)

            candidate.score = sum(scores) / len(scores) if scores else 0
            candidate.feedback = max(feedbacks, key=len) if feedbacks else ""

    def _run_skill(self, skill: Skill, example: Dict) -> str:
        """执行 Skill（模拟）"""
        # 实际使用时，这里应该调用 LLM
        # 这里简化为基于 Skill 内容生成响应
        prompt = f"""根据以下 Skill 指令处理输入：

## Skill 指令
{skill.content}

## 输入
{example.get('input', '')}

请输出结果：
"""
        if self.llm_provider:
            return self.llm_provider(prompt)
        # 无 LLM 时返回模拟结果
        return f"Skill '{skill.name}' 处理结果: {example.get('input', '')[:50]}"

    def _reflect_on_failures(self, dataset: List[Dict]) -> str:
        """
        反思失败模式[reference:16]

        GEPA 的 instruction_proposer 分析执行轨迹、反馈和失败，
        生成针对观察到的问题的改进指令。
        """
        failures = []
        for candidate in self.population[:5]:  # 只看前几个
            if candidate.score < 0.5:
                failures.append({
                    "skill": candidate.skill.name,
                    "score": candidate.score,
                    "feedback": candidate.feedback[:200],
                })

        if not failures:
            return "所有候选表现良好，继续优化细节。"

        reflection_prompt = f"""
分析以下 Skill 变体的失败模式：

{self._format_failures(failures)}

请总结：
1. 共性问题是什么？
2. 应该如何改进 Skill 指令？
3. 哪些策略应该被保留？

输出改进建议（200字以内）：
"""
        if self.llm_provider:
            return self.llm_provider(reflection_prompt)
        return "改进 Skill 的指令清晰度和步骤完整性。"

    def _format_failures(self, failures: List[Dict]) -> str:
        """格式化失败信息"""
        lines = []
        for f in failures:
            lines.append(f"- {f['skill']}: score={f['score']:.2f}")
            lines.append(f"  反馈: {f['feedback'][:100]}...")
        return "\n".join(lines)

    def _mutate_skill(self, skill: Skill, reflections: str = "") -> Skill:
        """变异 Skill"""
        mutation_prompt = f"""
对以下 Skill 进行变异改进：

## 当前 Skill
{skill.content}

## 改进方向
{reflections or "提升指令清晰度和执行效率"}

## 变异要求
1. 保持核心功能不变
2. 增加更清晰的步骤说明
3. 添加常见问题的处理方式
4. 优化输出格式

输出改进后的 Skill 内容（SKILL.md 格式）：
"""
        if self.llm_provider:
            new_content = self.llm_provider(mutation_prompt)
        else:
            new_content = self._template_mutate(skill, reflections)

        return Skill(
            name=skill.name,
            description=skill.description,
            content=new_content,
            source=SkillSource.GEPA,
            version=skill.version + 1,
            parent_name=skill.name,
        )

    def _template_mutate(self, skill: Skill, reflections: str) -> str:
        """模板变异（无 LLM）"""
        lines = skill.content.split("\n")
        # 简单变异：在末尾添加改进建议
        if reflections:
            lines.append(f"\n## 改进建议 (来自 GEPA)\n{reflections}")
        lines.append("\n## 执行检查清单\n- [ ] 确认输入有效性\n- [ ] 执行核心步骤\n- [ ] 验证输出结果")
        return "\n".join(lines)

    def _merge_skills(self, skill1: Skill, skill2: Skill, reflections: str = "") -> Skill:
        """合并两个 Skill"""
        merge_prompt = f"""
合并以下两个 Skill 的最佳实践：

## Skill A
{skill1.content}

## Skill B
{skill2.content}

## 合并要求
1. 保留两者的优点
2. 消除冗余
3. 生成更完整的流程
4. {reflections or "保持简洁"}

输出合并后的 Skill：
"""
        if self.llm_provider:
            new_content = self.llm_provider(merge_prompt)
        else:
            new_content = self._template_merge(skill1, skill2)

        return Skill(
            name=f"{skill1.name}-merged",
            description=f"合并自 {skill1.name} 和 {skill2.name}",
            content=new_content,
            source=SkillSource.GEPA,
            version=max(skill1.version, skill2.version) + 1,
            parent_name=skill1.name,
        )

    def _template_merge(self, skill1: Skill, skill2: Skill) -> str:
        """模板合并"""
        return f"""# 合并 Skill

## 来自 Skill A
{skill1.content[:500]}

## 来自 Skill B
{skill2.content[:500]}

## 合并版本
1. {skill1.name} 的核心步骤
2. {skill2.name} 的补充检查
3. 完整执行流程
"""

    def _update_pareto_front(self):
        """更新 Pareto 前沿"""
        # 简化为按分数排序取前 N 个
        sorted_candidates = sorted(self.population, key=lambda c: c.score, reverse=True)
        self.pareto_front = sorted_candidates[:self.population_size]

    def _prune_population(self):
        """剪枝种群"""
        # 保留 Pareto 前沿 + 随机保留一些多样性
        survivors = self.pareto_front.copy()
        remaining = self.population_size - len(survivors)
        if remaining > 0:
            others = [c for c in self.population if c not in survivors]
            survivors.extend(random.sample(others, min(remaining, len(others))))
        self.population = survivors
