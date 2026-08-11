"""示例：使用自进化系统优化代码审查 Skill"""

import sys

from self_evolving_agent.models import Skill, Trajectory
from self_evolving_agent.skill_system import SelfEvolvingSkillSystem

sys.path.append("..")


# 初始 Skill（人工编写）
INITIAL_CODE_REVIEW_SKILL = """---
name: code-reviewer
description: 代码审查专家，检查代码质量、发现bug、提出改进建议
---

# 代码审查专家

你是一个代码审查专家。请按以下步骤审查代码：

## 审查步骤
1. 读取代码文件，理解代码逻辑
2. 检查代码风格是否符合规范（命名、缩进、注释等）
3. 检查潜在bug（空指针、边界条件、异常处理等）
4. 检查性能问题（循环效率、数据库查询、内存使用等）
5. 检查安全问题（SQL注入、XSS、权限控制等）
6. 生成审查报告，按严重程度分类

## 输出格式
- 严重问题：必须修复
- 一般问题：建议修复
- 优化建议：可选
"""


def demo():
    """演示自进化 Skill 优化流程"""

    # 1. 初始化系统
    system = SelfEvolvingSkillSystem(
        skill_dir="~/.demo_skills/",
        memory_nudge_interval=3,
        skill_nudge_interval=3,
        gepa_population_size=3,
        gepa_max_generations=3,
    )

    # 2. 注册初始 Skill
    skill = Skill.from_markdown(INITIAL_CODE_REVIEW_SKILL)
    system.save_skill(skill)
    print(f"✓ 已注册初始 Skill: {skill.name}")

    # 3. 模拟任务执行 - 触发 Nudge
    trajectories = [
        Trajectory(
            task_description="审查一个 Python 项目的代码质量",
            messages=[{"role": "user", "content": "请审查这段代码"}],
            tool_calls=[{"name": "read_file", "type": "tool_call"}],
            tool_call_count=3,
            success=True,
        ),
        Trajectory(
            task_description="检查 JavaScript 代码的安全漏洞",
            messages=[{"role": "user", "content": "检查安全漏洞"}],
            tool_calls=[{"name": "scan_security", "type": "tool_call"}],
            tool_call_count=5,
            has_error=True,
            error_fixed=True,
            success=True,
        ),
    ]

    for traj in trajectories:
        system.on_task_complete(traj)
        print(f"✓ 处理轨迹: {traj.task_description[:30]}...")

    # 4. 查看 Nudge 生成的 Skill
    print(f"\n当前 Skills: {[s.name for s in system.list_skills()]}")

    # 5. 运行 GEPA 离线优化
    dataset = [
        {"input": "审查这段 Python 代码", "expected": "代码审查报告", "success": True},
        {"input": "检查安全漏洞", "expected": "安全报告", "success": True},
        {"input": "优化性能问题", "expected": "性能优化建议", "success": False},
    ]

    print("\n开始 GEPA 优化...")
    results = system.optimize_skills(
        skill_name="code-reviewer",
        train_dataset=dataset,
    )

    for r in results:
        print(f"\n优化结果:")
        print(f"  原始 Skill: {r.original_skill.name}")
        print(f"  优化后: {r.optimized_skill.name}")
        print(f"  提升得分: {r.improvement_score:.3f}")
        print(f"  指标: {r.metrics}")

    # 6. 运行 Curator
    print("\n运行 Curator...")
    stats = system.run_curator()
    print(f"  Curator 统计: {stats}")

    # 7. 最终统计
    print(f"\n系统统计: {system.get_stats()}")


if __name__ == "__main__":
    demo()
