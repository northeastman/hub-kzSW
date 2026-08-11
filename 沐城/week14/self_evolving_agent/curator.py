"""命令行接口"""
import argparse
import logging
import json
from pathlib import Path

from self_evolving_agent.models import Trajectory
from self_evolving_agent.skill_system import SelfEvolvingSkillSystem

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="自进化 Skill 系统")
    parser.add_argument("--skill-dir", default="~/.self_evolving_skills/",
                        help="Skill 存储目录")
    parser.add_argument("--action", choices=["list", "nudge", "optimize", "curator", "stats"],
                        default="stats", help="执行动作")
    parser.add_argument("--skill", help="Skill 名称")
    parser.add_argument("--iterations", type=int, default=5,
                        help="GEPA 迭代次数")
    parser.add_argument("--task", help="任务描述 (用于 nudge)")

    args = parser.parse_args()

    system = SelfEvolvingSkillSystem(skill_dir=args.skill_dir)

    if args.action == "list":
        skills = system.list_skills()
        print(f"共 {len(skills)} 个 Skills:")
        for s in skills:
            print(f"  - {s.name} (v{s.version}, {s.status})")

    elif args.action == "nudge":
        if not args.task:
            print("请提供 --task")
            return
        # 模拟轨迹
        traj = Trajectory(
            task_description=args.task,
            messages=[{"role": "user", "content": args.task}],
            tool_calls=[],
            tool_call_count=0,
            success=True,
        )
        system.on_task_complete(traj)
        print(f"已处理 Nudge: {args.task}")

    elif args.action == "optimize":
        dataset = [
            {"input": "审查这段 Python 代码", "expected": "代码审查报告", "success": True},
            {"input": "优化数据库查询", "expected": "优化建议", "success": True},
        ]
        results = system.optimize_skills(
            skill_name=args.skill,
            train_dataset=dataset,
        )
        for r in results:
            print(f"✓ {r.original_skill.name} → {r.optimized_skill.name}")
            print(f"  得分: {r.improvement_score:.3f}")

    elif args.action == "curator":
        stats = system.run_curator()
        print(f"Curator 完成: {json.dumps(stats, indent=2)}")

    else:  # stats
        stats = system.get_stats()
        print(f"系统统计: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
