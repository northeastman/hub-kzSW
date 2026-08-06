import yaml
import os
from agent.executor import SkillExecutor
from evaluator.evaluator import Evaluator
from agent.optimizer import SkillOptimizer

def load_config():
    """加载配置文件"""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_tests(executor, evaluator, skill_content, tasks, skill_version):
    """运行测试并返回结果"""
    results = {}

    for task in tasks["tasks"]:
        task_id = task["id"]
        task_name = task["name"]
        task_prompt = task["prompt"]

        print(f"\n📝 执行任务: {task_name}")

        # 执行任务
        result = executor.execute_task(skill_content, task_prompt)

        # 评估质量
        quality_score = evaluator.evaluate_quality(result["output_content"], task_prompt)

        # 计算综合评分
        scores = evaluator.calculate_score(result, quality_score)

        # 保存结果
        log_file = evaluator.save_result(result, scores, task_id, skill_version)

        # 存储结果
        results[task_id] = {
            "metrics": result,
            "scores": scores
        }

        print(f"   Token 消耗: {result['total_tokens']}")
        print(f"   响应时间: {result['response_time']:.2f}s")
        print(f"   质量评分: {quality_score:.2f}")
        print(f"   日志保存: {log_file}")

    return results

def main():
    """主函数"""
    print("="*50)
    print("=== Skill 优化实验系统 ===")
    print("="*50)

    # 1. 加载配置
    config = load_config()
    print("\n✅ 配置加载成功")

    # 2. 初始化模块
    executor = SkillExecutor(config)
    evaluator = Evaluator(config)
    optimizer = SkillOptimizer(config)

    # 3. 加载 Skill V1
    skill_v1_path = config["experiment"]["skill_v1_path"]
    skill_v1_content = executor.load_skill(skill_v1_path)
    v1_token_count = executor.count_tokens(skill_v1_content)
    print(f"✅ Skill V1 加载成功，Token 数: {v1_token_count}")

    # 4. 加载测试任务
    with open(config["experiment"]["tasks_path"], "r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f)
    print(f"✅ 加载了 {len(tasks['tasks'])} 个测试任务")

    # 5. 执行 V1 测试
    print("\n" + "-"*50)
    print("📊 阶段1：执行 V1 测试...")
    print("-"*50)

    v1_results = run_tests(executor, evaluator, skill_v1_content, tasks, "v1")

    print("\n" + "="*50)
    print("✅ V1 测试完成！")
    print("="*50)

    # 6. 优化 Skill
    print("\n" + "-"*50)
    print("🔧 阶段2：优化 Skill...")
    print("-"*50)

    skill_v2_path = config["experiment"]["skill_v2_path"]
    skill_v2_content = optimizer.optimize(skill_v1_content, skill_v2_path)

    v2_token_count = executor.count_tokens(skill_v2_content)
    print(f"\n📊 Skill Token 对比:")
    print(f"   V1: {v1_token_count} tokens")
    print(f"   V2: {v2_token_count} tokens")
    reduction = ((v1_token_count - v2_token_count) / v1_token_count) * 100
    print(f"   减少: {reduction:.1f}%")

    # 7. 执行 V2 测试
    print("\n" + "-"*50)
    print("📊 阶段3：执行 V2 测试...")
    print("-"*50)

    v2_results = run_tests(executor, evaluator, skill_v2_content, tasks, "v2")

    print("\n" + "="*50)
    print("✅ V2 测试完成！")
    print("="*50)

    # 8. 生成对比报告
    print("\n" + "-"*50)
    print("📈 阶段4：生成对比报告...")
    print("-"*50)

    # 对比每个任务的结果
    for task_id in v1_results:
        print(f"\n📋 任务 {task_id} 对比:")
        evaluator.compare_results(v1_results[task_id], v2_results[task_id])

    print("\n" + "="*50)
    print("🎉 实验完成！查看 logs/ 目录获取详细数据")
    print("="*50)

if __name__ == "__main__":
    main()
