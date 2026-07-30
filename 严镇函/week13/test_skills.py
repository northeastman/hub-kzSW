"""
技能系统测试脚本

测试渐进式加载、执行、热重载等功能
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

from src.skill_system import SkillRegistry


def test_discover_skills():
    """测试技能发现"""
    print("=" * 60)
    print("测试 1: 技能发现")
    print("=" * 60)
    
    registry = SkillRegistry()
    discovered = registry.discover_skills()
    
    print(f"发现 {len(discovered)} 个技能:")
    for skill in discovered:
        print(f"  - {skill.display_name} ({skill.name})")
        print(f"    触发词: {skill.triggers}")
        print(f"    需要记忆: {skill.requires_memory}")
    print()
    
    assert len(discovered) >= 3, "应该至少发现 3 个技能"
    print("✓ 技能发现测试通过\n")


def test_progressive_loading():
    """测试渐进式加载"""
    print("=" * 60)
    print("测试 2: 渐进式加载")
    print("=" * 60)
    
    registry = SkillRegistry()
    registry.discover_skills()
    
    # 初始状态：所有技能未加载
    stats = registry.get_skill_stats()
    print(f"初始状态: {stats['loaded_skills']}/{stats['total_skills']} 已加载")
    assert stats['loaded_skills'] == 0, "初始时应该没有技能被加载"
    
    # 加载单个技能
    success = registry.load_skill("calculator")
    print(f"加载 calculator: {success}")
    assert success, "calculator 技能应该加载成功"
    
    stats = registry.get_skill_stats()
    print(f"加载后: {stats['loaded_skills']}/{stats['total_skills']} 已加载")
    assert stats['loaded_skills'] == 1, "应该只有 1 个技能被加载"
    
    # 执行另一个技能（自动加载）
    result = registry.execute_skill("weather_query", {"city": "北京"})
    print(f"执行 weather_query: {result.success}")
    assert result.success, "weather_query 应该执行成功"
    print(f"输出: {result.output['message']}")
    
    stats = registry.get_skill_stats()
    print(f"执行后: {stats['loaded_skills']}/{stats['total_skills']} 已加载")
    assert stats['loaded_skills'] == 2, "应该有 2 个技能被加载"
    
    print("✓ 渐进式加载测试通过\n")


def test_skill_execution():
    """测试技能执行"""
    print("=" * 60)
    print("测试 3: 技能执行")
    print("=" * 60)
    
    registry = SkillRegistry()
    registry.discover_skills()
    
    # 测试计算器技能
    result = registry.execute_skill("calculator", {"expression": "2 + 3 * 4"})
    print(f"计算 2 + 3 * 4 = {result.output['result']}")
    assert result.success
    assert result.output['result'] == 14
    
    # 测试天气技能
    result = registry.execute_skill("weather_query", {"city": "上海"})
    print(f"天气查询: {result.output['message']}")
    assert result.success
    assert "上海" in result.output['message']
    
    # 测试执行时间
    print(f"执行时间: {result.execution_time:.4f}s")
    assert result.execution_time > 0
    
    print("✓ 技能执行测试通过\n")


def test_skill_matching():
    """测试技能匹配"""
    print("=" * 60)
    print("测试 4: 技能匹配")
    print("=" * 60)
    
    registry = SkillRegistry()
    registry.discover_skills()
    
    # 测试天气匹配
    matched = registry.match_skill_by_query("今天天气怎么样")
    print(f"'今天天气怎么样' -> {matched}")
    assert matched == "weather_query"
    
    # 测试计算器匹配
    matched = registry.match_skill_by_query("计算 123 + 456 等于多少")
    print(f"'计算 123 + 456 等于多少' -> {matched}")
    assert matched == "calculator"
    
    # 测试记忆总结匹配
    matched = registry.match_skill_by_query("帮我总结记忆")
    print(f"'帮我总结记忆' -> {matched}")
    assert matched == "memory_summary"
    
    # 测试无匹配
    matched = registry.match_skill_by_query("今天吃什么")
    print(f"'今天吃什么' -> {matched}")
    assert matched is None
    
    print("✓ 技能匹配测试通过\n")


def test_unload_skill():
    """测试技能卸载"""
    print("=" * 60)
    print("测试 5: 技能卸载")
    print("=" * 60)
    
    registry = SkillRegistry()
    registry.discover_skills()
    
    # 加载技能
    registry.load_skill("calculator")
    stats = registry.get_skill_stats()
    print(f"加载后: {stats['loaded_skills']} 已加载")
    assert stats['loaded_skills'] == 1
    
    # 卸载技能
    success = registry.unload_skill("calculator")
    print(f"卸载 calculator: {success}")
    assert success
    
    stats = registry.get_skill_stats()
    print(f"卸载后: {stats['loaded_skills']} 已加载")
    assert stats['loaded_skills'] == 0
    
    print("✓ 技能卸载测试通过\n")


def test_list_available_skills():
    """测试列出可用技能"""
    print("=" * 60)
    print("测试 6: 列出可用技能")
    print("=" * 60)
    
    registry = SkillRegistry()
    registry.discover_skills()
    
    skills = registry.list_available_skills()
    print(f"可用技能列表:")
    for skill in skills:
        status = "已加载" if skill["loaded"] else "未加载"
        print(f"  [{status}] {skill['display_name']} - {skill['description']}")
    
    assert len(skills) >= 3
    print("✓ 列出可用技能测试通过\n")


if __name__ == "__main__":
    print("\n技能系统测试\n")
    
    try:
        test_discover_skills()
        test_progressive_loading()
        test_skill_execution()
        test_skill_matching()
        test_unload_skill()
        test_list_available_skills()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)