### 技能目录结构
```
skills/
├── __init__.py
├── weather_query.py      # 天气查询技能（示例）
├── memory_summary.py     # 记忆总结技能（演示记忆访问）
└── calculator.py         # 计算器技能（简单工具示例）
```

### 创建新技能

1. 在 `skills/` 目录下创建新的 `.py` 文件
2. 定义 `SKILL_METADATA` 字典（必须包含 name, display_name, description, triggers）
3. 实现 `execute(context: dict) -> dict` 函数

示例：
```python
# skills/my_skill.py

SKILL_METADATA = {
    "name": "my_skill",
    "display_name": "我的技能",
    "description": "技能描述",
    "version": "1.0.0",
    "tags": ["标签"],
    "triggers": ["触发词"],
    "requires_memory": False,  # 是否需要访问记忆系统
}

def execute(context: dict) -> dict:
    # context 包含 query 和可选的 memory_access
    return {"message": "执行结果"}
```

### 技能热重载

修改技能文件后无需重启服务，下次执行时自动重新加载。

### Web API

| 接口 | 方法 | 功能 |
|------|------|------|
| `/skills` | GET | 列出所有可用技能 |
| `/skills/execute` | POST | 执行技能 |
| `/skills/{name}/load` | POST | 手动加载技能 |
| `/skills/{name}/unload` | POST | 卸载技能 |
| `/skills/stats` | GET | 技能系统统计 |
| `/skills/match` | POST | 根据查询匹配技能 |

### 测试技能系统

```bash
python test_skills.py
```

运行 6 个测试用例：
1. 技能发现
2. 渐进式加载
3. 技能执行
4. 技能匹配
5. 技能卸载
6. 列出可用技能

