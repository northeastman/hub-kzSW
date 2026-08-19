# # 数据集分析 Agent

基于 OTAC (Observe-Think-Act-Check) 循环的数据集自动分析系统。

## 功能特性

- **主 Agent + SubAgents 架构**: 主 Agent 负责任务分解和协调，SubAgents 并行执行具体分析任务
- **OTAC 循环**: 观察 - 思考 - 行动 - 验证的自主决策循环
- **多维度分析**:
  - 标签分布分析（均衡性、熵值、基尼系数）
  - 文本长度分布（统计指标、分位数、异常值）
  - 数据质量检查（缺失值、重复值、一致性）
- **可视化报告**: 自动生成统计图表和分析报告
- **优化建议**: 基于分析结果提供数据优化建议

## 安装依赖

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

## 快速开始

### 方式 1: 使用示例脚本

```bash
python -m dataset_analysis.run_analysis --dataset path/to/your/data.xlsx --format excel
```

### 方式 2: 编程方式使用

```python
from dataset_analysis import (
    CoordinatorAgent,
    ReportWriter,
    analyze_label_distribution,
    analyze_text_length,
    check_data_quality,
)
from dataset_analysis.llm import init_llm

# 初始化 LLM
llm = init_llm()

# 创建主 Agent
coordinator = CoordinatorAgent(llm, verbose=True)

# 准备数据
data = {
    "dataset_name": "我的数据集",
    "labels": ["A", "B", "A", "C", "B", ...],
    "texts": ["文本 1", "文本 2", ...],
    "data": [...],  # 原始数据
}

# 执行分析
results = coordinator.execute(data)

# 生成报告
writer = ReportWriter()
report_path = writer.generate_report(
    analysis_results=results["results"],
    dataset_name=data["dataset_name"],
    output_format="markdown"
)

# 获取优化建议
suggestions = writer.generate_optimization_suggestions(results["results"])
for s in suggestions:
    print(s)
```

### 方式 3: 单独使用分析工具

```python
from dataset_analysis import analyze_label_distribution, analyze_text_length, check_data_quality

# 标签分布分析
labels = ["A"] * 100 + ["B"] * 50 + ["C"] * 20
result = analyze_label_distribution(labels)
print(result["assessment"])  # 输出：中度不均衡

# 文本长度分析
texts = ["短文本", "中等长度的文本" * 10, "非常非常长的文本" * 100]
result = analyze_text_length(texts)
print(result["length_stats"]["mean"])

# 数据质量检查
data = [{"text": "abc", "label": "A"}, {"text": "", "label": "B"}]
result = check_data_quality(data)
print(f"质量评分：{result['quality_score']}")
```

## 项目结构

```
dataset_analysis/
├── __init__.py              # 模块入口
├── llm.py                   # LLM 客户端（现有）
├── core/
│   ├── __init__.py
│   ├── agent_state.py       # Agent 状态管理
│   ├── otat_loop.py         # OTAC 循环引擎
│   └── tool_registry.py     # 工具注册表
├── agents/
│   ├── __init__.py
│   ├── coordinator.py       # 主 Agent 协调器
│   └── subagents.py         # SubAgents 实现
├── tools/
│   ├── __init__.py
│   ├── label_analysis.py    # 标签分布分析
│   ├── text_analysis.py     # 文本长度分析
│   ├── quality_check.py     # 数据质量检查
│   ├── chart_generator.py   # 图表生成
│   └── summary_generator.py # 报告汇总
├── report/
│   ├── __init__.py
│   └── report_writer.py     # 报告生成器
├── tests/
│   ├── __init__.py
│   └── test_tools.py        #  单元测试单元测试
├── run_analysis.py          # 示例脚本
└── README.md                # 本文档
```

## OTAC 循环说明

OTAC 循环是本系统的核心决策机制：

1. **Observe (观察)**: 收集当前环境和状态信息
2. **Think (思考)**: LLM 基于观察进行推理，决定下一步行动
3. **Act (行动)**: 执行工具调用获取数据
4. **Check (验证)**: 验证结果有效性，决定是否需要继续循环

```
┌─────────────┐
│   Observe   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Think    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     Act     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Check    │───(需要继续)───┐
└──────┬──────┘               │
       │ (完成)                │
       ▼                       │
┌─────────────┐               │
│ Final Answer│               │
└─────────────┘               │
```

## 配置说明

### LLM 配置

编辑 `dataset_analysis/llm.py` 配置您的 LLM 服务：

```python
BASE_URL = "your-api-endpoint"
APP_ID = "your-app-id"
AK = "your-access-key"
SK = "your-secret-key"
```

### 分析参数

在调用 `CoordinatorAgent` 时可配置：

```python
coordinator = CoordinatorAgent(
    llm_client=llm,
    max_steps=8,        # 每个 SubAgent 最大执行步数
    max_workers=3,      # 并行工作线程数
    timeout=300,        # 单个任务超时时间（秒）
    verbose=True,       # 是否输出详细日志
)
```

## 输出示例

### 标签分布分析报告

```
=== 数据集 标签分布分析报告 ===

总样本数：1000
类别数：5

--- 分布详情 ---
  类别 A: 400 (40.0%)
  类别 B: 300 (30.0%)
  类别 C: 200 (20.0%)
  类别 D: 80 (8.0%)
  类别 E: 20 (2.0%)

--- 均衡性指标 ---
  最大/最小比值：20.00
  熵值：1.96
  基尼系数：0.32

--- 评估：严重不均衡 ---
```

## 常见问题

**Q: 为什么需要 OTAC 循环而不是直接调用工具？**

A: OTAC 循环允许 Agent 在分析过程中进行推理和验证，能够处理更复杂的分析任务，并在遇到问题时自我调整。

**Q: 可以添加自定义分析工具吗？**

A: 可以。在 `dataset_analysis/tools/` 目录下创建新工具，并在 `tool_registry.py` 中注册即可。

**Q: 如何修改报告输出格式？**

A: 在 `ReportWriter` 类中可以自定义输出格式，支持 Markdown 和 HTML。

## License

MIT
