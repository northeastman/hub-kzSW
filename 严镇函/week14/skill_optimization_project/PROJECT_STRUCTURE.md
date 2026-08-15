# Skill 优化实验项目 - 文件结构说明

## 项目目录总览

```
skill_optimization_project/
│
├── skills/                      # Skill 文件目录
│   ├── skill_v1.md             # 原始版本（你的学习导师）
│   └── skill_v2.md             # 优化版本
│
├── agent/                       # Agent 核心模块
│   ├── executor.py             # 执行器：加载 Skill + 调用 LLM
│   └── optimizer.py            # 优化器：分析 V1 生成 V2
│
├── evaluator/                   # 评价模块
│   └── evaluator.py            # 评估输出质量 + 收集指标
│
├── config/                      # 配置目录
│   └── config.yaml             # API 配置、测试任务配置
│
├── logs/                        # 实验日志
│   ├── run_v1.json             # V1 执行记录
│   └── run_v2.json             # V2 执行记录
│
├── tasks/                       # 测试任务
│   └── test_tasks.yaml         # 用于测试的相同任务集
│
├── main.py                      # 主入口：运行完整实验
└── requirements.txt             # 依赖
```

---

## 各文件详细说明

### 1. `skills/skill_v1.md`

**作用**：存储原始版本的 Skill 文件

**内容**：你的"学习导师"Skill 原始版本（269 行）

**为什么需要**：作为对比实验的基准线（Baseline）

---

### 2. `skills/skill_v2.md`

**作用**：存储优化后的 Skill 版本

**内容**：经过 LLM 分析优化后的精简版 Skill

**为什么需要**：与 V1 对比，验证优化效果

---

### 3. `agent/executor.py`

**作用**：Agent 执行器

**职责**：
- 读取 Skill 文件内容
- 读取测试任务
- 调用 LLM API 执行任务
- 记录执行过程（Token 消耗、时间、步骤）

**为什么需要**：模拟真实使用 Skill 的场景，收集实验数据

**被谁调用**：`main.py` 调用它执行 V1 和 V2

---

### 4. `agent/optimizer.py`

**作用**：Skill 优化器

**职责**：
- 分析 V1 Skill 的内容
- 识别冗余、重复、可精简的部分
- 调用 LLM 生成优化建议
- 生成 V2 Skill 文件

**为什么需要**：自动化优化流程，减少人工干预

**被谁调用**：`main.py` 在执行完 V1 后调用它

---

### 5. `evaluator/evaluator.py`

**作用**：评估模块

**职责**：
- 收集执行指标（Token 数量、响应时间）
- 评估输出质量（是否符合预期）
- 生成结构化评估报告
- 保存结果到 logs 目录

**为什么需要**：量化对比 V1 和 V2 的效果

**被谁调用**：`main.py` 在每次执行后调用它

---

### 6. `config/config.yaml`

**作用**：项目配置文件

**内容**：
- LLM API 配置（API Key、模型选择、基础 URL）
- 实验配置（文件路径、日志目录）
- 评估配置（权重设置）

**为什么需要**：
- 集中管理配置，避免硬编码
- 方便切换模型或 API
- 敏感信息（如 API Key）统一管理

**被谁读取**：`main.py`、`executor.py`、`evaluator.py` 都会读取

---

### 7. `logs/run_v1.json` 和 `logs/run_v2.json`

**作用**：存储实验运行数据

**内容**：
```json
{
  "skill_version": "v1",
  "task_id": "task_1",
  "input_tokens": 1200,
  "output_tokens": 800,
  "response_time": 3.5,
  "quality_score": 0.85,
  "output_content": "...",
  "timestamp": "2026-08-06T10:00:00"
}
```

**为什么需要**：
- 保存原始数据用于分析
- JSON 格式方便程序读取和人工查看
- 可追溯每次实验的详细情况

**被谁写入**：`evaluator.py` 写入

---

### 8. `tasks/test_tasks.yaml`

**作用**：测试任务集

**内容**：多个测试任务，每个任务包含：
- 任务 ID
- 任务名称
- 测试 prompt

**为什么需要**：
- 确保 V1 和 V2 使用相同的测试任务
- 保证对比实验的公平性
- 方便扩展更多测试用例

**被谁读取**：`main.py` 读取后传给 `executor.py`

---

### 9. `main.py`

**作用**：主入口文件

**职责**：
1. 加载配置文件
2. 加载 Skill V1
3. 加载测试任务
4. 执行 V1 测试
5. 评估 V1 结果
6. 调用优化器生成 V2
7. 执行 V2 测试
8. 评估 V2 结果
9. 生成对比报告

**为什么需要**：串联整个实验流程

**调用关系**：
```
main.py
  ├── executor.py (执行 V1)
  ├── evaluator.py (评估 V1)
  ├── optimizer.py (生成 V2)
  ├── executor.py (执行 V2)
  └── evaluator.py (评估 V2 + 生成报告)
```

---

### 10. `requirements.txt`

**作用**：项目依赖列表

**内容**：
- `openai`：调用 LLM API
- `pyyaml`：读取 YAML 配置文件
- `tiktoken`：计算 Token 数量

**为什么需要**：明确项目依赖，方便环境搭建

**使用方式**：`pip install -r requirements.txt`

---

## 文件依赖关系图

```
config.yaml ──────────────────────────────────┐
                                               ↓
tasks.yaml ──→ main.py ──→ executor.py ──→ LLM API
                           │
                           ├──→ optimizer.py ──→ skill_v2.md
                           │
                           └──→ evaluator.py ──→ logs/*.json
```

---

## 数据流向

```
1. 读取配置 ──→ config.yaml
2. 读取任务 ──→ tasks/test_tasks.yaml
3. 读取 Skill ──→ skills/skill_v1.md
4. 执行测试 ──→ executor.py + LLM API
5. 评估结果 ──→ evaluator.py
6. 保存日志 ──→ logs/run_v1.json
7. 优化 Skill ──→ optimizer.py → skills/skill_v2.md
8. 重复 4-6 ──→ logs/run_v2.json
9. 生成报告 ──→ 对比分析
```

---

## 扩展说明

### 如果想添加更多 Skill 版本？

在 `skills/` 目录下添加 `skill_v3.md`、`skill_v4.md`，修改 `config.yaml` 中的路径配置即可。

### 如果想添加更多测试任务？

在 `tasks/test_tasks.yaml` 中添加新的任务条目，保持格式一致即可。

### 如果想换模型？

修改 `config.yaml` 中的 `model` 和 `base_url` 配置。