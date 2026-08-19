# 多智能体系统 (基于 DeepSeek API)

## 项目结构
multi_agent_system/  
├── agents/  
│   ├── __init__.py  
│   ├── base_agent.py          # 基础 Agent 类，实现 ReAct 循环  
│   ├── main_agent.py          # 主 Agent，支持任务分解、子Agent创建、人类反馈  
│   └── sub_agent.py           # 子 Agent，轻量级 ReAct 循环  
├── layers/  
│   ├── __init__.py  
│   ├── perception.py          # 感知层：提示词组装、记忆检索  
│   ├── planning.py            # 规划层：LLM 调用封装与输出解析  
│   ├── execution.py           # 执行层：工具注册、并行执行、权限控制  
│   └── memory.py              # 记忆层：短期/长期记忆管理  
├── tools/  
│   ├── __init__.py  
│   ├── registry.py            # 工具注册表  
│   ├── builtin_tools.py       # 内置工具（搜索、计算器等）  
│   ├── create_subagent.py     # 创建子Agent工具  
│   ├── task_planner.py        # 任务图规划器工具  
│   ├── verify_result.py       # 结果验证工具  
│   └── ask_human.py           # 人类反馈工具（可选）  
├── utils/  
│   ├── __init__.py  
│   ├── llm_client.py          # DeepSeek API 封装  
│   └── config.py              # 配置文件  
├── main.py                    # 入口示例  
└── README.md                  # 说明文档  

## 功能特点
- 分层架构：感知层、规划层、执行层、记忆层
- 主 Agent 采用 ReAct 循环，可自主决定是否拆解任务
- 支持并行创建多个子 Agent 执行子任务
- 工具注册表机制，支持权限控制
- 任务图规划器（plan_tasks）
- 结果验证（verify_result）
- 人类反馈（ask_human）
- 长期/短期记忆管理

## 快速开始
1. 安装依赖：`pip install openai`
2. 设置 API Key：`export DEEPSEEK_API_KEY="your-key"` 或修改 `utils/config.py`
3. 运行：`python main.py`

## 自定义工具
在 `tools/` 目录下新建文件，定义函数，并使用 `global_registry.register()` 注册。
然后在 `tools/__init__.py` 的 `register_all_tools` 中调用注册函数。

## 架构说明
- **感知层** (`layers/perception.py`)：构建提示词，注入记忆
- **规划层** (`layers/planning.py`)：调用 LLM，解析输出
- **执行层** (`layers/execution.py`)：并行执行工具调用
- **记忆层** (`layers/memory.py`)：短期对话 + 长期记忆存储检索