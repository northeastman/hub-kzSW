# Progressive Skills Agent Loop

这是一个最小 Agent Loop，演示如何在 Function Calling 循环中渐进式加载 `SKILL.md`。

它只实现四件核心事情：

1. 启动时只读取每个 Skill 的 YAML Frontmatter。
2. 模型命中 Skill 后，通过 `load_skill` 加载完整正文。
3. `references/` 和脚本等资源继续按需读取或执行。
4. 当前轮结束后释放 Skill 正文和工具轨迹，只保留最终问答。

## 目录

```text
src/
├── agent.py            # CLI
├── agent_loop.py       # Function Calling 循环
├── llm_config.py       # DeepSeek / Qwen 配置
├── skill_registry.py   # 三层 Skill 加载
└── tool_registry.py    # 动态工具及安全校验

skills/
├── flash-card/         # 可执行示例：生成 HTML 单词闪卡
└── baoyu-diagram/      # 资源按需加载示例：生成 SVG 图表
```

## 安装

建议使用 Python 3.10 以上版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置模型

DeepSeek：

```bash
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-your-key
```

通义千问：

```bash
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY=sk-your-key
```

可以用 `AGENT_MODEL` 覆盖默认模型名。

## 运行

交互模式：

```bash
python -m src.agent --verbose
```

单次请求：

```bash
python -m src.agent \
  --question "给 resilient 做一张英语单词闪卡" \
  --verbose
```

图表请求：

```bash
python -m src.agent \
  --question "画一个用户、API、订单服务和数据库的系统架构图" \
  --verbose
```

交互模式支持：

- `/skills`：查看常驻的 Skill 摘要索引。
- `/clear`：清空最终问答历史。
- `/exit`：退出。

## 渐进式加载过程

### 第一层：摘要索引

`SkillRegistry` 扫描 `skills/*/SKILL.md`，读取到第二个 `---` 后立即停止。此时不会读取 Skill 正文。

常驻 Prompt 类似：

```text
- [flash-card] 为一个英语单词生成静态 HTML 学习闪卡……
- [baoyu-diagram] 创建自包含的暗色主题 SVG 图表……
```

### 第二层：完整 Skill

匹配任务时，模型先调用：

```json
{"name": "load_skill", "arguments": {"name": "flash-card"}}
```

工具返回完整 `SKILL.md`，同时开放该 Skill 在 `allowed_tools` 中声明的工具。

### 第三层：相关资源

`baoyu-diagram` 加载后仍不会自动读取 `references/`。如果用户要架构图，模型只调用：

```json
{
  "name": "read_skill_resource",
  "arguments": {
    "skill_name": "baoyu-diagram",
    "path": "references/architecture.md"
  }
}
```

其他图表参考不会进入当前 Context。

## 单轮释放为什么重要

`AgentLoop` 区分两种消息：

- `history`：用户消息和最终回答，可进入下一轮。
- `turn_messages`：完整 Skill、工具调用和工具结果，仅当前轮可见。

如果把工具消息直接永久追加到历史，上一轮加载的 Skill 会一直占用 Context，就不是真正的渐进式加载。

## Skill Frontmatter

新增 Skill 时至少提供：

```yaml
---
name: my-skill
description: 一句话说明何时使用以及能完成什么。
version: 1.0.0
triggers:
  - 精确触发短语
allowed_tools:
  - read_skill_resource
  - write_workspace_file
---
```

当前内置工具：

| 工具 | 作用 |
|---|---|
| `load_skill` | 加载完整 `SKILL.md`，始终可见 |
| `read_skill_resource` | 读取已激活 Skill 目录内的文本资源 |
| `write_workspace_file` | 写入当前工作区内的相对路径 |
| `run_skill_script` | 运行已激活 Skill 的 `scripts/*.py` |

除 `load_skill` 外，其他工具只有在已加载 Skill 明确声明权限后才会出现在下一次 LLM 调用中。ßß
