# ARCHITECTURE.md — Skill Harness 渐进式加载执行框架

## 一、项目定位

本项目把 `note/week13 skills和harness/` 下两个示例（`skills/` 与 `agent_memory_system/`）的思想合并，做一套 **可以渐进式加载并执行 skill 的 harness**：

- 借鉴 `skills/` 的 **SKILL.md 声明格式**（frontmatter + 自然语言流程）
- 借鉴 `agent_memory_system/` 的 **FastAPI + SSE 流式架构** 与 **LLM 提供商配置**
- 核心创新：**4 级渐进式加载**，让 100 个 skill 也只付 1 个 skill 的运行时成本

回答两个问题：
> "skill 一多就启动慢、占内存，怎么办？" —— 渐进式加载，按需升级
> "怎么让 LLM 真的按 SKILL.md 的描述干活？" —— ReAct 工具循环

---

## 二、4 级渐进式加载模型

```
┌──────────────────────────────────────────────────────────────┐
│  Level 0  SCAN     仅扫描目录名            零 IO，启动毫秒级  │
│     ↓ 意图匹配前批量预热                                     │
│  Level 1  META     解析 YAML frontmatter   只读前 2KB        │
│     ↓ 命中后才升级                                           │
│  Level 2  FULL     完整 SKILL.md 正文      执行前必经         │
│     ↓ 真正调工具时才升级                                     │
│  Level 3  ASSETS   scripts/references/data 目录列举           │
└──────────────────────────────────────────────────────────────┘
```

| 级别 | 读取内容 | 何时触发 | 成本 |
|------|---------|---------|------|
| 0 SCAN | `iterdir()` | 启动 | O(N) 目录列举，无文件读 |
| 1 META | SKILL.md 前 2KB | 意图匹配前批量预热 | 每 skill 一次小读 |
| 2 FULL | SKILL.md 全文 | 该 skill 被命中 | 仅命中 skill 付代价 |
| 3 ASSETS | `scripts/`、`references/`、`data/` 子树 | 真正执行工具前 | 仅执行 skill 付代价 |

**关键性质**：一次会话中绝大多数 skill 永远停在 Level 0/1，只有真正用上的那一个会走到 Level 3。

---

## 三、整体流水线

```
用户输入
    │
    ▼
[IntentMatcher]  registry.load_all_metadata()  →  所有 skill 升到 Level 1
    │             拼候选清单 → LLM 输出 JSON {matched:[{name,score,reason}]}
    │             按 score 降序，过滤 < 0.6
    ▼
[SkillExecutor]  取 Top-1
    │   ├── registry.load_full(name)      → Level 2
    │   ├── registry.load_assets(name)    → Level 3
    │   ├── memory.record_start()
    │   └── ReAct 循环（最多 10 轮）：
    │         LLM(system=SKILL.md 正文 + 工具说明, user=请求)
    │         → 输出工具调用 JSON
    │         → 执行 write_file / run_script / finish
    │         → 结果回灌 → 下一轮
    ▼
[MemoryStore]    record_finish(status, summary, duration_ms)
    │
    ▼
SSE 全程推送给前端：intent_* / exec_level_up / exec_step / exec_tool_call / exec_done
```

---

## 四、ReAct 工具循环

执行器给 LLM 三个工具（手动 function calling，不依赖各家 SDK 差异）：

| 工具 | 参数 | 作用 |
|------|------|------|
| `write_file` | `path`, `content` | 写文件到工作目录（路径越界会被拒） |
| `run_script` | `command` | 在 skill 根目录执行 shell 命令（脚本可用相对路径找 data/references） |
| `finish` | `summary` | 结束并返回给用户的最终摘要 |

每轮 LLM 输出一个工具调用 JSON，执行后把 observation 喂回去，直到 `finish` 或达到 `MAX_STEPS=10`。

**为什么手动 function calling？** OpenAI/DeepSeek/Qwen 各家 function calling 接口细节不同，手动 JSON 解析 + 正则兜底更通用、更可控，教学场景也更直观。

---

## 五、目录结构

```
week13/
├── src/
│   ├── __init__.py
│   ├── skill_registry.py    # 4 级渐进式加载核心
│   ├── intent_matcher.py    # LLM 意图匹配
│   ├── skill_executor.py    # ReAct 工具循环执行
│   ├── memory_store.py      # SQLite 使用记录
│   ├── llm_config.py        # LLM 提供商配置
│   └── serve.py             # FastAPI + SSE
│
├── skills/                  # 示例 skill（从 note/week13 复制）
│   ├── flash-card/          # 单词闪卡生成
│   └── baoyu-diagram/       # 暗色 SVG 图表生成
│
├── outputs/
│   ├── skill_memory.db      # 使用记录
│   └── work/                # 执行期工作目录（write_file 落点）
│
├── index.html               # Web UI
├── requirements.txt
├── ARCHITECTURE.md
└── USAGE_GUIDE.md
```

---

## 六、SSE 事件流

前端通过 `POST /chat` 的 SSE 流接收每一步进度，事件类型：

| 阶段 | 事件 | 含义 |
|------|------|------|
| 意图 | `intent_start` | 开始匹配 |
|      | `intent_candidates` | 候选清单 |
|      | `intent_done` | 匹配结果 |
| 执行 | `skill_selected` | 选中 Top-1 |
|      | `exec_start` | 开始执行 |
|      | `exec_level_up` | skill 加载级别升级（2→3） |
|      | `exec_step` | ReAct 新一轮 |
|      | `exec_tool_call` | LLM 决定调工具 |
|      | `exec_tool_result` | 工具执行结果 |
|      | `exec_finish` | LLM 调 finish |
|      | `exec_done` | 执行结束（含状态/步数/耗时） |

---

## 七、与两个参考项目的关系

| 参考 | 借鉴点 | 改造点 |
|------|--------|--------|
| `skills/flash-card`、`skills/baoyu-diagram` | SKILL.md 声明格式、frontmatter（name/description/version） | 直接复用，作为 harness 的被加载对象 |
| `agent_memory_system/` | FastAPI lifespan 单例、SSE 广播、`llm_config.py` 提供商切换、SQLite 持久化 | 把"四层记忆"换成"四级 skill 加载"，把 Memory Flush 换成 ReAct 执行 |
