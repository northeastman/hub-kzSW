# USAGE_GUIDE.md — Skill Harness 使用指南

## 一、环境准备

### 依赖安装
```powershell
cd nlp_chieh/钱杰/week13
pip install -r requirements.txt
```

### API Key 配置

默认 DeepSeek，备选 Qwen：
```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-xxx"
# 或切换 Qwen
$env:LLM_PROVIDER = "qwen"
$env:DASHSCOPE_API_KEY = "sk-xxx"
```

```bash
# Linux/Mac
export DEEPSEEK_API_KEY="sk-xxx"
```

---

## 二、启动服务

```powershell
cd nlp_chieh/钱杰/week13
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开 `http://localhost:8000`。

启动时 harness 只执行 Level 0（扫描目录名），秒级就绪，**不会读取任何 SKILL.md 内容**。

---

## 三、Web UI 演示

### 试这两句话
1. **「给我做一张 happy 的闪卡」** —— 会命中 `flash-card` skill
2. **「画一个微服务架构图」** —— 会命中 `baoyu-diagram` skill

### 观察右侧面板
- **已注册 Skills**：每个 skill 卡片下有 4 段进度条，对应 Level 0~3
- 每发生一次 SSE 事件，进度条会实时前进：SCAN → META → FULL → ASSETS
- **加载统计**：显示各级别已加载的 skill 数量
- **最近使用**：历史执行记录

### 观察左侧聊天流
每个 SSE 事件会以彩色小气泡显示：
- 青色 = 意图/步骤
- 黄色 = 工具调用
- 绿色 = 完成/结果
- 红色 = 错误

你会清楚看到：
```
[意图] 候选 2 个：baoyu-diagram, flash-card
[意图] 命中 1 个：flash-card(95%)
[选中] flash-card（95%）—— 用户要做单词闪卡
[执行] 启动 flash-card
[升级] → Level FULL(2)（body 1850 字符）
[升级] → Level ASSETS(3)（1 脚本资源）
[步骤] Step 1 · thinking
[工具] write_file({"path":"happy.json","content":"..."})
[观察] 已写入 happy.json（420 字符）
[步骤] Step 2 · thinking
[工具] run_script({"command":"python scripts/make_flashcard.py ..."})
[观察] exit_code=0 stdout: Generated happy.html
[步骤] Step 3 · thinking
[完成] 已生成 happy.html 闪卡
[结束] success · 3步 · 8420ms
```

---

## 四、HTTP 接口

### POST /chat — 主入口
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"给我做一张 crazy 的闪卡"}'
```
强制指定 skill（跳过意图匹配）：
```json
{"message":"...", "force_skill":"flash-card"}
```

### GET /skills — 列出所有 skill 及级别
```bash
curl http://localhost:8000/skills
```

### GET /skills/{name}/load?level=N — 手动升级
```bash
curl "http://localhost:8000/skills/flash-card/load?level=3"
```

### GET /stats — 注册表 + 使用统计
### GET /usage — 最近使用记录
### POST /reset — 清空使用记录

---

## 五、作为模块调用

```python
import sys
sys.path.insert(0, "path/to/week13")

from src.skill_registry import SkillRegistry
from src.intent_matcher import IntentMatcher
from src.skill_executor import SkillExecutor
from src.memory_store import MemoryStore

# 1. 注册表：Level 0 扫描
reg = SkillRegistry("skills")
reg.scan()                        # 发现 skill 名字，不读文件

# 2. 意图匹配：批量升到 Level 1
import asyncio
matcher = IntentMatcher(reg)
matched = asyncio.run(matcher.match("做一张 happy 的闪卡"))
# → [{"name":"flash-card","score":0.95,...}]

# 3. 执行：升到 Level 2 + 3，跑 ReAct 循环
memory = MemoryStore("outputs/skill_memory.db")
executor = SkillExecutor(reg, memory, "outputs/work")
result = asyncio.run(executor.execute("flash-card", "做一张 happy 的闪卡"))
# → {"status":"success","summary":"...","steps":3,"duration_ms":8420}

# 4. 单独测试各级加载
reg.load_metadata("flash-card")   # Level 1
reg.load_full("flash-card")       # Level 2
reg.load_assets("flash-card")     # Level 3
print(reg.stats())
```

---

## 六、添加自己的 skill

在 `skills/` 下新建目录，放一个 `SKILL.md`：

```markdown
---
name: my-skill
description: >-
  当用户说"做 XX"时触发本 skill。一句话描述能力边界。
---

# My Skill

## 执行流程
1. 第一步...
2. 调用 scripts/xxx.py ...
3. 返回结果

## 可用脚本
- `scripts/run.py` —— 做某事
```

可选子目录：`scripts/`（可执行脚本）、`references/`（参考文档）、`data/`（数据）。

重启服务，harness 自动发现新 skill（Level 0 扫描）。

---

## 七、调试 FAQ

**Q: 启动很慢？**
A: 不会。启动只做 Level 0 目录扫描，不读 SKILL.md。慢的话检查是不是误装了 `faiss-gpu`（本项目不需要 FAISS）。

**Q: 意图匹配总是不命中？**
A: 检查 SKILL.md 的 `description` 是否写清了触发场景。也可以用 `force_skill` 参数绕过匹配直接执行。

**Q: 执行器卡在某一步不动？**
A: 看 SSE 事件流。`exec_tool_result` 会显示脚本 stdout/stderr。`run_script` 有 120 秒超时。

**Q: write_file 报"路径越界"？**
A: 路径必须相对 `outputs/work/`，不能写 `..` 或绝对路径。这是安全限制。

**Q: 怎么看 skill 加载到哪一级了？**
A: `GET /skills` 或 `GET /stats`，看 `level` 字段（0~3）。

**Q: ReAct 循环最多几轮？**
A: `MAX_STEPS=10`，在 `skill_executor.py` 顶部可改。
