# 渐进式 Skill Harness

这是一套可运行的 skill 发现、选择、按需加载、执行与语义记忆 harness。核心目标是：skill 数量增长时，不在启动阶段把所有提示词、资源和 Python 模块全部塞进内存或上下文。

## 四层加载模型

| 层 | 何时加载 | 内容 |
|---|---|---|
| Layer 1 — Metadata | 启动、HEARTBEAT | `skill.json` 中的名称、描述、关键词 |
| Layer 2 — Instructions | 明确选中某个 skill 后 | `SKILL.md` |
| Layer 3 — Runtime | 即将执行时 | 资源文件与 Python handler |
| Layer 4 — Memory | 每次执行前后 | FAISS 检索历史经验，执行结果写回 |

自动路由只对 Layer 1 目录做 embedding 排序。未命中的 skill 始终停留在元数据层；指定 `skill` 时甚至不需要给其他 skill 加载说明。

## 目录结构

```text
.
├── main.py
├── requirements.txt
├── skill_harness/
│   ├── api.py          # FastAPI 与 HEARTBEAT 生命周期
│   ├── config.py       # 环境配置
│   ├── executor.py     # 超时控制、上下文组装、执行与记忆写回
│   ├── llm.py          # OpenAI 兼容 LLM/Embedding
│   ├── memory.py       # FAISS Layer 4
│   ├── models.py       # Pydantic 协议
│   └── registry.py     # 发现、路由与逐层提升
├── skills/
│   ├── calculator/
│   └── summarizer/
└── tests/
```

## 启动

```powershell
cd 第十三周作业
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

接口文档位于 `http://127.0.0.1:8000/docs`。不配置 key 时使用本地确定性 hash embedding，计算器和摘要离线回退仍可运行；配置 OpenAI 兼容接口后，embedding 和摘要会使用远程模型。

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "https://your-compatible-endpoint/v1" # 官方接口可不设
```

## 调用示例

查看当前加载层级：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/skills
```

指定 skill 执行：

```powershell
$body = @{ input = "2 ** 10 + 1"; skill = "calculator" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/execute `
  -Method Post -ContentType "application/json" -Body $body
```

让 harness 自动选择：

```powershell
$body = @{
  input = "请总结：渐进式加载能够减少启动成本，并让大量技能保持冷状态。"
  arguments = @{ max_chars = 80 }
} | ConvertTo-Json -Depth 3
Invoke-RestMethod http://127.0.0.1:8000/execute `
  -Method Post -ContentType "application/json" -Body $body
```

查询 Layer 4：

```powershell
$body = @{ query = "之前计算过什么"; skill = "calculator"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/memory/search `
  -Method Post -ContentType "application/json" -Body $body
```

## Skill 协议

每个 skill 是 `skills/<name>/` 下的一个目录，至少包含：

```json
{
  "name": "my-skill",
  "version": "1.0.0",
  "description": "用于 Layer 1 路由的简短说明",
  "keywords": ["关键词"],
  "instructions": "SKILL.md",
  "handler": "handler.py:run",
  "resources": ["optional-resource.txt"],
  "enabled": true
}
```

handler 可以是同步或异步函数，签名固定为：

```python
def run(user_input: str, arguments: dict, context) -> object:
    ...
```

`context` 提供 `instructions`、`resources`、`memories` 与 `ai`。路径会被限制在该 skill 目录内，但 handler 本质上仍是受信任的本地 Python 插件；若要接收第三方 skill，应进一步放进容器或独立进程沙箱。

HEARTBEAT 默认每 30 秒重新扫描 manifest 并持久化 FAISS，可用 `HEARTBEAT_SECONDS` 调整，也可调用 `POST /skills/refresh` 立即刷新。

## 测试

```powershell
python -m unittest discover -s tests -v
```

