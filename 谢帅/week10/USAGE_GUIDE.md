# 使用指南

> 电动自行车标准/法律 RAG 问答系统 — 从环境到评估的操作手册（仅原生版）

本项目所有命令使用 conda 虚拟环境 `study_env` 的解释器：

- Windows 路径：`D:\soft\Python\miniconda3\envs\study_env\python.exe`
- Git Bash 路径：`/d/soft/Python/miniconda3/envs/study_env/python.exe`

下文用 `PY` 代指该解释器。

---

## 一、环境准备

### 1.1 安装依赖
```bash
PY=/d/soft/Python/miniconda3/envs/study_env/python.exe
"$PY" -m pip install -r requirements.txt
```
> RAGAS 与 langchain 版本已在 requirements.txt 中锁定（ragas 0.2.15 + langchain 0.3.x），请勿随意升级 langchain-community，否则 ragas 导入会失败。

### 1.2 配置 API Key
```bash
# Windows CMD
set DASHSCOPE_API_KEY=sk-xxxx
# Git Bash / Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxx
```
控制台：https://dashscope.console.aliyun.com/ ；需可调用 `text-embedding-v3`、`qwen-plus`（改写用 `qwen-turbo`）。

---

## 二、构建流程（按顺序）

在 `homework/` 目录下执行。含中文/日志乱码时可加 `PYTHONIOENCODING=utf-8`。

### 步骤 1：生成文档清单（不需 key）
```bash
"$PY" src/build_manifest.py
```
产出 `data/manifest.json`（3 份可用文档；GB+811-2022 被排除）。

### 步骤 2：解析 PDF（不需 key）
```bash
"$PY" src/parse_pdf.py
```
产出 `data/parsed/*.json`。未安装 tesseract 时，扫描/乱码页降级为占位符（不阻断）。

### 步骤 3：文档分块（不需 key）
```bash
"$PY" src/chunk_documents.py         # 默认 semantic
```
产出 `data/chunks/all_semantic.json`。改脚本顶部 `STRATEGY="fixed"/"hierarchical"` 可生成其它策略。

### 步骤 4：构建向量索引（需 key）
```bash
"$PY" src/build_index.py
```
产出 `vectorstore/faiss_index.bin` + `faiss_meta.json`（218 向量）。

---

## 三、问答（rag_pipeline.py，需 key）

```bash
# 交互式
"$PY" src/rag_pipeline.py

# 单次查询
"$PY" src/rag_pipeline.py --query "电动自行车的最高设计车速是多少"

# 按文档/类型过滤
"$PY" src/rag_pipeline.py --query "饮酒后驾驶如何处罚" --doc-type 法律
"$PY" src/rag_pipeline.py --query "整车质量上限" --doc GB17761-2024

# 查询改写 / 消融开关
"$PY" src/rag_pipeline.py --query "..." --query-rewrite
"$PY" src/rag_pipeline.py --query "..." --no-bm25 --no-rerank
```

返回结构 `{answer, citations, retrieved}`，答案句末带 `[n]` 来源编号。

示例输出：
```
问题：电动自行车的最高设计车速是多少
电动自行车的最高设计车速不应超过25 km/h[1]。
── 来源 ──
  [1] GB17761-2024《...电动自行车安全技术规范》（GB17761-2024） · 车速限值要求 · 第11页
```

---

## 四、HTTP 服务（serve.py）

```bash
cd src
"$PY" -m uvicorn serve:app --host 127.0.0.1 --port 8000
```

- 浏览器打开 http://127.0.0.1:8000/ ：教学可视化页面（展示向量/BM25/RRF/上下文/生成各步）。
- Swagger 文档：http://127.0.0.1:8000/docs
- 健康检查：`GET /health`
- 问答：`POST /query`，body `{"question": "...", "doc_id": "GB17761-2024", "doc_type": "标准"}`（后两者可选）
- 调试：`POST /query/debug`，返回每一步中间结果。

Python 调用示例：
```python
import requests
r = requests.post("http://127.0.0.1:8000/query",
                  json={"question": "电动自行车整车质量的上限是多少"})
print(r.json()["answer"])
```
> 用 curl 传中文时注意终端编码，Windows 下建议用 Python requests。

---

## 五、评估

### 5.1 RAGAS 四指标（需 key）
```bash
# 全部 15 题
"$PY" evaluation/evaluate.py --pipeline native
# 仅部分题（调试）
"$PY" evaluation/evaluate.py --pipeline native --question-ids 6,9
# 跳过打分，只看答案
"$PY" evaluation/evaluate.py --pipeline native --skip-ragas
```
结果存 `evaluation/results/native_{时间戳}.json` 与 `.csv`。

### 5.2 消融实验（需 key，检索指标不调用打分 LLM）
```bash
# 默认 semantic × (vector_only, hybrid)
"$PY" evaluation/compare_strategies.py

# 完整 3 策略 × 3 检索（需先构建 fixed/hierarchical 索引，见下）
"$PY" evaluation/compare_strategies.py \
    --strategies fixed,semantic,hierarchical \
    --modes vector_only,bm25_only,hybrid
```
输出 Hit Rate@4 / MRR 汇总表，存 `evaluation/results/ablation_results.json`。

**构建 fixed / hierarchical 索引**（消融多策略前置）：分别把 `chunk_documents.py` 的 `STRATEGY` 改为对应值生成 `all_fixed.json` / `all_hierarchical.json`，再将其 embedding 存到 `vectorstore/faiss_fixed/{index.bin,meta.json}`、`vectorstore/faiss_hierarchical/{index.bin,meta.json}`（`compare_strategies.load_index` 按此路径查找；semantic 用默认主索引）。

---

## 六、常见问题

**Q: FAISS 报 `could not open ... for writing/reading: No such file or directory`**
含中文路径导致。本项目已改用 `serialize_index`/`deserialize_index` + Python 文件 IO 规避；若自行改动读写请沿用该方式。

**Q: RAGAS 报 `No module named 'langchain_community.chat_models.vertexai'`**
langchain-community 版本过新。按 requirements.txt 锁定 `ragas==0.2.15` + `langchain==0.3.27` + `langchain-community==0.3.30`。

**Q: 答案返回“根据提供的资料无法回答此问题”**
可能相关性低于阈值（`SCORE_THRESHOLD=0.25`）或问题超范围。确认索引已构建，或换更贴合标准/法律的问法。

**Q: 想纳入 GB811-2022（头盔标准）**
该 PDF 为 CID 乱码需 OCR：安装 tesseract + 中文语言包，从 `build_manifest.py` 的 `EXCLUDE_FILES` 移除后重跑流程。

**Q: 日志中文乱码**
Windows 控制台 GBK 所致，仅显示问题；文件（JSON）均为 UTF-8。命令前加 `PYTHONIOENCODING=utf-8` 可改善。
