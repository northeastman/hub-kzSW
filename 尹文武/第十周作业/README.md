# Local RAG System

基于原生 Python 实现的本地知识库增强生成系统（Retrieval Augmented Generation）。

本项目不依赖 LangChain / LlamaIndex 等 RAG 框架，所有核心模块自主实现，方便理解底层机制、实验不同检索策略，并支持后续扩展。

---

# 1. 项目特点

## 1.1 核心能力

* PDF 文档解析
* OCR 文本提取
* 中文文本清洗
* 书籍级语义切分
* 本地 Embedding
* FAISS 向量检索
* 混合检索
* Query Expansion
* Reranking
* 本地大模型生成
* 自动化效果评估

---

# 2. 技术栈

| 模块               | 技术                      |
| ---------------- | ----------------------- |
| OCR              | 百度 Unlimited-OCR        |
| Embedding        | BAAI/bge-large-zh-v1.5  |
| Vector DB        | FAISS                   |
| Sparse Retrieval | BM25                    |
| Dense Retrieval  | FAISS Cosine Similarity |
| Fusion           | RRF                     |
| Reranker         | BAAI/bge-reranker-large |
| LLM              | Ollama                  |
| 生成模型             | Qwen2.5-7B              |
| 评估模型             | Qwen3-14B               |
| 语言               | Python 3.11+            |

---

# 3. 项目结构

```text
local-RAG/

│
├── data/
│
│   ├── raw_pdfs/
│   │       原始PDF文件
│   │
│   ├── ocr_output/
│   │       OCR解析结果
│   │
│   ├── chunks/
│   │       文档切片结果
│   │
│   ├── faiss_index/
│   │       FAISS索引
│   │
│   └── eval/
│           评估数据
│
│
├── models/
│       本地模型文件
│
│
├── src/
│
│   ├── ingestion/
│   │
│   │   ├── pdf_loader.py
│   │   ├── ocr_engine.py
│   │   ├── text_cleaner.py
│   │   └── chunker.py
│   │
│   │
│   ├── embedding/
│   │
│   │   └── embedder.py
│   │
│   │
│   ├── vectorstore/
│   │
│   │   └── faiss_store.py
│   │
│   │
│   ├── retrieval/
│   │
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── rrf.py
│   │   ├── reranker.py
│   │   ├── query_transform.py
│   │   ├── pipeline.py
│   │   └── factory.py
│   │
│   │
│   ├── llm/
│   │
│   │   ├── ollama_client.py
│   │   ├── generator.py
│   │   └── evaluator.py
│   │
│   │
│   └── pipeline/
│
│       ├── rag_pipeline.py
│       └── eval_pipeline.py
│
│
├── scripts/
│
│   ├── parse_pdfs.py
│   ├── build_chunks.py
│   ├── build_vectorstore.py
│   ├── chat.py
│   └── evaluate.py
│
│
├── requirements.txt
│
└── README.md
```

---

# 4. 环境安装

## Python环境

推荐：

```bash
Python >=3.11
```

创建虚拟环境：

```bash
python -m venv rag
```

激活：

Windows:

```powershell
rag\Scripts\activate
```

---

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 5. 模型准备

## Embedding模型

模型：

```
BAAI/bge-large-zh-v1.5
```

路径：

例如：

```
~\LLM_modules\bge-large-zh-v1.5
```

用途：

* 文本向量化
* Query embedding

参数：

```
dimension=1024
```

---

## Reranker模型

```
BAAI/bge-reranker-large
```

用途：

对召回结果重新排序。

流程：

```
Retriever Top30
       |
       ↓
Reranker
       |
       ↓
Top5
```

---

## LLM模型

通过 Ollama 部署：

生成模型：

```
qwen2.5:7b
```

评估模型：

```
qwen3:14b
```

启动：

```bash
ollama serve
```

检查：

```bash
ollama list
```

---

# 6. 数据处理流程

## 6.1 PDF OCR

输入：

```
data/raw_pdfs/
```

执行：

```bash
python scripts/parse_pdfs.py
```

输出：

```
data/ocr_output/
```

格式：

```json
{
 "pages":[
    {
      "page_num":1,
      "text":"..."
    }
 ]
}
```

---

# 7. Chunk处理

执行：

```bash
python scripts/build_chunks.py
```

处理流程：

```
OCR JSON

 ↓

文本清洗

 ↓

章节识别

 ↓

语义切分

 ↓

chunks.json
```

默认参数：

```python
chunk_size=800

overlap=150
```

输出：

```
data/chunks/

├── 词学通论/
│      chunks.json
```

---

# 8. 构建向量库

执行：

```bash
python src/pipeline/build_vectorstore.py
```

流程：

```
chunks

 ↓

BGE Embedding

 ↓

FAISS Index
```

输出：

```
data/faiss_index/

├── index.faiss

└── metadata.json
```

---

# 9. Retriever设计

系统支持：

## Dense Retrieval

向量搜索：

```
Query
 ↓
Embedding
 ↓
FAISS
```

---

## BM25

关键词匹配：

适合：

* 人名
* 专业术语
* 固定概念

---

## RRF Fusion

融合：

```
Dense Ranking

+

BM25 Ranking

↓

RRF
```

公式：

[
score(d)=\sum\frac1{k+rank(d)}
]

---

## Multi Query

利用LLM生成多个查询：

例如：

原问题：

```
吴梅如何理解词的气韵？
```

扩展：

```
吴梅词论中的气韵观是什么？

词和曲的气韵有什么区别？

吴梅如何评价词的艺术特点？
```

---

## HyDE

生成假设答案：

```
问题

↓

假设文档

↓

Embedding检索
```

---

## Self Query

LLM生成过滤条件：

例如：

```json
{
 "book":"词学通论",
 "chapter":"绪论"
}
```

---

# 10. RAG生成流程

执行：

```bash
python scripts/chat.py
```

流程：

```
用户问题

↓

Retriever

↓

Top-K Context

↓

Prompt Template

↓

qwen2.5:7b

↓

答案
```

Prompt原则：

* 只使用资料
* 禁止幻觉
* 输出来源

---

# 11. 效果评估

执行：

```bash
python scripts/evaluate.py
```

评估模型：

```
qwen3:14b
```

评估维度：

| 指标                | 说明   |
| ----------------- | ---- |
| relevance         | 相关性  |
| faithfulness      | 忠实度  |
| completeness      | 完整性  |
| citation_quality  | 引用质量 |
| clarity           | 表达质量 |
| retrieval_quality | 检索质量 |
| overall           | 综合评分 |

输出：

```
data/eval/results.json
```

---

# 12. Retriever实验

## Baseline

Dense:

```python
mode="dense"
```

BM25:

```python
mode="bm25"
```

混合：

```python
mode="rrf"
```

---

## 开启Rerank

```python
enable_rerank=True
```

---

## Multi Query

```python
enable_multi_query=True
```

---

## HyDE

```python
enable_hyde=True
```

---

# 13. 当前系统能力总结

目前已经实现：

✅ PDF知识库
✅ OCR解析
✅ 中文文本清洗
✅ 书籍级Chunk
✅ BGE Embedding
✅ FAISS向量库
✅ BM25检索
✅ Dense Retrieval
✅ RRF融合
✅ Query Expansion
✅ HyDE
✅ Self Query
✅ Cross Encoder Rerank
✅ Ollama本地生成
✅ 自动化RAG评估

