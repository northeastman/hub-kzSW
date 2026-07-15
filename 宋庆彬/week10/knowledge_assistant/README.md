# Knowledge Assistant

问答项目，运行方式改为终端交互。

## 模块

- `core`: 配置和日志
- `faq_retrieval`: MySQL FAQ、Redis 缓存、BM25 检索
- `rag_retrieval`: Milvus 混合检索、问题分类、策略选择、RAG 生成
- `conversation`: MySQL 会话历史
- `ingestion`: 文档加载、切分、入库
- `llm`: DashScope/OpenAI-compatible 模型客户端
- `orchestrator`: FAQ 与 RAG 的顶层编排

## 配置

复制示例配置并填写密钥：

```bash
cp knowledge_assistant.example.toml knowledge_assistant.toml
```

模型路径默认指向原项目，不会移动大模型文件。

也可以用环境变量覆盖配置：

```bash
export KA_LLM_API_KEY="你的 DashScope API Key"
```

## 初始化数据

导入 FAQ CSV：

```bash
python -m knowledge_assistant.manage import-faq /Users/songqingbin/PycharmProjects/rag_learn/integrated_qa_system/mysql_qa/data/JP学科知识问答.csv
```

摄取文档到 Milvus：

```bash
python -m knowledge_assistant.manage ingest-docs /Users/songqingbin/PycharmProjects/rag_learn/integrated_qa_system/rag_qa/data/ai_data
```

## 命令行问答

```bash
python -m knowledge_assistant.cli
```

可选指定知识来源：

```bash
python -m knowledge_assistant.cli --source ai
```

运行时命令：

- `/history`: 查看最近会话历史
- `/clear`: 清空当前会话历史
- `/exit`: 退出
