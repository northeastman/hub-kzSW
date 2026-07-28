# ReAct Financial Agent - 简化版

## 项目结构

```
第12周作业/
├── src/
│   ├── agent.py          # 统一入口
│   ├── react_manual.py   # 手写Prompt解析版ReAct
│   ├── react_chat.py     # 多轮对话模式
│   └── tools.py          # 工具集
└── vectorstore/          # FAISS向量索引（需自行复制）
```

## 使用方式

### 1. 安装依赖
```bash
pip install openai faiss-cpu akshare numpy
```

### 2. 单轮模式
```bash
cd src
python agent.py
python agent.py --question "贵州茅台2023年毛利率是多少？"
```

### 3. 多轮对话模式
```bash
python agent.py --chat
```

## 工具列表

| 工具名 | 功能 |
|--------|------|
| company_lookup | 公司名称转股票代码 |
| financial_indicator | 获取财务指标（毛利率、ROE等） |
| stock_price | 获取历史股价及涨跌幅 |
| calculator | 数学表达式计算 |
| rag_search | 年报语义检索 |

## 环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key"  # RAG检索需要
export DEEPSEEK_API_KEY="your-api-key"   # LLM推理需要
```

## 示例问题

- "贵州茅台和五粮液2023年毛利率对比"
- "茅台2023年股价涨跌幅"
- "宁德时代的ROE是多少"