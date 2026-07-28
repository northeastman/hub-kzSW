"""
简化版工具集 - 供 ReAct Agent 调用
"""

import os
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from openai import OpenAI

# 路径配置
BASE_DIR = Path(__file__).parent.parent.resolve()
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Embedding 客户端（注意：需要设置环境变量 DASHSCOPE_API_KEY）
_embed_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 公司映射
COMPANY_MAP = {
    "贵州茅台": "600519", "茅台": "600519",
    "五粮液": "000858",
    "宁德时代": "300750",
    "中国平安": "601318", "平安": "601318",
    "海康威视": "002415", "海康": "002415",
}

# RAG 工具
_faiss_index = None
_faiss_meta = None

def _load_rag():
    global _faiss_index, _faiss_meta
    if _faiss_index is not None:
        return
    original_cwd = os.getcwd()
    try:
        os.chdir(VECTORSTORE_DIR)
        _faiss_index = faiss.read_index("faiss_index.bin")
        with open("faiss_meta.json", encoding="utf-8") as f:
            _faiss_meta = json.load(f)
    except Exception as e:
        _faiss_index = None
        _faiss_meta = None
    finally:
        os.chdir(original_cwd)

def _embed_query(text):
    try:
        resp = _embed_client.embeddings.create(model="text-embedding-v3", input=[text])
        vec = np.array(resp.data[0].embedding, dtype="float32")
        return vec / np.linalg.norm(vec)
    except Exception as e:
        return None

def tool_rag_search(query, top_k=5):
    """在年报中语义检索"""
    try:
        _load_rag()
        if _faiss_index is None:
            return "RAG索引未加载，请确保 vectorstore 目录存在"
        vec = _embed_query(query)
        if vec is None:
            return "Embedding 调用失败，请检查 API Key"
        scores, indices = _faiss_index.search(vec.reshape(1, -1), top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0:
                continue
            meta = _faiss_meta[idx]
            results.append(f"[{rank}] {meta.get('stock_code','')} {meta.get('year','')}年报 第{meta.get('page_num','')}页\n{meta['content']}")
        return "\n\n".join(results) if results else "未检索到相关内容"
    except Exception as e:
        return f"rag_search 执行出错: {e}"

def tool_company_lookup(name):
    """公司名称转股票代码"""
    code = COMPANY_MAP.get(name.strip())
    if code:
        return f"{name} 的股票代码为 {code}"
    candidates = [k for k in COMPANY_MAP if name in k]
    if candidates:
        return "相似公司：" + "、".join(f"{k}({COMPANY_MAP[k]})" for k in candidates)
    return f"未找到 '{name}'，支持：{'、'.join(COMPANY_MAP.keys())}"

def tool_calculator(expr):
    """数学计算器"""
    _SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    _SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)
        return str(round(float(result), 6))
    except Exception as e:
        return f"计算出错: {e}"

def tool_financial_indicator(symbol):
    """获取财务指标"""
    try:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol=symbol)
        if df is None or df.empty:
            return f"未获取到 {symbol} 的财务数据"
        date_cols = [c for c in df.columns if str(c).endswith("1231")][:3]
        if not date_cols:
            date_cols = df.columns[2:5].tolist()
        target_rows = ["归母净利润", "营业总收入", "毛利率", "净利率", "净资产收益率"]
        lines = [f"股票代码: {symbol}"]
        for _, row in df.iterrows():
            label = str(row.get("指标", ""))
            if any(t in label for t in target_rows):
                vals = []
                for col in date_cols:
                    v = row.get(col)
                    try:
                        v = f"{float(v):.4g}"
                    except:
                        v = str(v)
                    vals.append(f"{col[:4]}年: {v}")
                lines.append(f"  {label}: {' | '.join(vals)}")
        return "\n".join(lines) if len(lines) > 1 else f"未找到 {symbol} 的财务指标"
    except Exception as e:
        return f"financial_indicator 执行出错: {e}"

def tool_stock_price(symbol, start_date, end_date):
    """获取历史股价"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is None or df.empty:
            return f"未获取到 {symbol} 在 {start_date}~{end_date} 的行情数据"
        first_close = float(df.iloc[0]["收盘"])
        last_close = float(df.iloc[-1]["收盘"])
        change_pct = (last_close - first_close) / first_close * 100
        return f"{symbol} {start_date}~{end_date}\n起始: {first_close:.2f} | 末尾: {last_close:.2f} | 涨跌幅: {change_pct:+.2f}%"
    except Exception as e:
        return f"stock_price 执行出错: {e}"

# 工具注册表
TOOLS_MAP: dict[str, Any] = {
    "rag_search":          tool_rag_search,
    "company_lookup":      tool_company_lookup,
    "calculator":          tool_calculator,
    "financial_indicator": tool_financial_indicator,
    "stock_price":         tool_stock_price,
}