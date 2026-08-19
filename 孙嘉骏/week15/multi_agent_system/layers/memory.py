# layers/memory.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re

@dataclass
class Memory:
    """记忆层，管理短期对话历史和长期记忆"""
    short_term: List[Dict[str, str]] = field(default_factory=list)   # 短期消息列表
    long_term: List[Dict[str, Any]] = field(default_factory=list)    # 长期记忆条目

    def add_message(self, message: Dict[str, str]):
        """添加一条对话消息（role: user/assistant/tool）"""
        self.short_term.append(message)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """返回当前对话历史（复制）"""
        return self.short_term.copy()

    def clear_short_term(self):
        self.short_term = []

    def add_long_term(self, content: str, metadata: Optional[Dict] = None):
        """写入长期记忆"""
        self.long_term.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": __import__("datetime").datetime.now().isoformat()
        })

    def query_long_term(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        简单的关键词匹配检索长期记忆。
        实际生产环境可替换为向量检索（如 ChromaDB、FAISS）。
        """
        if not self.long_term:
            return []
        # 提取查询中的关键词
        keywords = set(re.findall(r'[\u4e00-\u9fa5A-Za-z0-9_]+', query.lower()))
        scored = []
        for item in self.long_term:
            content = item["content"].lower()
            score = sum(1 for kw in keywords if kw in content)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def add_subagent_result(self, task_description: str, result: str):
        """将子Agent结果写入长期记忆，便于后续引用"""
        self.add_long_term(
            content=f"子任务：{task_description}\n结果：{result}",
            metadata={"type": "subagent_result"}
        )