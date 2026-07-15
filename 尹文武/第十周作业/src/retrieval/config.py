from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    # dense / bm25 / rrf
    mode: str = "rrf"

    top_k: int = 5

    dense_k: int = 20
    bm25_k: int = 20

    # RRF参数
    rrf_k: int = 60

    # 查询增强
    enable_multi_query: bool = False
    multi_query_num: int = 3

    enable_hyde: bool = True
    enable_self_query: bool = False

    # rerank
    enable_rerank: bool = False
    rerank_top_k: int = 5
    rerank_candidate_k: int = 30
