"""
重排序模块 - 基于 bge-reranker-base cross-encoder 的精排

对混合检索召回的候选文档做 cross-encoder 精排，提升最终送入 LLM 的上下文质量。
模型复用 cache.py 的全局单例（get_reranker_model），不在本模块内 new。
"""
from typing import List, Dict

from src.core.cache import get_reranker_model


class Reranker:
    """Cross-Encoder 重排序器，复用全局单例模型。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """
        对候选文档精排。

        Args:
            query: 检索 query（改写后的）
            documents: 候选文档文本列表
            top_k: 精排后保留的条数

        Returns:
            [{index, score, rank}]，按 score 降序，截断 top_k。
            index 指向输入 documents 的下标，score 为 cross-encoder logit（越大越相关）。
        """
        if not documents:
            return []
        ce = get_reranker_model(self.model_name)
        pairs = [(query, doc) for doc in documents]
        # predict 一次性批量打分，避免逐条开销
        scores = ce.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            results.append({
                "index": idx,
                "score": float(score),
                "rank": rank,
            })
        return results

    @property
    def model(self):
        """获取底层模型（用于启动时预热）。"""
        return get_reranker_model(self.model_name)
