"""
BM25 索引模块 - 基于 jieba 分词 + rank_bm25 的关键词检索

提供与向量检索互补的关键词路检索，用于混合检索（Hybrid Retrieval）。
索引为内存态，支持 pickle 持久化；load 时不反序列化 BM25Okapi 对象，
而是用 tokenized_corpus 重建，避免版本兼容问题。
"""
import os
import pickle
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 关键词检索索引（内存态，支持 pickle 持久化）"""

    def __init__(self, index_path: str = "knowledge_base/processed/bm25_index.pkl"):
        self.index_path = index_path
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        """构建索引：对全部文档 jieba 分词后建立 BM25Okapi。"""
        self.ids = list(ids)
        self.documents = list(documents)
        self.metadatas = list(metadatas)
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索相关文档。
        返回 [{id, document, metadata, score, rank}]，score 越大越相关，按 score 降序。
        rank 为 0-based 名次，供 RRF 融合使用。
        """
        if self.bm25 is None or not self.ids:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # 按分数降序排序
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            results.append({
                "id": self.ids[idx],
                "document": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(score),
                "rank": rank,
            })
        return results

    def save(self):
        """持久化到 index_path（不存 BM25Okapi 对象，load 时重建）。"""
        dir_path = os.path.dirname(os.path.abspath(self.index_path))
        os.makedirs(dir_path, exist_ok=True)
        data = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> bool:
        """加载并重建 BM25Okapi；失败返回 False。"""
        if not os.path.exists(self.index_path):
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            return True
        except Exception as e:
            print(f"[bm25] 加载索引失败: {e}")
            self.bm25 = None
            return False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """jieba 分词，过滤空白 token。"""
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]

    def __len__(self):
        return len(self.ids)
