"""
混合检索编排模块 - 向量检索 + BM25 + RRF 融合 + Rerank 精排

这是检索链路的中枢，统一管理「向量检索 → BM25 → RRF 融合 → rerank 精排」，
并通过开关与失败回退保证最坏情况下退化为纯向量检索（等价改造前现状）。

输出统一为 [{id, document, metadata, distance, score}]：
- distance：越小越相关（兼容 agent._format_context 的 `distance > 阈值` 过滤逻辑）
- score：越大越相关（调试/日志用）

distance 与 score 的语义对齐（最大风险点）：
- 走 rerank：distance = sigmoid(rerank_logit)，正 logit（相关）→ distance < 0.5
- 仅 RRF（无 rerank）：归一化到 [0, 0.5]（信任 RRF 排序，不因无绝对意义过度过滤）
- 纯向量：distance = 原始 cosine distance
"""
import math
import logging
from typing import List, Dict

from src.core.reranker import Reranker

logger = logging.getLogger("campus_rag")


class HybridRetriever:
    """混合检索编排器。"""

    def __init__(self, knowledge_loader, config: Dict):
        self.knowledge_loader = knowledge_loader
        self.config = config or {}
        self.enable_hybrid = bool(self.config.get("enable_hybrid", True))
        self.enable_rerank = bool(self.config.get("enable_rerank", True))

        hybrid_cfg = self.config.get("hybrid", {})
        self.vector_top_n = int(hybrid_cfg.get("vector_top_n", 10))
        self.bm25_top_n = int(hybrid_cfg.get("bm25_top_n", 10))
        self.rrf_k = int(hybrid_cfg.get("rrf_k", 60))

        rerank_cfg = self.config.get("rerank", {})
        self.rerank_top_k = int(rerank_cfg.get("top_k", 5))

        self.reranker = Reranker(rerank_cfg.get("model_name", "BAAI/bge-reranker-base"))

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        """
        主入口：返回统一结构的检索结果，已按相关性降序、截断到 top_k。
        """
        # —— 1. 向量检索（基础路，必定执行）——
        vec_results = self._vector_search(query, self.vector_top_n)
        if not vec_results:
            return []

        candidates = vec_results
        used_rrf = False
        used_rerank = False

        # —— 2. 混合检索（BM25 + RRF 融合）——
        if self.enable_hybrid:
            try:
                bm25_results = self._bm25_search(query, self.bm25_top_n)
                if bm25_results:
                    candidates = self._rrf_fuse(vec_results, bm25_results, self.rrf_k)
                    used_rrf = True
            except Exception as e:  # noqa: BLE001 - 混合检索失败必须兜底
                logger.warning("[hybrid] BM25/RRF 失败，回退向量结果: %s", e)
                candidates = vec_results

        # —— 3. 重排序（cross-encoder 精排）——
        if self.enable_rerank:
            try:
                candidates = self._rerank(query, candidates, self.rerank_top_k)
                used_rerank = True
            except Exception as e:  # noqa: BLE001 - rerank 失败必须兜底
                logger.warning("[hybrid] rerank 失败，回退 RRF/向量排序: %s", e)

        # —— 4. 截断到调用方要求的 top_k ——
        candidates = candidates[:top_k]

        # —— 5. 统一计算 distance（与 score 语义对齐）——
        self._attach_distance(candidates, used_rerank, used_rrf)
        return candidates

    # ---------- 子检索路 ----------

    def _vector_search(self, query: str, top_n: int) -> List[Dict]:
        """向量检索，返回 [{id, document, metadata, score}]，score=1-distance（越大越相关）。"""
        results = self.knowledge_loader.search(query, top_n)
        out = []
        for r in results:
            distance = float(r["distance"])
            out.append({
                "id": r["id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": 1.0 - distance,  # cosine 相似度
            })
        # 显式按 score 降序，保证 RRF 的 rank 计算正确
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def _bm25_search(self, query: str, top_n: int) -> List[Dict]:
        """BM25 检索，返回 [{id, document, metadata, score}]，score 越大越相关。"""
        bm25_index = getattr(self.knowledge_loader, "bm25_index", None)
        if bm25_index is None or len(bm25_index) == 0:
            return []
        results = bm25_index.search(query, top_n)
        return [
            {
                "id": r["id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": float(r["score"]),
            }
            for r in results
        ]

    def _rrf_fuse(self, vec_results: List[Dict], bm25_results: List[Dict], k: int) -> List[Dict]:
        """
        Reciprocal Rank Fusion：RRF_score(d) = Σ 1/(k + rank_r(d))，rank 0-based。
        按 id 合并两路，输出按 rrf_score 降序。
        """
        fused: Dict[str, Dict] = {}
        for results in (vec_results, bm25_results):
            for rank, r in enumerate(results):
                doc_id = r["id"]
                if doc_id not in fused:
                    fused[doc_id] = {
                        "id": doc_id,
                        "document": r["document"],
                        "metadata": r["metadata"],
                        "score": 0.0,
                    }
                fused[doc_id]["score"] += 1.0 / (k + rank)
        return sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    def _rerank(self, query: str, candidates: List[Dict], top_k_rerank: int) -> List[Dict]:
        """对候选做 cross-encoder 精排，score 替换为 rerank logit。"""
        documents = [c["document"] for c in candidates]
        ranked = self.reranker.rerank(query, documents, top_k_rerank)
        out = []
        for r in ranked:
            c = candidates[r["index"]]
            out.append({
                "id": c["id"],
                "document": c["document"],
                "metadata": c["metadata"],
                "score": float(r["score"]),  # rerank logit，越大越相关
            })
        return out

    # ---------- distance 归一化 ----------

    def _attach_distance(self, candidates: List[Dict], used_rerank: bool, used_rrf: bool):
        """
        根据最终排序依据计算 distance（越小越相关），写入每条 candidate。
        - 走 rerank：sigmoid(rerank_logit)，正 logit → distance < 0.5（相关保留）
        - 仅 RRF：1 - minmax_norm(rrf_score)
        - 纯向量：还原原始 cosine distance（1 - score）
        """
        if not candidates:
            return
        if used_rerank:
            for c in candidates:
                c["distance"] = 1.0 / (1.0 + math.exp(c["score"]))
        elif used_rrf:
            # RRF score 无绝对相关性意义，归一化到 [0, 0.5]：
            # 最高分 distance=0，最低分 distance=0.5，均低于 relevance_threshold(0.6)，
            # 信任 RRF 排序，避免 min-max 极端化导致阈值过度过滤 top_k 结果。
            scores = [c["score"] for c in candidates]
            smin, smax = min(scores), max(scores)
            if smax == smin:
                for c in candidates:
                    c["distance"] = 0.0
            else:
                for c in candidates:
                    norm = (c["score"] - smin) / (smax - smin)
                    c["distance"] = 0.5 * (1.0 - norm)
        else:
            # 纯向量：score = 1 - cosine_distance，还原
            for c in candidates:
                c["distance"] = 1.0 - c["score"]
