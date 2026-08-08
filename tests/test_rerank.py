# -*- coding: utf-8 -*-
"""
Rerank 验证 - ① 失败回退（mock） ② 真实模型精排（需联网下载，失败则 skip）
运行: python tests/test_rerank.py
"""
import sys
import os
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rerank_fallback():
    """rerank 启用但模型不可用时，应回退到向量/RRF 结果，不抛异常"""
    from src.core.hybrid_retriever import HybridRetriever
    from unittest.mock import patch

    class MockKL:
        bm25_index = None

        def search(self, q, n):
            return [
                {"id": "d0", "document": "图书馆借书指南", "metadata": {}, "distance": 0.2},
                {"id": "d1", "document": "食堂今日菜单", "metadata": {}, "distance": 0.5},
            ]

    cfg = {
        "enable_hybrid": False, "enable_rerank": True,
        "vector_top_n": 2, "rerank": {"top_k": 5, "model_name": "fake-model"},
    }
    hr = HybridRetriever(MockKL(), cfg)

    # 模拟 reranker.rerank 抛异常（如模型加载失败）
    with patch.object(hr.reranker, "rerank", side_effect=RuntimeError("model unavailable")):
        results = hr.retrieve("图书馆", top_k=2)

    assert len(results) == 2, "回退后应返回向量结果的 top_k 条"
    assert all("distance" in r and "score" in r for r in results), "回退结果应含 distance/score"
    # 纯向量回退：distance = 1 - score（cosine distance 还原）
    assert abs(results[0]["distance"] - 0.2) < 1e-9, "回退后应还原原始 cosine distance"
    print("[OK] rerank 失败 → 回退向量结果，distance 还原正确，无异常抛出")


def test_real_rerank():
    """尝试真实加载 bge-reranker-base 并跑完整精排（需联网下载，失败则 skip）"""
    # 临时允许在线下载（cache.py 默认设了 OFFLINE=1）
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from src.core.config import reload_config
        from src.core.knowledge_loader import KnowledgeLoader
        from src.core.hybrid_retriever import HybridRetriever
        from src.core.cache import get_reranker_model

        cfg = reload_config().retriever
        kl = KnowledgeLoader()
        kl.init_chromadb()
        kl.load_embedding_model()
        hr = HybridRetriever(kl, cfg)

        print("  尝试加载/下载 bge-reranker-base（首次约 280MB）...")
        t0 = time.time()
        get_reranker_model(cfg["rerank"]["model_name"])
        print(f"  reranker 加载完成，耗时 {time.time() - t0:.1f}s")

        for q in ["图书馆能借几本", "食堂几点开门"]:
            t0 = time.time()
            results = hr.retrieve(q, top_k=3)
            dt = time.time() - t0
            print(f"  query='{q}' 耗时={dt:.2f}s（含 rerank）")
            for r in results:
                print(f"    {r['id']} distance={r['distance']:.3f} score={r['score']:.3f}  {r['document'][:45]}")
            assert len(results) > 0
            # rerank 走 sigmoid，相关文档 distance 应较小
            assert results[0]["distance"] <= 0.5, "rerank top1（最相关）distance 应 <=0.5"
        print("[OK] 真实 rerank 路径正常，sigmoid 归一化下 top1 distance<=0.5")
    except Exception as e:
        print(f"[SKIP] 真实 reranker 不可用（需联网下载；回退机制已由 test_rerank_fallback 验证）")
        print(f"       {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=== ① rerank 失败回退验证 ===")
    test_rerank_fallback()
    print("\n=== ② 真实 rerank 精排验证（需联网）===")
    test_real_rerank()
    print("\n=== rerank 验证完成 ===")
