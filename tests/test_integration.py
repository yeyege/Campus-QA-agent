# -*- coding: utf-8 -*-
"""
集成验证 - BM25 自愈重建 + 混合检索（向量+BM25+RRF）
暂关闭 rerank 以避免下载 reranker 大模型，单独验证混合检索链路。
运行: python tests/test_integration.py
"""
import sys
import os
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.core.config import reload_config
    from src.core.knowledge_loader import KnowledgeLoader
    from src.core.hybrid_retriever import HybridRetriever

    # 关掉 rerank，仅验证向量+BM25+RRF（避免下载 reranker 模型）
    cfg = dict(reload_config().retriever)
    cfg["enable_rerank"] = False
    print(f"配置: enable_hybrid={cfg['enable_hybrid']}, enable_rerank={cfg['enable_rerank']}")

    # 1. 初始化 ChromaDB（触发 BM25 自愈重建）
    print("\n[1] 初始化 KnowledgeLoader（应触发 BM25 自愈重建）...")
    kl = KnowledgeLoader()
    kl.init_chromadb()
    bm25_len = len(kl.bm25_index) if kl.bm25_index else 0
    print(f"    BM25 索引条数: {bm25_len}")
    assert bm25_len > 0, "BM25 自愈重建失败，索引为空"

    # 验证 Chroma 与 BM25 条数一致
    ids, docs, metas = kl.get_all_documents()
    print(f"    Chroma 文档条数: {len(ids)}")
    assert len(ids) == bm25_len, f"Chroma({len(ids)}) 与 BM25({bm25_len}) 条数不一致"
    print("[OK] BM25 自愈重建成功，与 ChromaDB 条数对齐")

    # 2. BM25 单独检索
    print("\n[2] BM25 检索 '图书馆借书' ...")
    res = kl.bm25_index.search("图书馆借书", top_k=3)
    for r in res:
        print(f"    {r['id']} score={r['score']:.3f}  {r['document'][:50]}")
    assert len(res) > 0, "BM25 检索无结果"

    # 3. 混合检索（向量+BM25+RRF）
    print("\n[3] 混合检索（向量+BM25+RRF）...")
    kl.load_embedding_model()
    hr = HybridRetriever(kl, cfg)

    queries = ["图书馆借书", "食堂几点开门", "图书馆能借几本"]
    for q in queries:
        t0 = time.time()
        results = hr.retrieve(q, top_k=3)
        dt = time.time() - t0
        print(f"\n  query='{q}'  耗时={dt:.2f}s  结果数={len(results)}")
        for r in results:
            print(f"    {r['id']} distance={r['distance']:.3f}  {r['document'][:50]}")
        assert all("distance" in r and "score" in r for r in results), "结果缺少 distance/score 字段"

    # 4. 验证 distance 语义：相关查询的 top1 distance 应较小
    print("\n[4] 验证 distance 语义（越小越相关）...")
    r0 = hr.retrieve("图书馆借书", top_k=3)
    if r0:
        assert r0[0]["distance"] <= r0[-1]["distance"], "top1 distance 应 <= 末位 distance"
        print(f"[OK] top1 distance({r0[0]['distance']:.3f}) <= 末位({r0[-1]['distance']:.3f})")

    # 5. 验证 BM25 pkl 已生成
    pkl_path = os.path.join("knowledge_base", "processed", "bm25_index.pkl")
    assert os.path.exists(pkl_path), f"BM25 pkl 未生成: {pkl_path}"
    print(f"\n[5] [OK] BM25 索引已持久化: {pkl_path}")

    print("\n=== 集成验证通过（向量+BM25+RRF 链路正常，rerank 未启用）===")


if __name__ == "__main__":
    main()
