# -*- coding: utf-8 -*-
"""
RAG 增强模块测试 - 验证查询改写/混合检索/重排序的核心逻辑
不依赖真实 LLM 与 reranker 模型（使用 mock）。
运行: python tests/test_rag_enhancement.py
"""
import sys
import os
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    from src.core.config import reload_config
    cfg = reload_config()
    r = cfg.retriever
    assert r["enable_rewrite"] is True, "enable_rewrite 应为 True"
    assert r["enable_hybrid"] is True, "enable_hybrid 应为 True"
    assert r["enable_rerank"] is True, "enable_rerank 应为 True"
    assert r["relevance_threshold"] == 0.6, "relevance_threshold 应为 0.6"
    assert r["rerank"]["model_name"] == "BAAI/bge-reranker-base"
    assert r["hybrid"]["rrf_k"] == 60
    print("[OK] config 加载 + 默认值合并")


def test_bm25():
    from src.core.bm25_index import BM25Index
    tmp = tempfile.mktemp(suffix=".pkl")
    idx = BM25Index(index_path=tmp)
    ids = ["d0", "d1", "d2"]
    docs = ["图书馆开放时间是早上8点", "食堂中午供应午餐", "图书馆可以借阅十本书"]
    metas = [{"q": x} for x in docs]
    idx.build(ids, docs, metas)
    res = idx.search("图书馆借书", top_k=2)
    assert len(res) <= 2
    assert res[0]["id"] in ids
    # 命中图书馆相关（d0/d2）而非食堂 d1
    top_ids = [r["id"] for r in res]
    assert "d1" not in top_ids, "食堂不应排在图书馆相关查询前面"
    # save + load 往返
    idx.save()
    idx2 = BM25Index(index_path=tmp)
    assert idx2.load() is True
    assert len(idx2) == 3
    res2 = idx2.search("图书馆借书", top_k=2)
    assert [r["id"] for r in res2] == top_ids, "load 后检索结果应一致"
    os.remove(tmp)
    print("[OK] BM25 build/search/save/load 往返一致")


def test_rrf():
    from src.core.hybrid_retriever import HybridRetriever

    class MockKL:
        bm25_index = None

        def search(self, q, n):
            return []

    cfg = {"enable_hybrid": False, "enable_rerank": False, "rerank": {"top_k": 5}}
    hr = HybridRetriever(MockKL(), cfg)
    vec = [
        {"id": "d0", "document": "a", "metadata": {}, "score": 0.9},
        {"id": "d1", "document": "b", "metadata": {}, "score": 0.7},
        {"id": "d2", "document": "c", "metadata": {}, "score": 0.5},
    ]
    bm = [
        {"id": "d1", "document": "b", "metadata": {}, "score": 2.0},
        {"id": "d3", "document": "d", "metadata": {}, "score": 1.0},
    ]
    fused = hr._rrf_fuse(vec, bm, k=60)
    ids = [f["id"] for f in fused]
    # d1 在两路都靠前（vec rank1 + bm rank0），RRF 分最高
    assert "d0" in ids and "d1" in ids and "d3" in ids, "融合应包含两路所有 id"
    assert fused[0]["id"] == "d1", "两路都靠前的 d1 应排第一"
    print("[OK] RRF 融合：两路都靠前的文档排名第一")


def test_distance_normalization():
    """关键测试：验证 distance 与 score 语义不反转（防 _format_context 过滤反向）"""
    from src.core.hybrid_retriever import HybridRetriever

    class MockKL:
        bm25_index = None

        def search(self, q, n):
            return []

    cfg = {"enable_hybrid": False, "enable_rerank": False, "rerank": {"top_k": 5}}
    hr = HybridRetriever(MockKL(), cfg)

    # 场景1: rerank，sigmoid，最高分 distance 最小，正 logit → distance<0.5（被保留）
    cands = [{"id": "a", "score": 8.5}, {"id": "b", "score": 2.0}, {"id": "c", "score": -1.0}]
    hr._attach_distance(cands, used_rerank=True, used_rrf=False)
    assert cands[0]["distance"] < cands[1]["distance"] < cands[2]["distance"], "rerank: distance 应随 score 降序递增"
    assert cands[0]["distance"] < 0.5, "正 logit（相关）distance 应 <0.5"
    assert cands[2]["distance"] > 0.5, "负 logit（不相关）distance 应 >0.5"
    print("[OK] distance(rerank): 最高分→distance最小，正logit<0.5（相关保留），负logit>0.5（过滤）")

    # 场景2: rrf only, 归一化到 [0, 0.5]（信任排序，避免阈值过度过滤）
    cands2 = [{"id": "a", "score": 0.1}, {"id": "b", "score": 0.05}, {"id": "c", "score": 0.02}]
    hr._attach_distance(cands2, used_rerank=False, used_rrf=True)
    assert abs(cands2[0]["distance"] - 0.0) < 1e-9, "rrf 最高分 → distance=0"
    assert abs(cands2[2]["distance"] - 0.5) < 1e-9, "rrf 最低分 → distance=0.5"
    assert all(c["distance"] <= 0.5 for c in cands2), "rrf 所有 distance 应 <=0.5（不被 0.6 阈值过滤）"
    print("[OK] distance(rrf): 最高分→0，最低分→0.5，均<=0.5不被过滤")

    # 场景3: 纯向量，还原原始 cosine distance
    cands3 = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
    hr._attach_distance(cands3, used_rerank=False, used_rrf=False)
    assert abs(cands3[0]["distance"] - 0.1) < 1e-9, "纯向量: distance = 1 - score"
    assert abs(cands3[1]["distance"] - 0.5) < 1e-9
    print("[OK] distance(纯向量): 还原 cosine distance")


def test_query_rewriter():
    from src.core.query_rewriter import QueryRewriter
    history = [
        {"role": "user", "content": "图书馆几点开门"},
        {"role": "assistant", "content": "早上8点开门"},
    ]

    # 1. 超时回退
    class SlowLLM:
        def chat(self, messages, temp, max_tokens):
            import time
            time.sleep(2)
            return "图书馆借书"

    rw = QueryRewriter(SlowLLM(), {"rewrite": {"timeout": 0.5, "max_history_pairs": 3}})
    assert rw.rewrite("那借书呢", history) == "那借书呢", "超时应回退原 query"
    print("[OK] QueryRewriter 超时回退原 query")

    # 2. 异常回退
    class ErrLLM:
        def chat(self, *a, **k):
            raise RuntimeError("api error")

    rw2 = QueryRewriter(ErrLLM(), {"rewrite": {"timeout": 2, "max_history_pairs": 3}})
    assert rw2.rewrite("那借书呢", history) == "那借书呢", "异常应回退原 query"
    print("[OK] QueryRewriter 异常回退原 query")

    # 3. 正常改写
    class OkLLM:
        def chat(self, *a, **k):
            return "图书馆借书"

    rw3 = QueryRewriter(OkLLM(), {"rewrite": {"timeout": 2, "max_history_pairs": 3}})
    assert rw3.rewrite("那借书呢", history) == "图书馆借书", "正常应返回改写结果"
    print("[OK] QueryRewriter 正常改写")

    # 4. 无历史不改写
    assert rw3.rewrite("你好", []) == "你好", "无历史应原样返回"
    print("[OK] QueryRewriter 无历史直接返回原 query")

    # 5. 空结果/未变化回退
    class EmptyLLM:
        def chat(self, *a, **k):
            return ""

    rw4 = QueryRewriter(EmptyLLM(), {"rewrite": {"timeout": 2, "max_history_pairs": 3}})
    assert rw4.rewrite("那借书呢", history) == "那借书呢", "空结果应回退"
    print("[OK] QueryRewriter 空结果回退")

    # 6. 思考性输出回退（MiMo no_think 未生效时模型输出推理过程而非改写）
    class ReasoningLLM:
        def chat(self, *a, **k):
            return "首先，用户当前的问题是借书，回顾历史对话，用户先问图书馆..."

    rw5 = QueryRewriter(ReasoningLLM(), {"rewrite": {"timeout": 2, "max_history_pairs": 3}})
    assert rw5.rewrite("那借书呢", history) == "那借书呢", "思考性输出应回退原 query"
    print("[OK] QueryRewriter 思考性输出回退（含推理词/过长）")


if __name__ == "__main__":
    test_config()
    test_bm25()
    test_rrf()
    test_distance_normalization()
    test_query_rewriter()
    print("\n=== 所有模块测试通过 ===")
