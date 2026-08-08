# -*- coding: utf-8 -*-
"""
端到端验证 - 多轮对话中的查询改写 + 混合检索（RRF）
临时关闭 rerank（环境无法下载 reranker 模型），聚焦验证改写与混合检索。
改写真实调用 MiMo LLM；混合检索走向量+BM25+RRF。
运行: python tests/test_e2e.py
"""
import sys
import os
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # 临时关闭 rerank（环境无法下载 reranker），聚焦验证改写 + 混合检索
    import src.core.config as cfgmod
    cfg = cfgmod.reload_config()
    cfg.retriever["enable_rerank"] = False
    print(f"配置: rewrite={cfg.retriever['enable_rewrite']}, "
          f"hybrid={cfg.retriever['enable_hybrid']}, rerank={cfg.retriever['enable_rerank']}")

    from src.core.agent import CampusAgent
    from src.core.conversation import conversation_manager

    print("\n初始化 CampusAgent（加载 embedding，跳过 reranker）...")
    t0 = time.time()
    agent = CampusAgent()
    print(f"初始化完成，耗时 {time.time() - t0:.1f}s")

    sid = "e2e_rewrite_test"
    # 手动清理残留（避免触发 conversation_manager.clear_history 的既有 _delete_file 缺失 bug）
    conv_file = os.path.join("data", "conversations", f"{sid}.json")
    if os.path.exists(conv_file):
        os.remove(conv_file)
    conversation_manager.conversations.pop(sid, None)

    # 多轮指代场景：验证改写能否消解「那」「它」
    convs = [
        "食堂几点开门？",       # 首轮，无历史，应原样检索
        "那图书馆呢？",         # 指代「开门时间」+ 省略，应改写为「图书馆几点开门」
        "它可以借多少本书？",   # 指代「图书馆」+ 省略，应改写为「图书馆可以借多少本书」
    ]

    print("\n=== 多轮对话检索验证 ===")
    for i, q in enumerate(convs):
        t0 = time.time()
        # _retrieve 内部会打印改写日志（[retrieve] 改写: ...）
        results = agent._retrieve(q, sid, top_k=3)
        dt = time.time() - t0
        print(f"\n[轮{i+1}] Q: {q}  (检索耗时 {dt:.2f}s)")
        for r in results:
            print(f"   {r['id']} distance={r['distance']:.3f}  {r['document'][:50]}")
        # 模拟历史保存，让下一轮改写能看到上下文
        conversation_manager.add_message(sid, "user", q)
        conversation_manager.add_message(sid, "assistant", "(参考上文)")

    if os.path.exists(conv_file):
        os.remove(conv_file)
    conversation_manager.conversations.pop(sid, None)
    print("\n=== 端到端验证完成 ===")
    print("注：若上方出现 '[retrieve] 改写: ...' 日志，说明查询改写真实生效；")
    print("    若改写超时/失败则自动回退原 query（回退机制已由单元测试验证）。")


if __name__ == "__main__":
    main()
