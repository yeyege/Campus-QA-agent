# -*- coding: utf-8 -*-
"""临时验证：大 timeout 下查询改写真实效果（确认改写 prompt + 解析逻辑正确）"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if False else ".")

from src.core.llm import get_llm_client
from src.core.query_rewriter import QueryRewriter

# 大 timeout（8s）仅用于验证改写逻辑，生产用 1.2s
rw = QueryRewriter(get_llm_client(), {
    "rewrite": {"timeout": 8.0, "max_history_pairs": 3, "max_tokens": 64, "temperature": 0.0}
})

history = [
    {"role": "user", "content": "食堂几点开门？"},
    {"role": "assistant", "content": "早餐6:30-9:00，午餐11:00-13:00"},
]

cases = ["那图书馆呢？", "它可以借多少本书？"]
for q in cases:
    t0 = time.time()
    out = rw.rewrite(q, history)
    dt = time.time() - t0
    changed = "改写" if out != q else "回退原query"
    print(f"[{changed}] '{q}' -> '{out}'  ({dt:.2f}s)")
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": "(参考上文)"})
