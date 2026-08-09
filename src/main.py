# -*- coding: utf-8 -*-
"""
校园答疑智能客服 - FastAPI主入口
"""
import os
import sys

# 注意：HF_ENDPOINT 由 cache.py 在模块导入时统一设置
# TRANSFORMERS_OFFLINE 由 cache.py 中的加载策略动态切换（先离线再在线）
# 此处不强制设置离线模式，避免首次启动无法下载缺失模型

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app):
    """启动时预加载模型"""
    print("正在预加载 Embedding 模型...")
    from src.core.cache import get_embedding_model
    get_embedding_model()
    print("Embedding 模型加载完成")

    # 条件预加载 Reranker（避免首问冷启动；失败不阻塞，重排序将自动回退）
    try:
        from src.core.config import get_config
        from src.core.cache import get_reranker_model
        retriever_cfg = get_config().retriever
        if retriever_cfg.get("enable_rerank", True):
            print("正在预加载 Reranker 模型...")
            rerank_cfg = retriever_cfg.get("rerank", {})
            get_reranker_model(rerank_cfg.get("model_name", "BAAI/bge-reranker-base"))
            print("Reranker 模型加载完成")
    except Exception as e:
        print(f"Reranker 预加载失败（重排序将回退）: {e}")

    yield


app = FastAPI(
    title="校园答疑智能客服 API",
    description="基于RAG的校园答疑系统API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

# 前端目录
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


@app.get("/")
async def root():
    """返回前端页面"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# 挂载静态文件（CSS/JS/图片等）
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    print("启动服务: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
