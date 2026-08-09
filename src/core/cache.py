"""
缓存模块 - 全局模型缓存和检索结果缓存
"""
import os
from functools import lru_cache
from typing import List, Dict

# 设置HuggingFace镜像（优先国内镜像，下载更快）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _load_model_offline_first(loader_func, model_name: str, model_type: str):
    """
    模型加载通用策略：
    1. 先尝试离线模式加载（使用本地缓存，速度快）
    2. 离线失败则切换到在线模式，从镜像源下载并缓存
    3. 在线也失败则抛出异常，由上层处理降级

    Args:
        loader_func: 实际的模型加载函数，接受 model_name 参数
        model_name: 模型标识
        model_type: 用于日志显示，如 "Embedding" / "Reranker"
    """
    # 阶段1：尝试离线加载
    original_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    try:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"[{model_type}] 尝试离线加载: {model_name}")
        model = loader_func(model_name)
        print(f"[{model_type}] 离线加载成功（使用缓存）")
        return model
    except Exception as offline_err:
        print(f"[{model_type}] 离线加载失败，尝试在线下载: {offline_err}")

    # 阶段2：切换到在线模式下载
    try:
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        print(f"[{model_type}] 从 {os.environ.get('HF_ENDPOINT', 'hf-mirror.com')} 下载模型: {model_name}")
        model = loader_func(model_name)
        print(f"[{model_type}] 在线下载成功（已缓存）")
        return model
    except Exception as online_err:
        print(f"[{model_type}] 在线下载也失败: {online_err}")
        raise
    finally:
        # 恢复原始离线模式设置
        if original_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = original_offline
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        if original_hf_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = original_hf_offline
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


class ModelCache:
    """Embedding模型单例缓存"""
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, model_name: str = "BAAI/bge-small-zh-v1.5"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self._model_name = model_name

    def get_model(self):
        """获取模型（懒加载）"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            def _loader(name):
                return SentenceTransformer(name)

            print(f"首次加载Embedding模型: {self._model_name}")
            self._model = _load_model_offline_first(_loader, self._model_name, "Embedding")
            print("模型加载完成")
        return self._model

    @property
    def model(self):
        return self.get_model()


# 全局模型缓存实例
_embedding_cache = None


def get_embedding_model():
    """获取全局Embedding模型"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = ModelCache()
    return _embedding_cache.model


class RerankerCache:
    """Reranker 模型单例缓存（镜像 ModelCache 实现）"""
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, model_name: str = "BAAI/bge-reranker-base"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self._model_name = model_name

    def get_model(self):
        """获取模型（懒加载）"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            def _loader(name):
                return CrossEncoder(name)

            print(f"首次加载Reranker模型: {self._model_name}")
            self._model = _load_model_offline_first(_loader, self._model_name, "Reranker")
            print("Reranker模型加载完成")
        return self._model

    @property
    def model(self):
        return self.get_model()


# 全局 Reranker 缓存实例
_reranker_cache = None


def get_reranker_model(model_name: str = "BAAI/bge-reranker-base"):
    """获取全局 Reranker 模型（bge-reranker-base cross-encoder）"""
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = RerankerCache(model_name)
    return _reranker_cache.model


# 检索结果LRU缓存
@lru_cache(maxsize=128)
def cached_search(query: str, top_k: int = 3) -> tuple:
    """缓存检索结果"""
    from src.core.knowledge_loader import KnowledgeLoader
    loader = KnowledgeLoader()
    loader.init_chromadb()
    loader.load_embedding_model()

    results = loader.search(query, top_k)
    # 转换为tuple以便缓存
    return tuple(str(r) for r in results)


def get_cached_search_results(query: str, top_k: int = 3) -> List[dict]:
    """获取缓存的检索结果"""
    cached = cached_search(query, top_k)
    # 重新转换为dict列表
    import ast
    results = []
    for item in cached:
        results.append(ast.literal_eval(item))
    return results


# 清除缓存
def clear_cache():
    """清除所有缓存（含模型单例与检索结果缓存）"""
    global _embedding_cache, _reranker_cache
    _embedding_cache = None
    _reranker_cache = None
    # 重置类级单例状态，确保下次 get_* 能重新加载
    ModelCache._instance = None
    ModelCache._model = None
    RerankerCache._instance = None
    RerankerCache._model = None
    cached_search.cache_clear()
    print("缓存已清除")
