"""
配置加载模块 - 统一加载 config.yaml，提供全局单例配置访问

注意：.env 中的 API 密钥等仍由各模块通过 os.getenv 读取（保持向后兼容），
本模块主要负责 retriever 三项增强（查询改写/混合检索/重排序）相关的参数加载。
"""
import os
from copy import deepcopy
from typing import Any, Dict

# 项目根目录（与 llm.py 保持一致的三级上溯）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认配置：当 config.yaml 缺失某字段时使用，保证旧配置文件不崩
_DEFAULT_CONFIG: Dict[str, Any] = {
    "llm": {
        "provider": "xiaomi",
        "model": "mimo-v2.5",
        "temperature": 0.5,
        "max_tokens": 1024,
    },
    "vector_db": {
        "collection_name": "campus_faq",
        "persist_directory": "knowledge_base/processed",
    },
    "retriever": {
        "top_k": 3,                 # 最终送 LLM 的上下文条数
        "top_n": 10,                # 混合检索召回数（rerank 候选池）
        "relevance_threshold": 0.6, # distance 过滤阈值（越小越相关，兼容现有 0.6）
        "enable_rewrite": True,
        "rewrite": {
            "max_tokens": 64,
            "temperature": 0.0,
            "timeout": 1.2,         # 秒，超时回退原 query
            "max_history_pairs": 3, # 取最近 N 轮历史进 prompt
        },
        "enable_hybrid": True,
        "hybrid": {
            "vector_top_n": 10,
            "bm25_top_n": 10,
            "rrf_k": 60,            # RRF 常数
            "bm25_index_path": "knowledge_base/processed/bm25_index.pkl",
        },
        "enable_rerank": True,
        "rerank": {
            "model_name": "BAAI/bge-reranker-base",
            "top_n": 10,            # 参与 rerank 的候选数
            "top_k": 5,             # rerank 后保留数（应 >= retriever.top_k）
        },
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
    },
}


def _deep_merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并：override 覆盖 default，仅对 dict 类型递归合并。"""
    result = deepcopy(default)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class AppConfig:
    """应用配置，按段提供属性访问。"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @property
    def llm(self) -> Dict[str, Any]:
        return self._data.get("llm", {})

    @property
    def vector_db(self) -> Dict[str, Any]:
        return self._data.get("vector_db", {})

    @property
    def retriever(self) -> Dict[str, Any]:
        return self._data.get("retriever", {})

    @property
    def api(self) -> Dict[str, Any]:
        return self._data.get("api", {})

    def get(self, section: str, default: Any = None) -> Any:
        return self._data.get(section, default)


_config: AppConfig = None


def get_config(config_path: str = None) -> AppConfig:
    """
    获取全局配置单例。
    首次调用时读取 config/config.yaml 并与默认值深度合并。
    """
    global _config
    if _config is None:
        if config_path is None:
            config_path = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
        data: Dict[str, Any] = {}
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[config] 读取 config.yaml 失败，使用默认配置: {e}")
                data = {}
        merged = _deep_merge(_DEFAULT_CONFIG, data)
        _config = AppConfig(merged)
    return _config


def reload_config(config_path: str = None) -> AppConfig:
    """强制重新加载配置（主要用于测试）。"""
    global _config
    _config = None
    return get_config(config_path)
