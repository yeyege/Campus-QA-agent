"""
Agent核心模块 - 多轮对话支持
"""
import os
import re
from typing import List, Dict, Generator, Optional, Tuple

from src.core.llm import get_llm_client
from src.core.knowledge_loader import KnowledgeLoader
from src.core.conversation import conversation_manager
from src.core.config import get_config
from src.core.hybrid_retriever import HybridRetriever
from src.core.query_rewriter import QueryRewriter

# Prompt模板
SYSTEM_PROMPT = """你是校园智能助手"小智"，专注于校园知识问答。

回答规则：
1. 如果有参考资料，基于资料简洁回答
2. 如果没有资料，用你的知识回答基础常识问题（如1+1=2、北京是首都）
3. 实时信息问题（如今天日期、天气）：说明无法获取，建议查看手机
4. 超出范围问题：礼貌引导回校园话题
5. 直接给出答案，不要输出思考过程

语气：友好亲切，像学长学姐"""


class CampusAgent:
    def __init__(self, llm_provider: str = None):
        self.llm_client = get_llm_client(llm_provider)
        self.knowledge_loader = KnowledgeLoader()
        self.knowledge_loader.init_chromadb()
        self.knowledge_loader.load_embedding_model()

        # 检索增强配置（查询改写 / 混合检索 / 重排序）
        self._retriever_config = get_config().retriever
        self._enable_rewrite = bool(self._retriever_config.get("enable_rewrite", True))
        self._relevance_threshold = float(self._retriever_config.get("relevance_threshold", 0.6))

        # 混合检索编排器（向量 + BM25 + RRF + rerank，含失败回退）
        self._hybrid_retriever = HybridRetriever(self.knowledge_loader, self._retriever_config)

        # 查询改写器（条件实例化）
        if self._enable_rewrite:
            self._query_rewriter = QueryRewriter(self.llm_client, self._retriever_config)
        else:
            self._query_rewriter = None

        # 预热 reranker（条件；失败不阻塞，重排序将自动回退）
        if bool(self._retriever_config.get("enable_rerank", True)):
            try:
                from src.core.cache import get_reranker_model
                rerank_cfg = self._retriever_config.get("rerank", {})
                get_reranker_model(rerank_cfg.get("model_name", "BAAI/bge-reranker-base"))
            except Exception as e:
                print(f"[agent] Reranker 预加载失败（重排序将回退）: {e}")

        # 检索缓存
        self._search_cache = {}

    @staticmethod
    def _parse_answer(text: str) -> str:
        """从模型输出中提取答案部分，去掉思考过程"""
        # 优先提取 "答案：" 后面的内容
        for sep in ["答案：", "答案:"]:
            if sep in text:
                return text.split(sep, 1)[1].strip()
        # 没有 "答案："，去掉所有 "思考：..." 行
        text = re.sub(r"^思考[：:].*$", "", text, flags=re.MULTILINE).strip()
        return text

    @staticmethod
    def _extract_images(text: str) -> Tuple[str, List[str]]:
        """从文本中提取图片URL，返回清理后的文本和图片列表"""
        # 匹配 Markdown 图片格式: ![alt](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        images = re.findall(image_pattern, text)
        # 移除图片标记，只保留文本
        clean_text = re.sub(image_pattern, '', text).strip()
        # 清理多余的空行
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        return clean_text, [url for alt, url in images]

    def _retrieve(self, question: str, session_id: str, top_k: int = 2) -> List[dict]:
        """
        检索入口：查询改写（可选）→ 缓存 → 混合检索（向量+BM25+RRF+rerank）。
        改写后的 query 仅用于检索；任何增强环节失败均回退，不阻塞主流程。
        """
        # 1. 查询改写（仅多轮有意义；改写失败/超时回退原 question）
        query = question
        if self._query_rewriter is not None:
            history = conversation_manager.get_history(session_id)
            query = self._query_rewriter.rewrite(question, history)
            if query != question:
                print(f"[retrieve] 改写: '{question}' -> '{query}'")

        # 2. 缓存（基于改写后的 query，避免历史变化时错误命中旧改写）
        cache_key = f"{query}_{top_k}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        # 3. 混合检索
        results = self._hybrid_retriever.retrieve(query, top_k)
        self._search_cache[cache_key] = results
        return results

    def _format_context(self, search_results: List[dict]) -> Tuple[str, List[str]]:
        """格式化检索结果"""
        if not search_results:
            return "无相关资料", []

        parts = []
        all_images = []
        for r in search_results:
            if r["distance"] > self._relevance_threshold:
                continue
            doc = r["document"]
            clean, images = self._extract_images(doc)
            parts.append(clean)
            all_images.extend(images)

        if not parts:
            return "无相关资料", []

        context = "\n---\n".join(parts)
        return context, all_images

    def _build_messages_with_history(self, session_id: str, question: str, context: str) -> List[Dict]:
        """构建带历史的消息"""
        # 获取历史对话
        history = conversation_manager.get_history(session_id)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 添加历史对话（最近4轮）
        recent = history[-8:] if len(history) > 8 else history
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 添加当前带上下文的问题
        user_message = f"资料：{context}\n问：{question}\n答："
        messages.append({"role": "user", "content": user_message})

        return messages

    def chat(self, question: str, session_id: str = "default", top_k: int = 2) -> Dict:
        """多轮对话聊天，返回包含文本和图片的字典"""
        # 检索知识库（含查询改写 + 混合检索 + 重排序）
        search_results = self._retrieve(question, session_id, top_k)
        context, images = self._format_context(search_results)

        # 构建带历史的消息
        messages = self._build_messages_with_history(session_id, question, context)

        # 调用LLM
        response = self.llm_client.chat(messages, max_tokens=1024)
        answer = self._parse_answer(response)

        # 保存对话历史
        conversation_manager.add_message(session_id, "user", question)
        conversation_manager.add_message(session_id, "assistant", answer)

        return {
            "text": answer,
            "images": images
        }

    def chat_stream(self, question: str, session_id: str = "default", top_k: int = 2) -> Generator[Dict, None, None]:
        """多轮对话流式聊天，返回包含文本和图片的字典"""
        full_answer = ""
        images = []
        try:
            # 检索知识库（含查询改写 + 混合检索 + 重排序）
            search_results = self._retrieve(question, session_id, top_k)
            context, images = self._format_context(search_results)

            # 构建带历史的消息
            messages = self._build_messages_with_history(session_id, question, context)

            # 流式调用LLM — 只 yield 正式回答，跳过思考过程
            for chunk_type, chunk_text in self.llm_client.chat_stream(messages, max_tokens=1024):
                if chunk_type == "answer":
                    full_answer += chunk_text
                    yield {"type": "text", "content": chunk_text}

            # 兜底：如果流式过程没有 yield 任何 answer（模型全放在 reasoning 里），用同步接口取答案
            if not full_answer:
                response = self.llm_client.chat(messages, max_tokens=1024)
                answer = self._parse_answer(response)
                if answer:
                    full_answer = answer
                    yield {"type": "text", "content": answer}

            # 如果有图片，在文本流结束后发送图片
            if images:
                yield {"type": "images", "content": images}

        except GeneratorExit:
            pass
        finally:
            # 保存对话历史（即使客户端断开也保存已生成的内容）
            if full_answer:
                conversation_manager.add_message(session_id, "user", question)
                conversation_manager.add_message(session_id, "assistant", full_answer)

    def get_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return conversation_manager.get_history(session_id)

    def clear_history(self, session_id: str):
        """清除对话历史"""
        conversation_manager.clear_history(session_id)
        # 同时清除该session的检索缓存
        keys_to_delete = [k for k in self._search_cache.keys()]
        for key in keys_to_delete:
            del self._search_cache[key]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    import time

    agent = CampusAgent()
    session_id = "test_session"

    # 模拟多轮对话
    conversations = [
        "食堂几点开门？",
        "那图书馆呢？",
        "它可以借多少本书？",
    ]

    for q in conversations:
        start = time.time()
        result = agent.chat(q, session_id)
        print(f"Q: {q}")
        print(f"A: {result['text']}")
        if result['images']:
            print(f"图片: {result['images']}")
        print(f"时间: {time.time() - start:.2f}秒\n")

    # 显示完整历史
    print("=== 对话历史 ===")
    history = agent.get_history(session_id)
    for msg in history:
        print(f"[{msg['role']}] {msg['content'][:50]}...")
