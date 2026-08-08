"""
查询改写模块 - 基于 LLM 的指代消解与查询补全

在多轮对话中，把用户的省略/指代问题改写为独立、完整的检索 query，
提升向量与 BM25 检索的召回质量。

设计要点：
- 始终启用（由 config.retriever.enable_rewrite 控制）
- 用 no_think 快速模式 + 小 max_tokens + temperature=0 控制延迟
- 超时（默认 1.2s）/异常/空结果 一律回退原 question，绝不抛异常，不阻塞主流程
- 改写后的 query 只用于检索；喂给最终 LLM 的仍是用户原话（在 agent.py 控制）
"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import List, Dict

logger = logging.getLogger("campus_rag")

# 复用轻量线程池（改写是检索前唯一并发点，1 worker 即够）
_executor = ThreadPoolExecutor(max_workers=1)


class QueryRewriter:
    """LLM 查询改写器：把多轮对话中的指代/省略补全为独立检索 query。"""

    REWRITE_SYSTEM_PROMPT = (
        "你是查询改写助手。把用户最后一条消息改写成一个独立、完整、适合检索的中文查询。\n\n"
        "要求：\n"
        "1. 消解指代（它/那个/这个 → 具体实体）\n"
        "2. 补全省略的主语/宾语\n"
        "3. 只输出改写后的查询，不要解释、不要引号、不要多余标点\n"
        "4. 若用户消息已是完整查询，原样输出"
    )

    def __init__(self, llm_client, config: Dict):
        self.llm_client = llm_client
        cfg = (config or {}).get("rewrite", {}) if isinstance(config, dict) else {}
        self.max_tokens = int(cfg.get("max_tokens", 64))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout = float(cfg.get("timeout", 1.2))
        self.max_history_pairs = int(cfg.get("max_history_pairs", 3))

    def rewrite(self, question: str, history: List[Dict]) -> str:
        """
        改写 query；失败/超时/异常一律回退原 question，绝不抛异常。
        无历史时直接返回原 question（无需改写）。
        """
        if not question:
            return question
        if not history:
            return question
        try:
            messages = self._build_rewrite_messages(question, history)
            future = _executor.submit(
                self.llm_client.chat, messages, self.temperature, self.max_tokens
            )
            raw = future.result(timeout=self.timeout)
            rewritten = self._parse_rewrite(raw)
            if not rewritten or rewritten == question:
                # 空/未变化 → 回退
                return question
            # 改写应简短（通常 <30 字）。过长或含推理词说明模型输出了思考过程
            # （如 MiMo no_think 未生效时），回退以保证检索不退化
            if len(rewritten) > 50 or self._looks_like_reasoning(rewritten):
                logger.warning("[rewrite] 改写结果异常（过长或含推理），回退原 query: %s", rewritten[:60])
                return question
            logger.info("[rewrite] '%s' -> '%s'", question, rewritten)
            return rewritten
        except FutureTimeout:
            logger.warning("[rewrite] 超时(%.1fs)，回退原 query: %s", self.timeout, question)
            return question
        except Exception as e:  # noqa: BLE001 - 改写失败必须兜底
            logger.warning("[rewrite] 改写失败，回退原 query: %s", e)
            return question

    def _build_rewrite_messages(self, question: str, history: List[Dict]) -> List[Dict]:
        """构建改写 prompt：历史对话 + 当前问题。"""
        # 取最近 max_history_pairs 轮（每轮 user+assistant = 2 条）
        if self.max_history_pairs > 0:
            recent = history[-(self.max_history_pairs * 2):]
        else:
            recent = []
        lines = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"[用户] {content}")
            elif role == "assistant":
                lines.append(f"[助手] {content}")
        history_block = "\n".join(lines) if lines else "（无）"
        user_content = f"【历史对话】\n{history_block}\n\n【当前问题】\n{question}"
        return [
            {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # 推理性输出标志（MiMo no_think 未生效时模型可能输出思考过程而非改写结果）
    _REASONING_MARKERS = ("首先", "回顾", "历史对话", "用户当前", "用户先问",
                          "分析", "根据", "因此", "所以", "接着", "思考", "让我")

    @classmethod
    def _looks_like_reasoning(cls, text: str) -> bool:
        """检测输出是否为推理/思考过程（而非改写后的查询）"""
        return any(m in text for m in cls._REASONING_MARKERS)

    @staticmethod
    def _parse_rewrite(text: str) -> str:
        """清洗改写结果：去引号/前缀/换行，截断。"""
        if not text:
            return ""
        text = text.strip()
        # 去常见前缀
        for prefix in ["改写后：", "改写后:", "改写：", "改写:", "查询：", "查询:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        # 去首尾引号（中英文）
        quotes = ['"', "'", "“", "”", "「", "」", "『", "』"]
        while text and text[0] in quotes:
            text = text[1:]
        while text and text[-1] in quotes:
            text = text[:-1]
        text = text.strip()
        # 去换行
        text = text.replace("\n", " ").replace("\r", " ").strip()
        return text
