# 校园答疑智能客服 Campus QA Bot

基于大语言模型与 RAG（检索增强生成）的校园智能问答系统。围绕「查询改写 + 混合检索 + 重排序」三大检索增强能力，结合多轮对话与流式输出，提供低延迟、高命中率的校园答疑服务。

## 功能特性

- **三大检索增强**（独立开关、失败可回退）：
  - **查询改写**：基于 LLM 消解多轮对话中的指代/省略，生成独立完整的检索 query
  - **混合检索**：向量检索（ChromaDB + BGE）与 BM25（jieba + rank_bm25）通过 RRF 融合
  - **重排序**：bge-reranker-base cross-encoder 对候选文档精排
- **多轮对话**：基于 session_id 关联上下文，支持连续追问
- **流式输出**：SSE 实时返回生成内容，首 Token 约 3 秒
- **会话管理**：对话历史查询 / 清除
- **多格式知识导入**：支持 TXT / Markdown / Word / PDF 文档入库
- **优雅降级**：任一增强环节异常均回退，最坏情况退化为纯向量检索，不阻塞主流程

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 大模型 | 小米 MiMo API（mimo-v2.5-pro） |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-small-zh-v1.5 |
| 稀疏检索 | jieba 分词 + rank_bm25 |
| 融合策略 | Reciprocal Rank Fusion (RRF) |
| 重排序 | BAAI/bge-reranker-base (Cross-Encoder) |
| 前端 | 原生 HTML + JavaScript |

## 系统架构

```
用户请求
   │
   ▼
 src/api/chat.py ──► src/core/agent.py (CampusAgent)
                         │
   ┌─────────────────────┼───────────────────────────┐
   │                     │                           │
   ▼                     ▼                           ▼
查询改写              混合检索                     重排序
query_rewriter      hybrid_retriever              reranker
(LLM 指代消解)     (向量 + BM25 + RRF)        (cross-encoder 精排)
   │                     │                           │
   └──────────► 失败任一环节自动回退 ◄─────────────┘
                         │
                         ▼
              上下文组装 + 多轮历史
                         │
                         ▼
                src/core/llm.py (MiMo)
                         │
                         ▼
              流式/同步返回 → 前端
```

## 项目结构

```
agentproject/
├── start.py                        # 一键启动脚本（预加载模型 + FastAPI）
├── src/
│   ├── main.py                     # FastAPI 入口（lifespan 预热模型）
│   ├── api/chat.py                 # 聊天 API 路由（同步 + 流式 SSE + 历史）
│   ├── core/
│   │   ├── agent.py                # Agent 核心：编排改写→检索→重排→生成
│   │   ├── query_rewriter.py       # 查询改写（超时/异常回退原 query）
│   │   ├── hybrid_retriever.py     # 混合检索编排（向量+BM25+RRF+rerank）
│   │   ├── reranker.py             # Cross-Encoder 重排序
│   │   ├── bm25_index.py           # BM25 索引构建与持久化（pickle）
│   │   ├── knowledge_loader.py     # 知识库加载、ChromaDB 构建
│   │   ├── llm.py                  # MiMo LLM API 封装（同步 + 流式）
│   │   ├── conversation.py         # 对话历史管理
│   │   ├── cache.py                # Embedding / Reranker 模型单例缓存
│   │   └── config.py               # config.yaml 配置加载
│   └── frontend/                   # 前端页面（index.html）
├── config/
│   ├── config.yaml                 # 检索增强主配置（三大开关）
│   └── .env.example                # LLM API 密钥示例
├── knowledge_base/
│   ├── faq/                         # FAQ 知识源（Markdown）
│   └── processed/                   # ChromaDB 向量数据 + bm25_index.pkl
├── scripts/import_docs.py          # 文档批量导入工具
├── tests/                           # 单元/集成/端到端测试
└── docs/                            # 开发文档与技术路线图
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 国内网络建议设置 HuggingFace 镜像（启动脚本已内置 `HF_ENDPOINT=https://hf-mirror.com`）。

### 2. 配置环境变量

复制示例文件并填写密钥：

```bash
cp config/.env.example config/.env
```

编辑 `config/.env`：

```env
LLM_PROVIDER=xiaomi
XIAOMI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
XIAOMI_API_KEY=你的API密钥
XIAOMI_MODEL=mimo-v2.5
LLM_MAX_TOKENS=256
LLM_TEMPERATURE=0.5
```

### 3. 构建知识库

将 Markdown 知识文件放入 `knowledge_base/faq/`，然后构建向量库与 BM25 索引：

```bash
python -m src.core.knowledge_loader
```

### 4. 启动服务

```bash
python start.py
```

访问 http://127.0.0.1:8000 打开前端页面，API 文档见 http://127.0.0.1:8000/docs。

## 配置说明

三大检索增强均通过 `config/config.yaml` 的独立开关控制，互不影响：

```yaml
retriever:
  top_k: 3                  # 最终送 LLM 的上下文条数
  top_n: 10                 # 混合检索召回数（rerank 候选池）
  relevance_threshold: 0.6  # distance 过滤阈值

  enable_rewrite: true       # 查询改写开关（失败/超时回退原 query）
  rewrite:
    timeout: 1.2
    max_history_pairs: 3

  enable_hybrid: true        # 混合检索开关（向量 + BM25 + RRF）
  hybrid:
    vector_top_n: 10
    bm25_top_n: 10
    rrf_k: 60

  enable_rerank: true        # 重排序开关（模型不可用回退 RRF 结果）
  rerank:
    model_name: BAAI/bge-reranker-base
    top_k: 5
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/chat` | 同步聊天 |
| POST | `/api/chat/stream` | 流式聊天（SSE，`data: [DONE]` 结束） |
| GET | `/api/history/{session_id}` | 获取对话历史 |
| DELETE | `/api/history/{session_id}` | 清除对话历史 |

请求体示例：

```json
{
  "message": "图书馆几点开门？",
  "session_id": null,
  "top_k": 2
}
```

## 知识库管理

### 修改 FAQ 内容

```bash
# 1. 编辑 knowledge_base/faq/*.md
# 2. 重建索引（项目根目录执行）
python src/core/knowledge_loader.py
# 3. 重启服务
python start.py
```

### 导入外部文档

支持 TXT / Markdown / Word / PDF 格式：

```bash
# 单文件导入
python scripts/import_docs.py my_faq.txt 选课指南

# 批量导入目录
python scripts/import_docs.py --batch ./my_docs
```

## 开发文档

- [开发文档](docs/开发文档-校园答疑智能客服AI%20Agent.html)
- [技术路线图](docs/技术路线图-校园答疑智能客服AI%20Agent.html)
- [系统测试说明](docs/系统测试说明.md)
- [项目优化方案](docs/项目优化方案.md)

## 测试

```bash
python -m pytest tests/
```

包含端到端、集成、RAG 增强、重排序与查询改写等测试用例。

## License

本项目仅用于学习与校园内部使用。
