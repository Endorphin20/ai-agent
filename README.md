# ai-agent

- 基于 FastAPI + LangChain + LangGraph 重构并优化原 Spring AI 智能体项目，提供多轮记忆、SSE 流
式输出、结构化报告、RAG 问答与多工具调用能力。
## 技术亮点
- 智能体架构设计：基于 LangGraph 重构智能体运行时，支持任务规划、会话状态管理与多工具协同执行
- RAG 与长上下文优化：结合摘要压缩、Query Rewrite 与混合检索机制，提升长对话与知识库问答效果

## 核心设计

- `LangChain`：`init_chat_model`、`create_agent`、`middleware`、`response_format`
- `LangGraph`：agent runtime、`checkpointer`、`store`
- `FastAPI`：对外暴露 HTTP / SSE 接口

## 功能

- 恋爱顾问同步聊天
- 恋爱顾问流式聊天
- 恋爱报告结构化输出
- RAG 知识库问答
- 多工具通用智能体

## 启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn ai_agent.main:app --reload --host 0.0.0.0 --port 8123 --app-dir src
```

说明：

- 默认不开启 embedding，RAG 会使用本地关键词检索
- 如果后续你有兼容的 embedding 模型，再填写 `EMBEDDING_MODEL`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`
- 如果填写 `ELASTICSEARCH_URL`，知识库会在启动时同步到 `ELASTICSEARCH_INDEX`，检索切换为 Elasticsearch 模式：
  - 未配置 embedding 时使用 Elasticsearch BM25
  - 配置 embedding 时使用 BM25 + kNN 向量召回，并通过 RRF 融合排序
