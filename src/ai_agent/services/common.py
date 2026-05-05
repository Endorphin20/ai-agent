from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

# service层的公共胶水
# 它把 LangGraph 偏底层、偏框架化的数据结构，统一适配成项目业务层能直接消费的形式

# 本质上是一个小包装器，把会话 ID 变成 LangGraph 需要的配置格式；service里到处都要传这个结构（可以点开看看），封装一下减少样板代码
def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


# 这个函数是做“统一文本抽取”，把项目当前需要的“可展示文本”稳定提取出来
def normalize_content(content: Any) -> str:
    # 如果本来就是 str，直接返回
    if isinstance(content, str):
        return content
    # 如果是 list，就把其中的文本片段拼起来
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    # 其他类型，就兜底转成字符串
    return str(content or "")


# 从 agent 返回结果里，倒序找到最后一条 AI 消息，然后抽取文本（AI Agent标准约定，最后一条AI消息就是用户想要的最终回答，前面可能是AI思考、工具消息等结果）
# 所以只取最后一条消息在99%的情况下是没问题的
# 下面这个分情况设计是踩过无数坑之后，经过大量实践才知道这么写的。这是为了提高Agent系统的稳定性，因为Agent返回的消息可能有两种格式（AI天生不可控，不确定是哪种）：
# LangChain 原生 AIMessage（这是严格类型，标准情况）、被序列化 / 转成字典 / 简化后的消息（另一种常见形式）
def extract_last_ai_text(result: dict[str, Any]) -> str:
    # 遍历 result["messages"]，从后往前找
    for message in reversed(result.get("messages", [])):
        # 找到 AIMessage 就取其 content
        if isinstance(message, AIMessage):
            return normalize_content(message.content)
        # 如果不是严格的 AIMessage 类型，但 type == "ai"，也兼容处理
        if getattr(message, "type", "") == "ai":
            return normalize_content(getattr(message, "content", ""))
    return ""


# 从 LangGraph 的 state snapshot 这个短期记忆里把 messages 拿出来
# 因为同一会话中，可能会有report的agent和rag的agent需要复用普通chat的已有聊天记录
# 他们都用到了_get_chat_history，而_get_chat_history这个函数就是调用extract_messages_from_state来获取该会话的短期记忆state的
def extract_messages_from_state(state_snapshot: Any) -> list[BaseMessage]:
    if state_snapshot is None:
        return []
    values = getattr(state_snapshot, "values", None) or {}
    return list(values.get("messages", []))


# chat_stream的真正的流式细节被下沉到此处，将业务逻辑和流式协议细节分开
# service 只表达“我要流式聊天”，common 工具函数负责怎么从 graph 事件里抽出文本 chunk
async def stream_graph_text(graph, message: str, thread_id: str) -> AsyncIterator[str]:
    # 调 graph.astream_events(...)，输入一条 user message，监听所有事件
    async for event in graph.astream_events(
        {"messages": [{"role": "user", "content": message}]},
        config=thread_config(thread_id),
        version="v2",
    ):
        # 只保留 on_chat_model_stream
        if event["event"] != "on_chat_model_stream":
            continue
        # 从事件数据里拿 chunk
        chunk = event["data"].get("chunk")
        # 从 chunk.content 里抽文本
        text = normalize_content(getattr(chunk, "content", ""))
        # 有文本就yield
        if text:
            yield text
