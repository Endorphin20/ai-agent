from functools import lru_cache
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import create_retriever_tool

from ai_agent.config.modeling import get_chat_model, get_embeddings
from ai_agent.config.runtime import get_checkpointer, get_store
from ai_agent.config.settings import get_settings
from ai_agent.knowledge.base import KnowledgeBase
from ai_agent.models.schemas import LoveReport
from ai_agent.services.common import (
    extract_last_ai_text,
    extract_messages_from_state,
    stream_graph_text,
    thread_config,
)

# 垂直领域的 agent，“恋爱顾问”这个垂直 AI 应用的业务编排层。
# 亮点1:RAG 不是简单硬编码，而是做了 query rewrite 和工具化检索
# 亮点2:对长对话引入了摘要中间件，考虑了 token 成本

# LOVE_SYSTEM_PROMPT 定义了“恋爱顾问”的角色边界：让它围绕单身、恋爱、已婚三种状态提问，并引导用户讲清背景。
LOVE_SYSTEM_PROMPT = (
    "扮演深耕恋爱心理领域的专家。开场向用户表明身份，告知用户可倾诉恋爱难题。"
    "围绕单身、恋爱、已婚三种状态提问：单身状态询问社交圈拓展及追求心仪对象困扰；"
    "恋爱状态询问沟通、习惯差异引发的矛盾；已婚状态询问家庭责任与亲情关系处理的问题。"
    "引导用户详述事情经过、对方反应及自身想法，以便给出专属解决方案。"
)

# LOVE_RAG_SYSTEM_PROMPT 则是在前者基础上，强制先检索再答。这是 Agentic RAG 很典型的控制方式
LOVE_RAG_SYSTEM_PROMPT = (
    LOVE_SYSTEM_PROMPT
    + "回答知识库问题时，必须至少调用一次知识库检索工具。"
    + "如果知识库没有相关内容，要明确告诉用户你只能回答恋爱相关知识。"
)

# 初始化阶段
class LoveAppService:
    def __init__(self) -> None:
        # 第一，拿配置和基础组件
        self.settings = get_settings()
        self.model = get_chat_model()
        self.checkpointer = get_checkpointer()
        self.store = get_store()
        # 第二，初始化知识库
        self.knowledge_base = KnowledgeBase(self.settings.docs_dir, embeddings=get_embeddings())
        # 第三，初始化 query rewriter
        self.query_rewriter = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", "请把用户问题改写成更适合知识库检索的一句话，只输出改写结果。"),
                    ("human", "{query}"),
                ]
            )
            | self.model
            | StrOutputParser()
        )
        # 第四，初始化对话摘要中间件，当消息累计到 12 条时，就触发摘要压缩，只保留 8 条左右的上下文并配合摘要继续对话。
        # 它解决的是多轮对话越来越长、token 成本越来越高的问题
        self.middleware = [
            SummarizationMiddleware(
                model=self.model,
                trigger=("messages", 12),
                keep=("messages", 8),
            )
        ]
        # 第五，创建两个常驻 agent，一个用于普通聊天，一个用于结构化报告
        # 这里非常关键：这个 service 不是只有一个 agent，而是至少有两类固定 agent，另加一个按需动态创建的 RAG agent
        # 普通聊天agent，本质上是一个“有记忆的纯聊天 agent”，没有工具调用
        self.chat_agent = create_agent(
            model=self.model,
            system_prompt=LOVE_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            store=self.store,
            middleware=self.middleware,
            name="love-chat",
        )
        self.report_agent = create_agent(
            model=self.model,
            system_prompt=LOVE_SYSTEM_PROMPT + "每次回答都要产出结构化恋爱报告。",
            # 多了一个参数，限制输出必须符合LoveReport结构
            response_format=LoveReport,
            checkpointer=self.checkpointer,
            store=self.store,
            middleware=self.middleware,
            name="love-report",
        )

    # thread_id 用来隔离上下文，避免不同agent的会话串台，这叫“通过 thread namespace 避免不同 agent 状态互相污染”
    # 普通聊天
    def _chat_thread(self, thread_id: str) -> str:
        return f"love-chat:{thread_id}"

    def _report_thread(self, thread_id: str) -> str:
        return f"love-report:{thread_id}"

    def _rag_thread(self, thread_id: str) -> str:
        return f"love-rag:{thread_id}"

    def _get_chat_history(self, thread_id: str):
        try:
            snapshot = self.chat_agent.get_state(thread_config(self._chat_thread(thread_id)))
        except Exception:
            return []
        return extract_messages_from_state(snapshot)

    # 纯聊天，输入格式是标准消息列表，不是裸字符串
    def chat(self, message: str, thread_id: str) -> str:
        result = self.chat_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=thread_config(self._chat_thread(thread_id)),
        )
        return extract_last_ai_text(result)

    # service层 只表达“我要流式聊天”，流式协议细节被下沉到common.py实现
    async def chat_stream(self, message: str, thread_id: str):
        async for text in stream_graph_text(self.chat_agent, message, self._chat_thread(thread_id)):
            yield text

    # 上面两个都用的self.chat_agent，这个用的self.report_agent
    def report(self, message: str, thread_id: str) -> LoveReport:
        # 先拿聊天历史，输出更像“诊断报告”，而不是一次独立问答
        # 注意取的是 chat_agent 的历史，而不是 report_agent 自己的历史
        # 也就是chat_agent先产生聊天过程，然后report_agent来做结构化整理
        history = self._get_chat_history(thread_id)
        result = self.report_agent.invoke(
            {"messages": [*history, {"role": "user", "content": message}]},
            config=thread_config(self._report_thread(thread_id)),
        )
        # 前面self.report_agent那里定义了输出格式，这里可以直接返回LoveReport结构的数据；而且routes.py的controller层还有一层结构化校验，双保险链路
        return result["structured_response"]

    # 根据status动态构建rag_agent，单独拆出来
    # 注意，这里的 RAG 不是传统“先检索，再把文档拼进 prompt”的固定链，而是“给 agent 一个检索工具，让 agent 自己决定如何使用”的工具型 RAG
    # 即使用的是agentic RAG，而不是静态拼接上下文的 RAG chain
    def _build_rag_agent(self, status: Optional[str] = None):
        # 先通过知识库得到一个retriever
        retriever = self.knowledge_base.as_retriever(status=status, top_k=3)
        # 再转成工具，把 RAG retriever 检索当成一个工具来调用
        retriever_tool = create_retriever_tool(
            retriever=retriever,
            name="search_love_knowledge",
            description="Search the love knowledge base before answering user questions.",
        )
        return create_agent(
            model=self.model,
            # 检索工具
            tools=[retriever_tool],
            # 用RAG的系统提示词
            system_prompt=LOVE_RAG_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            store=self.store,
            middleware=self.middleware,
            name="love-rag",
        )

    def chat_with_rag(self, message: str, thread_id: str, status: Optional[str] = None) -> str:
        # 第一步，query 改写，改写之后更好检索，更容易命中关键词或语义相近片段，提升检索质量
        rewritten = self.query_rewriter.invoke({"query": message}).strip()
        status_hint = f"\n优先状态：{status}" if status else ""
        # 第二步，构造增强提示，把“原问题 + 改写结果 + 行为要求”一起交给 agent。这样 agent 既保留用户原意，也知道推荐检索词是什么
        rag_message = (
            f"原始问题：{message}\n"
            f"检索改写：{rewritten}\n"
            f"请务必先用知识库工具检索，再根据知识库内容回答。{status_hint}"
        )
        # 第三步，动态创建 RAG agent，RAG agent 不是在初始化时固定建好，而是每次按 status 参数动态构建，因为 retriever 可能需要带不同过滤条件
        rag_agent = self._build_rag_agent(status=status)
        # 第四步，带聊天历史调用 agent
        result = rag_agent.invoke(
            # 也复用了普通聊天的历史，可以保留之前用户已经交代过的背景。
            {"messages": [*self._get_chat_history(thread_id), {"role": "user", "content": rag_message}]},
            config=thread_config(self._rag_thread(thread_id)),
        )
        return extract_last_ai_text(result)


@lru_cache(maxsize=1)
def get_love_app_service() -> LoveAppService:
    return LoveAppService()
