from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from ai_agent.config.modeling import get_chat_model
from ai_agent.config.runtime import get_checkpointer, get_store
from ai_agent.config.settings import get_settings
from ai_agent.services.common import extract_last_ai_text, stream_graph_text, thread_config
from ai_agent.tools.builtin import get_manus_tools


MANUS_SYSTEM_PROMPT = """You are an all-capable AI assistant, aimed at solving any task presented by the user.
You have various tools at your disposal that you can call upon to efficiently complete complex requests.
When the task is complete, provide a concise final answer and stop calling tools."""


class ManusService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = get_chat_model()
        self.agent = create_agent(
            model=self.model,
            # 只要给 create_agent() 一个可调用工具列表，agent 就具备了 ReAct 的基础条件
            tools=get_manus_tools(),
            system_prompt=MANUS_SYSTEM_PROMPT
            + f"\nTry to finish within {self.settings.manus_max_steps} tool-using steps.",
            middleware=[
                # 长上下文对话压缩
                SummarizationMiddleware(
                    model=self.model,
                    trigger=("messages", 16),
                    keep=("messages", 10),
                )
            ],
            checkpointer=get_checkpointer(),
            store=get_store(),
            name="general-agent",
        )

    def run(self, message: str, thread_id: str = "default") -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            # 和love_app的类似，会话状态天然隔离
            config=thread_config(f"manus:{thread_id}"),
        )
        return extract_last_ai_text(result)

    async def run_stream(self, message: str, thread_id: str = "default"):
        # 复用common.py的流式适配逻辑
        async for text in stream_graph_text(self.agent, message, f"manus:{thread_id}"):
            yield text

# 第一次请求时创建 service，后续请求复用同一个实例
# 意味着 Manus agent 的工具列表、模型对象、store/checkpointer 都是进程内复用的
@lru_cache(maxsize=1)
def get_manus_service() -> ManusService:
    return ManusService()
