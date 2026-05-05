from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from ai_agent.config.settings import get_settings

# 模型工厂

# 创建全局唯一的聊天模型，全局复用同一实例，避免多次创建浪费资源
@lru_cache(maxsize=1)
def get_chat_model():
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("Missing LLM_API_KEY. Copy .env.example to .env and fill it in.")
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )


# 创建全局唯一的向量模型
@lru_cache(maxsize=1)
def get_embeddings():
    settings = get_settings()
    if not settings.embedding_model:
        return None
    api_key = settings.embedding_api_key or settings.llm_api_key
    if not api_key:
        return None
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        base_url=settings.embedding_base_url or settings.llm_base_url,
    )
