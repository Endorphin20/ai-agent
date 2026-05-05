from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 这里还是内存实现，如果上生产，通常要把 checkpointer/store 换成外部持久化方案

@lru_cache(maxsize=1)
def get_checkpointer():
    return InMemorySaver()


@lru_cache(maxsize=1)
def get_store():
    return InMemoryStore()
