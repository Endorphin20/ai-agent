import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict, Field

# 项目的本地知识库实现，核心目的：把 data/ 目录里的 Markdown 文档，变成一个可以被 agent 检索的本地知识库

# 从文件名推断文档类型，打上标签
def _infer_status(file_name: str) -> str:
    if "单身" in file_name:
        return "单身"
    if "恋爱" in file_name:
        return "恋爱"
    if "已婚" in file_name:
        return "已婚"
    return "未知"

# 把文本转成token集合
def _tokenize(text: str) -> set[str]:
    lowered = text.lower()
    # 全部转小写，这样英文检索时大小写就不敏感了，按正则切分token，提取英文、数字、下划线组成的“词”
    latin_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    # 没有接中文分词库，而是用“把每个汉字拿出来”的方法提取中文字符，轻量但能用
    cjk_tokens = {char for char in lowered if "\u4e00" <= char <= "\u9fff"}
    # 把英文单词 + 中文单字取并集，合并成一个集合返回，最终结果完全去重
    return latin_tokens | cjk_tokens


class KnowledgeBase:
    # 没向量数据库也能跑，纯关键词检索照样跑（优雅降级）
    def __init__(self, docs_dir: Path, embeddings: Any | None = None) -> None:
        # 从目录里读取文档并切分文档（Extract+Transform），得到切块后的 Document 列表
        self.documents = self._load_documents(docs_dir)
        # 先准备一个向量库属性，默认没有
        self.vector_store: InMemoryVectorStore | None = None
        # 为每一个文档块提前算好 token 集合，存在内存里，以后每次搜索时，不用重新给每个文档分词，直接拿预计算结果比较就行
        self.doc_tokens = [_tokenize(doc.page_content) for doc in self.documents]
        # Load 加载
        # 如果有 embedding：走向量检索 + 关键词检索的混合召回策略
        if embeddings is not None and self.documents:
            self.vector_store = InMemoryVectorStore(embedding=embeddings)
            self.vector_store.add_documents(self.documents)

    # ETL工程，加载并切分文档
    def _load_documents(self, docs_dir: Path) -> list[Document]:
        raw_documents: list[Document] = []
        # Extract 抽取
        for path in sorted(docs_dir.glob("*.md")):
            # 给每个文档带上filename+status这两个元数据
            raw_documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"filename": path.name, "status": _infer_status(path.name)},
                )
            )
        # Transform 转换
        # 按800字符切分，重叠120
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        return splitter.split_documents(raw_documents)

    # 检索核心逻辑，采用的是“融合检索”思路，但实现得很轻量，分两段：
    def search(self, query: str, status: Optional[str] = None, top_k: int = 3) -> list[Document]:
        merged: list[Document] = [] # 最终结果列表
        seen: set[tuple[str, int]] = set() # 去重用，因为同一个文档块可能被向量检索和关键词检索同时命中，不去重回重复

        # 内部add函数
        def add(documents: Iterable[Document]) -> None:
            for doc in documents:
                # 如果用户传了 status，那文档块必须匹配这个状态，否则跳过；比如你搜“怎么扩大社交圈”，并且 status="单身"，那只会保留单身篇里的块
                if status and doc.metadata.get("status") != status:
                    continue
                # 用 (文件名 + 内容hash) 做唯一键，根据这个去重
                key = (doc.metadata.get("filename", ""), hash(doc.page_content))
                if key in seen: # 已经加过就跳过
                    continue
                seen.add(key)
                merged.append(doc)
        # 第一段，如果有向量数据库，就先做一轮语义检索；用向量相似度找语义最接近的段落，召回多一点（2 倍），后面再精排
        if self.vector_store is not None:
            add(self.vector_store.similarity_search(query, k=max(top_k * 2, top_k)))

        # 第二段，再做本地 token overlap 打分
        query_tokens = _tokenize(query) # 把问题query分词
        scored: list[tuple[float, Document]] = [] # 遍历所有文档chunk
        for doc, doc_tokens in zip(self.documents, self.doc_tokens): # 把文档块对象和对应的token集合一一配对，像拉拉链一样
            # 先按状态筛掉不相关的块
            if status and doc.metadata.get("status") != status:
                continue
            # 核心评分逻辑：
            overlap = len(query_tokens & doc_tokens) # 重叠的token长度
            if overlap == 0: # 没有交集
                continue
            score = overlap / math.sqrt(len(doc_tokens) + 1) # 归一化，得到得分
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True) # 按分数排序
        add([doc for _, doc in scored[:top_k]]) # 把这轮检索得到的结果加入结果集merged中
        return merged[:top_k] # 返回前top_k条

    # 把知识库包装成retriever
    def as_retriever(self, status: Optional[str] = None, top_k: int = 3) -> "KnowledgeRetriever":
        return KnowledgeRetriever(knowledge_base=self, status=status, top_k=top_k)


class KnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_base: KnowledgeBase
    status: Optional[str] = None
    top_k: int = Field(default=3)

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return self.knowledge_base.search(query=query, status=self.status, top_k=self.top_k)
