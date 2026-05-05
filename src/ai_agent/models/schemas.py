from typing import Optional

from pydantic import BaseModel, Field

# 数据模型层，Pedantic模板，约束HTTP输入输出结构和LLM结构化输出格式

class LoveReport(BaseModel):
    title: str = Field(..., description="恋爱报告标题")
    suggestions: list[str] = Field(..., description="给用户的建议列表")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class RagRequest(BaseModel):
    message: str
    thread_id: str = "default"
    status: Optional[str] = None
