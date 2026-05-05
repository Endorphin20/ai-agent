from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse

from ai_agent.models.schemas import LoveReport
# 下面这两行已经把要注入的get_love_app_service和get_manus_service导入进来了，所以后面可以直接Depends调用
from ai_agent.services.love_app import LoveAppService, get_love_app_service
from ai_agent.services.manus import ManusService, get_manus_service

# 如果说 main.py 是应用入口，那么 routes.py 就是“HTTP 接口层”（controller层），负责把外部请求翻译成内部 service 调用
# 项目里的两类agent：垂直领域agent(love_app), 通用工具agent(manus)

# 业务模块路由初始化，后面会被 main.py 挂载，统一加前缀 /api（可以点开看一下，能看到有个prefix，对应的就是 /api）
router = APIRouter()


# 服务存活探针，运维/网关用
@router.get("/health/check")
def health_check() -> str:
    return "ok"


# 同步对话，等待全部生成完再返回
@router.get("/ai/love_app/chat/sync")
def do_chat_with_love_app_sync(
    message: str = Query(...),
    thread_id: str = Query(default="default"),
    # FastAPI 的依赖注入，不让路由自己 LoveAppService()，而是交给框架管理依赖创建
    love_app: LoveAppService = Depends(get_love_app_service),
):
    return love_app.chat(message, thread_id)


# 报告生成，多了一个结构化检验的参数，用response_model=LoveReport做HTTP输出校验，和service层的校验配合形成双保险链路
@router.get("/ai/love_app/chat/report", response_model=LoveReport)
def do_chat_with_love_app_report(
    message: str = Query(...),
    thread_id: str = Query(default="default"),
    love_app: LoveAppService = Depends(get_love_app_service),
):
    return love_app.report(message, thread_id)


# RAG知识库对话，会查本地文档/向量库
@router.get("/ai/love_app/chat/rag")
def do_chat_with_love_app_rag(
    message: str = Query(...),
    thread_id: str = Query(default="default"),
    # 检索时可以按照用户状态status过滤知识库（单身、恋爱、已婚）
    status: str | None = Query(default=None),
    love_app: LoveAppService = Depends(get_love_app_service),
):
    return love_app.chat_with_rag(message, thread_id, status)

'''
LLM 生成文字片段
   ↓
async for chunk 拿到片段
   ↓
yield 把片段推出去
   ↓
StreamingResponse / EventSourceResponse 接住
   ↓
通过 SSE 协议实时发给前端
'''

# 普通 AI 聊天 + 手动拼接 SSE 格式
@router.get("/ai/love_app/chat/sse")
async def do_chat_with_love_app_sse(
    message: str = Query(...),
    thread_id: str = Query(default="default"),
    love_app: LoveAppService = Depends(get_love_app_service),
):
    # 定义一个异步生成器，不断往外吐数据
    async def stream():
        # 当前实现依赖 FastAPI 的全局异常处理器，因此路由层没有重复写 try/except。不过 SSE 属于长连接流式场景，流开始后再抛异常的处理会比普通 JSON 接口更复杂，因此生产级实现通常会在流生成器内部再做一层异常兜底。
        # try:
        # 异步迭代：从 love_app 拿一个个消息片段
        async for chunk in love_app.chat_stream(message, thread_id):
            # 【关键】手动拼 SSE 标准格式：data: 内容\n\n
            # 流式输出底层用yield，不同于return等很久后一次性返回，yield可以实时返回给前端
            yield f"data: {chunk}\n\n"

    # 返回原生的FastAPI StreamingResponse流式响应（FastAPI 原生，通用流，底层手动版）
    return StreamingResponse(stream(), media_type="text/event-stream")


# 和上面功能完全一样，但用封装好的 SSE 工具，代码更干净
@router.get("/ai/love_app/chat/server_sent_event")
async def do_chat_with_love_app_server_sent_event(
    message: str = Query(...),
    thread_id: str = Query(default="default"),
    love_app: LoveAppService = Depends(get_love_app_service),
):
    async def events():
        async for chunk in love_app.chat_stream(message, thread_id):
            # 不用拼字符串！直接yield返回字典
            yield {"data": chunk}

    # 直接用 EventSourceResponse（专门给 SSE 用，自动处理格式、重连、心跳，封装优雅版）
    return EventSourceResponse(events())


# LangGraph Agent，流式输出 思考过程 + 最终回答
@router.get("/ai/manus/chat")
async def do_chat_with_manus(
    message: str = Query(...),
    thread_id: str = Query(default="default"),
    manus: ManusService = Depends(get_manus_service),
):
    async def events():
        async for chunk in manus.run_stream(message, thread_id):
            yield {"data": chunk}

    return EventSourceResponse(events())
