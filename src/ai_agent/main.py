import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import APIError, APIStatusError, OpenAIError, RateLimitError

from ai_agent.api.routes import router
from ai_agent.config.settings import get_settings

# main.py 是应用入口，本质上只做 3 件事：建 FastAPI 应用、注册全局异常处理、挂载业务路由建 FastAPI 应用、注册全局异常处理、挂载业务路由

# 下面这几个函数是我对线上异常可观测性的考虑
# 修复乱码
def _fix_mojibake(value: str) -> str:
    try:
        repaired = value.encode("latin1").decode("utf-8")
        if repaired != value:
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return value


# 递归处理 str、dict、list，把异常响应里的所有字符串都做一次乱码修复
def _normalize(obj: Any) -> Any:
    if isinstance(obj, str):
        return _fix_mojibake(obj)
    if isinstance(obj, dict):
        return {key: _normalize(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    return obj


# 把异常对象整理成结构化 JSON
def _extract_error_details(exc: Exception) -> dict[str, Any]:
    detail: dict[str, Any] = {"message": _fix_mojibake(str(exc))}
    response = getattr(exc, "response", None)
    if response is None:
        return detail

    payload = None
    try:
        payload = response.json()
    except Exception:
        text = getattr(response, "text", "")
        if text:
            try:
                payload = json.loads(text)
            except Exception:
                detail["raw"] = _fix_mojibake(text)

    if isinstance(payload, dict):
        payload = _normalize(payload)
        detail["provider_error"] = payload
        provider_error = payload.get("error", {})
        if isinstance(provider_error, dict):
            code = provider_error.get("code")
            message = provider_error.get("message")
            if code is not None:
                detail["provider_code"] = str(code)
            if message:
                detail["provider_message"] = str(message)
    return detail


settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/")
def index():
    return {
        "name": settings.app_name,
        "status": "ok",
        "docs": "/docs",
        "health": f"{settings.app_context_path}/health/check",
        "chat_sync_example": f"{settings.app_context_path}/ai/love_app/chat/sync?message=你好&thread_id=test",
    }

# 下面是全局异常处理链，注册了 5 类异常处理器；
# 入口层负责把底层 SDK 抛出的异常统一映射成稳定的 HTTP 协议语义，避免把 SDK 细节泄漏到业务接口层。

# RateLimitError -> HTTP 429
@app.exception_handler(RateLimitError)
async def handle_rate_limit_error(_: Request, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"error_type": "rate_limit_error", **_extract_error_details(exc)})


# APIStatusError -> 用供应商原始状态码，默认 502
@app.exception_handler(APIStatusError)
async def handle_api_status_error(_: Request, exc: APIStatusError):
    return JSONResponse(
        status_code=exc.status_code or 502,
        content={"error_type": "api_status_error", **_extract_error_details(exc)},
    )


# APIError -> 502
@app.exception_handler(APIError)
async def handle_api_error(_: Request, exc: APIError):
    return JSONResponse(status_code=502, content={"error_type": "api_error", **_extract_error_details(exc)})


# OpenAIError -> 502
@app.exception_handler(OpenAIError)
async def handle_openai_error(_: Request, exc: OpenAIError):
    return JSONResponse(status_code=502, content={"error_type": "openai_error", **_extract_error_details(exc)})


# 兜底 Exception -> 500
@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error_type": "internal_server_error", **_extract_error_details(exc)},
    )


app.include_router(router, prefix=settings.app_context_path)
