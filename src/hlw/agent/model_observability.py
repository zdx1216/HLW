
"""HLW 模型调用全链路可观测性 Middleware。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp

from hlw.core.event_logger import emit_event
from hlw.core.request_context import get_request_id


_active_call_id: ContextVar[str | None] = ContextVar(
    "hlw_model_call_id",
    default=None,
)

_attempt_state: ContextVar[dict[str, int] | None] = ContextVar(
    "hlw_model_attempt_state",
    default=None,
)


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _context_fields(request: ModelRequest[Any]) -> dict:
    """提取请求、用户、会话和运行 ID。"""

    try:
        config = get_config()
    except RuntimeError:
        config = {}

    configurable = _mapping(config.get("configurable"))
    metadata = _mapping(config.get("metadata"))

    runtime = getattr(request, "runtime", None)
    execution = getattr(runtime, "execution_info", None)

    request_id = get_request_id()

    if request_id == "-":
        request_id = (
            configurable.get("langgraph_request_id")
            or metadata.get("request_id")
            or None
        )

    thread_id = (
        getattr(execution, "thread_id", None)
        or configurable.get("thread_id")
        or metadata.get("thread_id")
    )

    run_id = (
        getattr(execution, "run_id", None)
        or config.get("run_id")
        or configurable.get("run_id")
    )

    user_id = (
        configurable.get("user_id")
        or metadata.get("user_id")
    )

    return {
        "request_id": str(request_id) if request_id else None,
        "trace_id": str(request_id or run_id) if request_id or run_id else None,
        "thread_id": str(thread_id) if thread_id else None,
        "run_id": str(run_id) if run_id else None,
        "user_id": str(user_id) if user_id else None,
    }


def _message_data(message: Any) -> Any:
    """序列化 LangChain 消息。"""

    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")

    return {
        "type": getattr(message, "type", type(message).__name__),
        "content": getattr(message, "content", str(message)),
    }


def _model_data(model: Any) -> dict:
    """提取模型基本信息及常用参数。"""

    result = {
        "class": type(model).__name__,
    }

    for name in (
        "model_name",
        "model",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "timeout",
        "max_retries",
        "streaming",
    ):
        value = getattr(model, name, None)

        if isinstance(value, (str, int, float, bool)):
            result[name] = value

    return result


def _tool_data(tool: Any) -> dict:
    """提取工具名称、说明和输入 Schema。"""

    if isinstance(tool, Mapping):
        return dict(tool)

    result = {
        "name": getattr(tool, "name", type(tool).__name__),
        "description": getattr(tool, "description", None),
    }

    schema = getattr(tool, "args_schema", None)

    if hasattr(schema, "model_json_schema"):
        result["input_schema"] = schema.model_json_schema()

    return result


def _request_data(request: ModelRequest[Any]) -> dict:
    """完整提取模型请求内容。"""

    return {
        "model": _model_data(request.model),
        "messages": [
            _message_data(message)
            for message in request.messages
        ],
        "system_message": (
            _message_data(request.system_message)
            if request.system_message is not None
            else None
        ),
        "system_prompt": getattr(request, "system_prompt", None),
        "tool_choice": getattr(request, "tool_choice", None),
        "tools": [
            _tool_data(tool)
            for tool in (request.tools or [])
        ],
        "response_format": getattr(request, "response_format", None),
        "model_settings": getattr(request, "model_settings", None),
    }


def _response_data(response: Any) -> dict:
    """提取模型返回消息、工具调用和 Token 信息。"""

    messages = getattr(response, "result", None)

    if messages is None:
        messages = (
            [response]
            if getattr(response, "type", None) == "ai"
            else []
        )

    serialized = [
        _message_data(message)
        for message in messages
    ]

    return {
        "messages": serialized,
        "structured_response": getattr(
            response,
            "structured_response",
            None,
        ),
        "usage": [
            getattr(message, "usage_metadata", None)
            for message in messages
            if getattr(message, "usage_metadata", None)
        ],
        "response_metadata": [
            getattr(message, "response_metadata", {})
            for message in messages
        ],
    }


def _emit(
    event: str,
    request: ModelRequest[Any],
    payload: dict,
    *,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    duration_ms: float | None = None,
    status: str | None = None,
) -> None:
    """统一写入结构化事件日志。"""

    fields = _context_fields(request)

    emit_event(
        event,
        payload=payload,
        request_id=fields["request_id"],
        trace_id=fields["trace_id"],
        span_id=span_id,
        parent_span_id=parent_span_id,
        user_id=fields["user_id"],
        thread_id=fields["thread_id"],
        run_id=fields["run_id"],
        duration_ms=duration_ms,
        status=status,
    )


class ModelCallAuditMiddleware(AgentMiddleware):
    """记录一次完整模型调用的请求和最终响应。"""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]],
            ModelResponse[Any],
        ],
    ) -> ModelResponse[Any]:

        call_id = uuid4().hex
        started = perf_counter()

        call_token = _active_call_id.set(call_id)
        attempt_token = _attempt_state.set({"count": 0})

        _emit(
            "model.request",
            request,
            {
                "model_call_id": call_id,
                "request": _request_data(request),
            },
            span_id=call_id,
            status="started",
        )

        try:
            response = handler(request)

        except GraphBubbleUp:
            _emit(
                "model.call.control_flow",
                request,
                {"model_call_id": call_id},
                span_id=call_id,
                status="control_flow",
            )
            raise

        except asyncio.CancelledError:
            _emit(
                "model.call.cancelled",
                request,
                {"model_call_id": call_id},
                span_id=call_id,
                duration_ms=(perf_counter() - started) * 1000,
                status="cancelled",
            )
            raise

        except Exception as exc:
            _emit(
                "model.call.failed",
                request,
                {
                    "model_call_id": call_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                span_id=call_id,
                duration_ms=(perf_counter() - started) * 1000,
                status="failed",
            )
            raise

        else:
            _emit(
                "model.response",
                request,
                {
                    "model_call_id": call_id,
                    "response": _response_data(response),
                },
                span_id=call_id,
                duration_ms=(perf_counter() - started) * 1000,
                status="success",
            )
            return response

        finally:
            _attempt_state.reset(attempt_token)
            _active_call_id.reset(call_token)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelResponse[Any]:

        call_id = uuid4().hex
        started = perf_counter()

        call_token = _active_call_id.set(call_id)
        attempt_token = _attempt_state.set({"count": 0})

        _emit(
            "model.request",
            request,
            {
                "model_call_id": call_id,
                "request": _request_data(request),
            },
            span_id=call_id,
            status="started",
        )

        try:
            response = await handler(request)

        except GraphBubbleUp:
            _emit(
                "model.call.control_flow",
                request,
                {"model_call_id": call_id},
                span_id=call_id,
                status="control_flow",
            )
            raise

        except asyncio.CancelledError:
            _emit(
                "model.call.cancelled",
                request,
                {"model_call_id": call_id},
                span_id=call_id,
                duration_ms=(perf_counter() - started) * 1000,
                status="cancelled",
            )
            raise

        except Exception as exc:
            _emit(
                "model.call.failed",
                request,
                {
                    "model_call_id": call_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                span_id=call_id,
                duration_ms=(perf_counter() - started) * 1000,
                status="failed",
            )
            raise

        else:
            _emit(
                "model.response",
                request,
                {
                    "model_call_id": call_id,
                    "response": _response_data(response),
                },
                span_id=call_id,
                duration_ms=(perf_counter() - started) * 1000,
                status="success",
            )
            return response

        finally:
            _attempt_state.reset(attempt_token)
            _active_call_id.reset(call_token)


class ModelAttemptLoggingMiddleware(AgentMiddleware):
    """记录每一次实际模型调用，包括重试和备用模型。"""

    def _start(self, request):
        state = _attempt_state.get()

        if state is None:
            attempt_number = 1
        else:
            state["count"] += 1
            attempt_number = state["count"]

        call_id = _active_call_id.get()
        attempt_id = uuid4().hex
        started = perf_counter()

        _emit(
            "model.attempt.started",
            request,
            {
                "model_call_id": call_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "model": _model_data(request.model),
            },
            span_id=attempt_id,
            parent_span_id=call_id,
            status="started",
        )

        return attempt_id, call_id, attempt_number, started

    def _finish(
        self,
        request,
        attempt,
        call_id,
        attempt_number,
        started,
        *,
        status,
        error=None,
    ):
        elapsed = (perf_counter() - started) * 1000

        payload = {
            "model_call_id": call_id,
            "attempt_id": attempt,
            "attempt_number": attempt_number,
            "model": _model_data(request.model),
        }

        if error is not None:
            payload.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
            })

        _emit(
            f"model.attempt.{status}",
            request,
            payload,
            span_id=attempt,
            parent_span_id=call_id,
            duration_ms=elapsed,
            status=status,
        )

    def wrap_model_call(self, request, handler):
        attempt, call_id, number, started = self._start(request)

        try:
            response = handler(request)

        except GraphBubbleUp:
            self._finish(
                request, attempt, call_id, number, started,
                status="control_flow",
            )
            raise

        except asyncio.CancelledError:
            self._finish(
                request, attempt, call_id, number, started,
                status="cancelled",
            )
            raise

        except Exception as exc:
            self._finish(
                request, attempt, call_id, number, started,
                status="failed",
                error=exc,
            )
            raise

        else:
            self._finish(
                request, attempt, call_id, number, started,
                status="completed",
            )
            return response

    async def awrap_model_call(self, request, handler):
        attempt, call_id, number, started = self._start(request)

        try:
            response = await handler(request)

        except GraphBubbleUp:
            self._finish(
                request, attempt, call_id, number, started,
                status="control_flow",
            )
            raise

        except asyncio.CancelledError:
            self._finish(
                request, attempt, call_id, number, started,
                status="cancelled",
            )
            raise

        except Exception as exc:
            self._finish(
                request, attempt, call_id, number, started,
                status="failed",
                error=exc,
            )
            raise

        else:
            self._finish(
                request, attempt, call_id, number, started,
                status="completed",
            )
            return response
