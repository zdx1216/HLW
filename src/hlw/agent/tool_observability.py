"""HLW 工具调用全链路日志 Middleware。

记录工具输入、输出、执行耗时及异常。
API 和 LangGraph Dev 共用本模块。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp

from hlw.core.event_logger import emit_event
from hlw.core.request_context import get_request_id

logger = logging.getLogger(__name__)


def _mapping(value: Any) -> Mapping:
    """将对象安全地转换为 Mapping。"""
    return value if isinstance(value, Mapping) else {}


def _identifier(*values: Any) -> str | None:
    """提取第一个有效标识符。"""

    for value in values:
        if value is None:
            continue

        text = str(value).strip()

        if text and text != "-":
            return text

    return None


def _context_fields(request: ToolCallRequest) -> dict[str, Any]:
    """从工具运行上下文提取关联 ID。"""

    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)

    if not isinstance(config, Mapping):
        try:
            config = get_config()
        except RuntimeError:
            config = {}

    config = _mapping(config)
    configurable = _mapping(config.get("configurable"))
    metadata = _mapping(config.get("metadata"))
    execution = getattr(runtime, "execution_info", None)

    request_id = _identifier(
        get_request_id(),
        configurable.get("langgraph_request_id"),
        metadata.get("request_id"),
    )

    run_id = _identifier(
        getattr(execution, "run_id", None),
        config.get("run_id"),
        configurable.get("run_id"),
    )

    return {
        "request_id": request_id,
        "trace_id": request_id or run_id,
        "thread_id": _identifier(
            getattr(execution, "thread_id", None),
            configurable.get("thread_id"),
        ),
        "run_id": run_id,
        "user_id": _identifier(
            configurable.get("user_id"),
            metadata.get("user_id"),
        ),
    }


def _tool_details(request: ToolCallRequest) -> dict[str, Any]:
    """提取工具名称、调用 ID 和原始参数。"""

    call = _mapping(getattr(request, "tool_call", None))
    tool = getattr(request, "tool", None)

    return {
        "tool_name": call.get("name") or getattr(tool, "name", None),
        "tool_call_id": call.get("id")
        or getattr(
            getattr(request, "runtime", None),
            "tool_call_id",
            None,
        ),
        "arguments": call.get("args", {}),
    }


def _safe_emit(
    event: str,
    request: ToolCallRequest,
    payload: dict[str, Any],
    *,
    span_id: str,
    status: str,
    duration_ms: float | None = None,
) -> None:
    """安全记录事件，日志失败不影响工具调用。"""

    try:
        fields = _context_fields(request)

        emit_event(
            event,
            payload=payload,
            request_id=fields["request_id"],
            trace_id=fields["trace_id"],
            span_id=span_id,
            thread_id=fields["thread_id"],
            run_id=fields["run_id"],
            user_id=fields["user_id"],
            duration_ms=duration_ms,
            status=status,
        )

    except Exception as exc:  # noqa: BLE001
        # 日志属于旁路系统，不能改变工具的执行结果。
        # 不打印异常原文，避免泄露敏感内容。
        logger.warning(
            "Tool audit logging failed: event=%s error_type=%s",
            event,
            type(exc).__name__,
        )


class ToolCallAuditMiddleware(AgentMiddleware):
    """统一记录同步和异步工具执行。"""

    @staticmethod
    def _started(request: ToolCallRequest) -> tuple[str, float]:
        """记录工具执行开始。"""

        execution_id = uuid4().hex
        started = perf_counter()

        details = _tool_details(request)

        _safe_emit(
            "tool.call.started",
            request,
            {
                **details,
                "tool_execution_id": execution_id,
            },
            span_id=execution_id,
            status="started",
        )

        return execution_id, started

    @staticmethod
    def _finished(
        request: ToolCallRequest,
        execution_id: str,
        started: float,
        *,
        status: str,
        output: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """记录工具完成、异常或取消。"""

        payload = {
            **_tool_details(request),
            "tool_execution_id": execution_id,
        }

        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)

        else:
            # ToolMessage 和 Command 等对象交给统一
            # event_logger 负责 JSON 序列化与脱敏。
            payload["output"] = output
            payload["output_type"] = type(output).__name__

        event_names = {
            "success": "tool.call.completed",
            "failed": "tool.call.failed",
            "cancelled": "tool.call.cancelled",
            "control_flow": "tool.call.control_flow",
        }

        _safe_emit(
            event_names[status],
            request,
            payload,
            span_id=execution_id,
            status=status,
            duration_ms=(perf_counter() - started) * 1000,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """同步工具执行审计。"""

        execution_id, started = self._started(request)

        try:
            result = handler(request)

        except GraphBubbleUp:
            self._finished(
                request,
                execution_id,
                started,
                status="control_flow",
            )
            raise

        except asyncio.CancelledError:
            self._finished(
                request,
                execution_id,
                started,
                status="cancelled",
            )
            raise

        except Exception as exc:
            self._finished(
                request,
                execution_id,
                started,
                status="failed",
                error=exc,
            )
            raise

        else:
            # 有些工具不抛异常，而是返回 status=error
            # 的 ToolMessage，同样应记录失败状态。
            is_error = getattr(result, "status", None) == "error"

            self._finished(
                request,
                execution_id,
                started,
                status="failed" if is_error else "success",
                output=result,
            )

            return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[Any],
        ],
    ) -> Any:
        """异步工具执行审计。"""

        execution_id, started = self._started(request)

        try:
            result = await handler(request)

        except GraphBubbleUp:
            self._finished(
                request,
                execution_id,
                started,
                status="control_flow",
            )
            raise

        except asyncio.CancelledError:
            self._finished(
                request,
                execution_id,
                started,
                status="cancelled",
            )
            raise

        except Exception as exc:
            self._finished(
                request,
                execution_id,
                started,
                status="failed",
                error=exc,
            )
            raise

        else:
            is_error = getattr(result, "status", None) == "error"

            self._finished(
                request,
                execution_id,
                started,
                status="failed" if is_error else "success",
                output=result,
            )

            return result
