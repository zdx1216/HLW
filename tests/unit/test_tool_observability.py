"""HLW 工具调用日志测试。"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

import hlw.agent.tool_observability as observability
from hlw.agent.request_context_middleware import (
    RequestContextMiddleware,
)
from hlw.core.request_context import get_request_id


def make_request():
    """构造模拟工具请求。"""

    return SimpleNamespace(
        tool_call={
            "name": "get_project_status",
            "args": {"detail": True},
            "id": "call-001",
        },
        tool=SimpleNamespace(name="get_project_status"),
        state={},
        runtime=SimpleNamespace(
            config={
                "configurable": {
                    "langgraph_request_id": "request-001",
                    "thread_id": "thread-001",
                    "user_id": "zdx",
                }
            },
            execution_info=SimpleNamespace(
                run_id="run-001",
                thread_id="thread-001",
            ),
            tool_call_id="call-001",
        ),
    )


def make_result(status="success"):
    """构造 LangChain ToolMessage。"""

    return ToolMessage(
        content='{"phase": "P0"}',
        tool_call_id="call-001",
        status=status,
    )


@pytest.fixture
def events(monkeypatch):
    """捕获结构化事件，不写真实文件。"""

    records = []

    def fake_emit(event, **kwargs):
        records.append(
            {
                "event": event,
                **kwargs,
            }
        )

    monkeypatch.setattr(
        observability,
        "emit_event",
        fake_emit,
    )

    return records


def test_sync_tool_success(events):
    """同步工具成功时记录输入、输出和耗时。"""

    middleware = observability.ToolCallAuditMiddleware()
    result = make_result()

    actual = middleware.wrap_tool_call(
        make_request(),
        lambda _: result,
    )

    assert actual is result

    started, completed = events

    assert started["event"] == "tool.call.started"
    assert completed["event"] == "tool.call.completed"

    assert started["payload"]["tool_name"] == ("get_project_status")
    assert started["payload"]["arguments"] == {"detail": True}

    assert completed["payload"]["output"] is result
    assert completed["duration_ms"] >= 0

    assert started["span_id"] == completed["span_id"]
    assert started["request_id"] == "request-001"
    assert completed["thread_id"] == "thread-001"


@pytest.mark.anyio
async def test_async_tool_success(events):
    """异步工具成功时记录执行过程。"""

    middleware = observability.ToolCallAuditMiddleware()

    async def handler(_):
        return make_result()

    actual = await middleware.awrap_tool_call(
        make_request(),
        handler,
    )

    assert actual.content == '{"phase": "P0"}'
    assert events[0]["event"] == "tool.call.started"
    assert events[1]["event"] == "tool.call.completed"
    assert events[1]["status"] == "success"


def test_tool_exception(events):
    """工具抛异常后必须记录并原样传播。"""

    middleware = observability.ToolCallAuditMiddleware()

    def failing_handler(_):
        raise ValueError("simulated tool failure")

    with pytest.raises(ValueError, match="simulated tool failure"):
        middleware.wrap_tool_call(
            make_request(),
            failing_handler,
        )

    assert events[-1]["event"] == "tool.call.failed"
    assert events[-1]["status"] == "failed"
    assert events[-1]["payload"]["error_type"] == "ValueError"


def test_tool_error_message(events):
    """工具返回错误 ToolMessage 时也要标记失败。"""

    middleware = observability.ToolCallAuditMiddleware()
    result = make_result(status="error")

    actual = middleware.wrap_tool_call(
        make_request(),
        lambda _: result,
    )

    assert actual is result
    assert events[-1]["event"] == "tool.call.failed"
    assert events[-1]["status"] == "failed"
    assert events[-1]["payload"]["output"] is result


@pytest.mark.anyio
async def test_tool_cancellation(events):
    """异步工具取消时记录取消事件。"""

    middleware = observability.ToolCallAuditMiddleware()

    async def cancelled_handler(_):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await middleware.awrap_tool_call(
            make_request(),
            cancelled_handler,
        )

    assert events[-1]["event"] == "tool.call.cancelled"
    assert events[-1]["status"] == "cancelled"


def test_logging_failure_does_not_break_tool(
    monkeypatch,
):
    """日志写入失败不能影响工具返回。"""

    def failing_emit(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(
        observability,
        "emit_event",
        failing_emit,
    )

    middleware = observability.ToolCallAuditMiddleware()
    result = make_result()

    actual = middleware.wrap_tool_call(
        make_request(),
        lambda _: result,
    )

    assert actual is result


def test_request_context_propagates_to_tool(events):
    """请求上下文必须传递到工具日志。"""

    request = make_request()
    context_middleware = RequestContextMiddleware()
    audit_middleware = observability.ToolCallAuditMiddleware()

    result = context_middleware.wrap_tool_call(
        request,
        lambda req: audit_middleware.wrap_tool_call(
            req,
            lambda _: make_result(),
        ),
    )

    assert result.status == "success"

    assert events[0]["request_id"] == "request-001"
    assert events[1]["request_id"] == "request-001"

    # 请求执行完成后，上下文必须恢复。
    assert get_request_id() == "-"
