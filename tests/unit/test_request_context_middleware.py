from types import SimpleNamespace

import pytest

import hlw.agent.request_context_middleware as middleware_module
from hlw.agent.request_context_middleware import RequestContextMiddleware
from hlw.core.request_context import get_request_id, request_context


def make_request(run_id=None):
    return SimpleNamespace(
        runtime=SimpleNamespace(execution_info=SimpleNamespace(run_id=run_id))
    )


def test_uses_langgraph_request_id(monkeypatch):
    """LangGraph Dev 优先使用 Server 请求 ID。"""

    monkeypatch.setattr(
        middleware_module,
        "get_config",
        lambda: {"configurable": {"langgraph_request_id": "lg-request-001"}},
    )

    middleware = RequestContextMiddleware()
    observed = []

    result = middleware.wrap_model_call(
        make_request(),
        lambda _: observed.append(get_request_id()) or "success",
    )

    assert result == "success"
    assert observed == ["lg-request-001"]
    assert get_request_id() == "-"


def test_api_request_id_has_priority(monkeypatch):
    """API 已有请求上下文时，不被覆盖。"""

    monkeypatch.setattr(
        middleware_module,
        "get_config",
        lambda: {"configurable": {"langgraph_request_id": "lg-request-001"}},
    )

    middleware = RequestContextMiddleware()
    observed = []

    with request_context("api-request-001"):
        middleware.wrap_model_call(
            make_request(),
            lambda _: observed.append(get_request_id()),
        )

        assert get_request_id() == "api-request-001"

    assert observed == ["api-request-001"]
    assert get_request_id() == "-"


def test_falls_back_to_graph_run_id(monkeypatch):
    """缺少 Server 请求 ID 时使用 Graph Run ID。"""

    def missing_config():
        raise RuntimeError("No runnable config")

    monkeypatch.setattr(
        middleware_module,
        "get_config",
        missing_config,
    )

    middleware = RequestContextMiddleware()
    observed = []

    middleware.wrap_model_call(
        make_request("graph-run-001"),
        lambda _: observed.append(get_request_id()),
    )

    assert observed == ["graph-run-001"]
    assert get_request_id() == "-"


@pytest.mark.anyio
async def test_async_model_call_context(monkeypatch):
    """异步模型调用能够正确传递请求 ID。"""

    monkeypatch.setattr(
        middleware_module,
        "get_config",
        lambda: {"configurable": {"langgraph_request_id": "lg-async-001"}},
    )

    middleware = RequestContextMiddleware()
    observed = []

    async def handler(_):
        observed.append(get_request_id())
        return "success"

    result = await middleware.awrap_model_call(
        make_request(),
        handler,
    )

    assert result == "success"
    assert observed == ["lg-async-001"]
    assert get_request_id() == "-"

def make_tool_request(config):
    """构造工具调用测试请求。"""

    return SimpleNamespace(
        tool_call={
            "name": "test_tool",
            "args": {},
            "id": "tool-001",
        },
        tool=None,
        state={},
        runtime=SimpleNamespace(config=config),
    )


def test_sync_tool_call_context():
    """同步工具调用能够继承 request_id。"""

    middleware = RequestContextMiddleware()

    request = make_tool_request({
        "configurable": {
            "langgraph_request_id": "tool-request-001",
        }
    })

    observed = []

    result = middleware.wrap_tool_call(
        request,
        lambda _: observed.append(get_request_id()) or "success",
    )

    assert result == "success"
    assert observed == ["tool-request-001"]
    assert get_request_id() == "-"


@pytest.mark.anyio
async def test_async_tool_call_context():
    """异步工具调用能够继承 request_id。"""

    middleware = RequestContextMiddleware()

    request = make_tool_request({
        "configurable": {
            "langgraph_request_id": "tool-async-001",
        }
    })

    observed = []

    async def handler(_):
        observed.append(get_request_id())
        return "success"

    result = await middleware.awrap_tool_call(
        request,
        handler,
    )

    assert result == "success"
    assert observed == ["tool-async-001"]
    assert get_request_id() == "-"


def test_tool_call_uses_metadata_request_id():
    """工具调用可以从 metadata 获取 request_id。"""

    middleware = RequestContextMiddleware()

    request = make_tool_request({
        "metadata": {
            "request_id": "metadata-request-001",
        }
    })

    observed = []

    middleware.wrap_tool_call(
        request,
        lambda _: observed.append(get_request_id()),
    )

    assert observed == ["metadata-request-001"]


def test_tool_call_preserves_existing_context():
    """工具调用不覆盖已有 API 请求上下文。"""

    middleware = RequestContextMiddleware()

    request = make_tool_request({
        "configurable": {
            "langgraph_request_id": "langgraph-request-001",
        }
    })

    observed = []

    with request_context("api-request-001"):
        middleware.wrap_tool_call(
            request,
            lambda _: observed.append(get_request_id()),
        )

    assert observed == ["api-request-001"]
    assert get_request_id() == "-"


def test_tool_call_restores_context_after_exception():
    """工具调用异常后正确恢复请求上下文。"""

    middleware = RequestContextMiddleware()

    request = make_tool_request({
        "configurable": {
            "langgraph_request_id": "error-request-001",
        }
    })

    def failing_handler(_):
        assert get_request_id() == "error-request-001"
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        middleware.wrap_tool_call(
            request,
            failing_handler,
        )

    assert get_request_id() == "-"
