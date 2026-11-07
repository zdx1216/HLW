"""HLW 模型调用可观测性测试。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import hlw.agent.model_observability as observability


def make_request():
    """构造模拟模型请求，不调用真实模型。"""

    return SimpleNamespace(
        model=SimpleNamespace(model_name="qwen3-test"),
        messages=[HumanMessage(content="你好")],
        system_message=None,
        tools=[],
        model_settings={"temperature": 0.2},
        runtime=SimpleNamespace(execution_info=None),
    )


def make_response():
    """构造模拟模型响应。"""

    return SimpleNamespace(
        result=[AIMessage(content="你好，我是 HLW。")],
        structured_response=None,
    )


@pytest.fixture
def captured_events(monkeypatch):
    """拦截事件写入，检查 Middleware 是否正确产生事件。"""

    events = []

    def fake_emit(event, request, payload, **kwargs):
        events.append(
            {
                "event": event,
                "payload": payload,
                **kwargs,
            }
        )

    monkeypatch.setattr(
        observability,
        "_emit",
        fake_emit,
    )

    return events


def test_model_call_success(captured_events):
    """同步模型调用应记录输入和输出。"""

    middleware = observability.ModelCallAuditMiddleware()

    response = middleware.wrap_model_call(
        make_request(),
        lambda request: make_response(),
    )

    assert response.result[0].content == "你好，我是 HLW。"

    assert [item["event"] for item in captured_events] == [
        "model.request",
        "model.response",
    ]

    request_event = captured_events[0]
    response_event = captured_events[1]

    assert request_event["payload"]["request"]["model"]["model_name"] == ("qwen3-test")

    assert response_event["status"] == "success"
    assert response_event["duration_ms"] >= 0

    assert (
        request_event["payload"]["model_call_id"]
        == response_event["payload"]["model_call_id"]
    )


@pytest.mark.anyio
async def test_async_model_call_success(captured_events):
    """异步模型调用应记录完整生命周期。"""

    middleware = observability.ModelCallAuditMiddleware()

    async def handler(request):
        return make_response()

    response = await middleware.awrap_model_call(
        make_request(),
        handler,
    )

    assert response.result[0].content == "你好，我是 HLW。"

    assert [item["event"] for item in captured_events] == [
        "model.request",
        "model.response",
    ]


def test_model_call_failure(captured_events):
    """模型失败时必须记录异常并继续抛出。"""

    middleware = observability.ModelCallAuditMiddleware()

    def failing_handler(request):
        raise RuntimeError("simulated model error")

    with pytest.raises(RuntimeError, match="simulated model error"):
        middleware.wrap_model_call(
            make_request(),
            failing_handler,
        )

    assert [item["event"] for item in captured_events] == [
        "model.request",
        "model.call.failed",
    ]

    failure = captured_events[-1]

    assert failure["status"] == "failed"
    assert failure["payload"]["error_type"] == "RuntimeError"
    assert failure["duration_ms"] >= 0


def test_model_attempt_logging(captured_events):
    """每次实际模型尝试应有独立开始和结束事件。"""

    middleware = observability.ModelAttemptLoggingMiddleware()

    result = middleware.wrap_model_call(
        make_request(),
        lambda request: make_response(),
    )

    assert result.result[0].content == "你好，我是 HLW。"

    assert [item["event"] for item in captured_events] == [
        "model.attempt.started",
        "model.attempt.completed",
    ]

    started, completed = captured_events

    assert started["payload"]["attempt_number"] == 1
    assert started["payload"]["attempt_id"] == completed["payload"]["attempt_id"]
    assert completed["duration_ms"] >= 0


def test_model_call_survives_logging_failure(monkeypatch):
    """日志系统失败，模型仍然返回真实结果。"""

    def failing_writer(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        observability,
        "safe_emit_event",
        failing_writer,
    )

    middleware = observability.ModelCallAuditMiddleware()

    response = middleware.wrap_model_call(
        make_request(),
        lambda request: make_response(),
    )

    assert response.result[0].content == "你好，我是 HLW。"


def test_model_call_survives_snapshot_failure(monkeypatch):
    """消息序列化异常不能破坏模型返回。"""

    def failing_snapshot(*args, **kwargs):
        raise ValueError("serialization failure")

    monkeypatch.setattr(
        observability,
        "_request_data",
        failing_snapshot,
    )

    monkeypatch.setattr(
        observability,
        "_response_data",
        failing_snapshot,
    )

    middleware = observability.ModelCallAuditMiddleware()

    response = middleware.wrap_model_call(
        make_request(),
        lambda request: make_response(),
    )

    assert response.result[0].content == "你好，我是 HLW。"


def test_model_summary_inherits_run_id(monkeypatch):
    """模型汇总事件应继承当前模型调用的 Run ID。"""

    from hlw.core.request_context import get_run_id
    from hlw.models import telemetry
    from hlw.models.telemetry import ModelCallEvent, emit_model_call

    captured = []

    def fake_emit(event, **kwargs):
        captured.append(
            {
                "event": event,
                **kwargs,
            }
        )
        return True

    monkeypatch.setattr(
        telemetry,
        "safe_emit_event",
        fake_emit,
    )

    request = make_request()
    request.runtime.execution_info = SimpleNamespace(
        run_id="run-summary-test-001",
    )

    summary = ModelCallEvent(
        primary_model="qwen3-test",
        model_name="qwen3-test",
        status="success",
        error_kind=None,
        retry_count=0,
        fallback_used=False,
        latency_ms=10.0,
    )

    def handler(_):
        emit_model_call(summary)
        return make_response()

    middleware = observability.ModelCallAuditMiddleware()

    response = middleware.wrap_model_call(
        request,
        handler,
    )

    assert response.result[0].content == "你好，我是 HLW。"

    assert len(captured) == 1
    assert captured[0]["event"] == "model.call.summary"
    assert captured[0]["run_id"] == "run-summary-test-001"

    # Middleware 退出后必须恢复上下文。
    assert get_run_id() is None


@pytest.mark.anyio
async def test_async_model_summary_inherits_run_id(monkeypatch):
    """异步模型汇总事件应继承当前 Run ID。"""

    from hlw.core.request_context import get_run_id
    from hlw.models import telemetry
    from hlw.models.telemetry import ModelCallEvent, emit_model_call

    captured = []

    def fake_emit(event, **kwargs):
        captured.append(
            {
                "event": event,
                **kwargs,
            }
        )
        return True

    monkeypatch.setattr(
        telemetry,
        "safe_emit_event",
        fake_emit,
    )

    request = make_request()
    request.runtime.execution_info = SimpleNamespace(
        run_id="run-async-summary-001",
    )

    summary = ModelCallEvent(
        primary_model="qwen3-test",
        model_name="qwen3-test",
        status="success",
        error_kind=None,
        retry_count=0,
        fallback_used=False,
        latency_ms=10.0,
    )

    async def handler(_):
        emit_model_call(summary)
        return make_response()

    middleware = observability.ModelCallAuditMiddleware()

    response = await middleware.awrap_model_call(
        request,
        handler,
    )

    assert response.result[0].content == "你好，我是 HLW。"
    assert len(captured) == 1
    assert captured[0]["event"] == "model.call.summary"
    assert captured[0]["run_id"] == "run-async-summary-001"

    # 异步执行结束后，必须恢复原来的上下文。
    assert get_run_id() is None
