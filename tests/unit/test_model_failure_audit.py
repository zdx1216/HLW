"""模型失败、重试和回退的 JSONL 日志集成测试。"""

import json
import logging
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from hlw.agent.model_observability import (
    ModelAttemptLoggingMiddleware,
    ModelCallAuditMiddleware,
)
from hlw.core.logging_config import setup_logging
from hlw.core.request_context import request_context
from hlw.models.resilience import ControlledFallbackMiddleware


@dataclass(frozen=True)
class FakeModelRequest:
    model: str
    messages: tuple
    runtime: Any
    system_message: Any = None
    tools: tuple = ()
    model_settings: Any = None

    def override(self, *, model):
        return replace(self, model=model)


def make_request():
    return FakeModelRequest(
        model="primary",
        messages=(HumanMessage(content="测试重试日志"),),
        runtime=SimpleNamespace(
            execution_info=SimpleNamespace(
                run_id="test-run-001",
                thread_id="test-thread-001",
            ),
        ),
    )


def make_response():
    return SimpleNamespace(
        result=[AIMessage(content="模型成功回答")],
        structured_response=None,
    )


@pytest.fixture
def read_events(tmp_path):
    """将真实 JSONL 事件写到临时目录。"""

    directory = tmp_path / "api"

    setup_logging(runtime="api", log_dir=directory)

    def read():
        for handler in logging.getLogger("hlw.events").handlers:
            handler.flush()

        path = directory / "events.jsonl"

        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    yield read

    # 清理本测试创建的日志 Handler。
    for name in ("hlw.events", "hlw"):
        logger = logging.getLogger(name)

        for handler in list(logger.handlers):
            if getattr(handler, "_hlw_managed", False):
                logger.removeHandler(handler)
                handler.close()


async def invoke_with_audit(handler):
    """模拟实际模型 Middleware 的嵌套执行顺序。"""

    audit = ModelCallAuditMiddleware()

    fallback = ControlledFallbackMiddleware(
        fallback_model="fallback",
        max_retries=1,
        initial_delay=0,
    )

    attempts = ModelAttemptLoggingMiddleware()

    async def execute(request):
        return await fallback.awrap_model_call(
            request,
            lambda current: attempts.awrap_model_call(
                current,
                handler,
            ),
        )

    with request_context("test-request-001"):
        return await audit.awrap_model_call(
            make_request(),
            execute,
        )


@pytest.mark.anyio
async def test_retry_success_jsonl(read_events):
    """首次失败，重试成功。"""

    calls = []

    async def handler(request):
        calls.append(request.model)

        if len(calls) == 1:
            raise ConnectionError("simulated failure")

        return make_response()

    response = await invoke_with_audit(handler)

    assert response.result[0].content == "模型成功回答"
    assert calls == ["primary", "primary"]

    events = read_events()
    names = [item["event"] for item in events]

    assert names.count("model.attempt.started") == 2
    assert names.count("model.attempt.failed") == 1
    assert names.count("model.attempt.completed") == 1
    assert names.count("model.response") == 1

    summary = next(item for item in events if item["event"] == "model.call.summary")

    assert summary["payload"]["retry_count"] == 1
    assert summary["payload"]["fallback_used"] is False


@pytest.mark.anyio
async def test_fallback_success_jsonl(read_events):
    """主模型重试失败，备用模型成功。"""

    calls = []

    async def handler(request):
        calls.append(request.model)

        if request.model == "primary":
            raise ConnectionError("primary unavailable")

        return make_response()

    response = await invoke_with_audit(handler)

    assert response.result[0].content == "模型成功回答"

    assert calls == [
        "primary",
        "primary",
        "fallback",
    ]

    events = read_events()
    names = [item["event"] for item in events]

    assert names.count("model.attempt.started") == 3
    assert names.count("model.attempt.failed") == 2
    assert names.count("model.attempt.completed") == 1

    summary = next(item for item in events if item["event"] == "model.call.summary")

    assert summary["status"] == "success"
    assert summary["payload"]["fallback_used"] is True
    assert summary["payload"]["model_name"] == "fallback"
    assert summary["payload"]["retry_count"] == 1


@pytest.mark.anyio
async def test_all_models_fail_jsonl(read_events):
    """主模型和备用模型均失败。"""

    calls = []

    async def handler(request):
        calls.append(request.model)
        raise ConnectionError("simulated unavailable")

    with pytest.raises(ConnectionError):
        await invoke_with_audit(handler)

    assert calls == [
        "primary",
        "primary",
        "fallback",
    ]

    events = read_events()
    names = [item["event"] for item in events]

    assert names.count("model.attempt.started") == 3
    assert names.count("model.attempt.failed") == 3
    assert names.count("model.call.failed") == 1

    summary = next(item for item in events if item["event"] == "model.call.summary")

    assert summary["status"] == "failed"
    assert summary["payload"]["fallback_used"] is True

    # 所有事件必须能够追溯到同一次请求。
    assert all(item["request_id"] == "test-request-001" for item in events)

    assert all(item["run_id"] == "test-run-001" for item in events)
