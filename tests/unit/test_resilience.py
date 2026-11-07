"""HLW 受控回退策略测试。"""

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
import pytest
from langgraph.errors import GraphInterrupt
from openai import APIStatusError

from hlw.models.resilience import ControlledFallbackMiddleware


@dataclass(frozen=True)
class FakeRequest:
    model: str

    def override(self, *, model: str):
        return FakeRequest(model=model)


def make_middleware():
    return ControlledFallbackMiddleware(
        fallback_model="fallback",
        max_retries=1,
        initial_delay=0,
    )


def test_primary_success():
    """主模型正常时，不调用备用模型。"""

    calls = []

    def handler(request):
        calls.append(request.model)
        return "primary-ok"

    result = make_middleware().wrap_model_call(
        FakeRequest("primary"),
        handler,
    )

    assert result == "primary-ok"
    assert calls == ["primary"]


def test_primary_retry_success():
    """首次连接失败，重试成功，不触发回退。"""

    calls = []

    def handler(request):
        calls.append(request.model)

        if len(calls) == 1:
            raise ConnectionError("temporary failure")

        return "primary-recovered"

    result = make_middleware().wrap_model_call(
        FakeRequest("primary"),
        handler,
    )

    assert result == "primary-recovered"
    assert calls == ["primary", "primary"]


def test_fallback_after_retry():
    """主模型两次失败后，由备用模型接管。"""

    calls = []

    def handler(request):
        calls.append(request.model)

        if request.model == "primary":
            raise ConnectionError("unavailable")

        return "fallback-ok"

    result = make_middleware().wrap_model_call(
        FakeRequest("primary"),
        handler,
    )

    assert result == "fallback-ok"

    assert calls == [
        "primary",
        "primary",
        "fallback",
    ]


def test_client_error_no_fallback():
    """HTTP 400 不重试，也不回退。"""

    calls = []

    request = httpx.Request(
        "POST",
        "http://localhost/v1/chat/completions",
    )

    response = httpx.Response(
        400,
        request=request,
    )

    def handler(model_request):
        calls.append(model_request.model)

        raise APIStatusError(
            "Bad Request",
            response=response,
            body=None,
        )

    with pytest.raises(APIStatusError):
        make_middleware().wrap_model_call(
            FakeRequest("primary"),
            handler,
        )

    assert calls == ["primary"]


def test_interrupt_propagates():
    """LangGraph 中断不能被重试或回退吞掉。"""

    calls = []

    def handler(request):
        calls.append(request.model)
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        make_middleware().wrap_model_call(
            FakeRequest("primary"),
            handler,
        )

    assert calls == ["primary"]


def test_async_fallback():
    """验证真正的异步回退路径。"""

    calls = []

    async def handler(request):
        calls.append(request.model)

        if request.model == "primary":
            raise ConnectionError("offline")

        return "async-fallback-ok"

    async def run():
        return await make_middleware().awrap_model_call(
            FakeRequest("primary"),
            handler,
        )

    result = asyncio.run(run())

    assert result == "async-fallback-ok"

    assert calls == [
        "primary",
        "primary",
        "fallback",
    ]


def test_telemetry_primary_success(caplog):
    """主模型成功时自动生成正确的事件。"""

    middleware = make_middleware()

    def handler(request):
        return "ok"

    with caplog.at_level(
        logging.INFO,
        logger="hlw.models.telemetry",
    ):
        middleware.wrap_model_call(
            FakeRequest("primary"),
            handler,
        )

    records = [r for r in caplog.records if r.name == "hlw.models.telemetry"]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("model_call "))

    assert payload["status"] == "success"
    assert payload["model_name"] == "primary"
    assert payload["retry_count"] == 0
    assert payload["fallback_used"] is False
    assert payload["latency_ms"] >= 0


def test_telemetry_fallback_success(caplog):
    """主模型失败并回退后自动生成汇总事件。"""

    middleware = make_middleware()

    def handler(request):
        if request.model == "primary":
            raise ConnectionError("offline")

        return "fallback-ok"

    with caplog.at_level(
        logging.INFO,
        logger="hlw.models.telemetry",
    ):
        result = middleware.wrap_model_call(
            FakeRequest("primary"),
            handler,
        )

    assert result == "fallback-ok"

    records = [r for r in caplog.records if r.name == "hlw.models.telemetry"]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage().removeprefix("model_call "))

    assert payload["status"] == "success"
    assert payload["primary_model"] == "primary"
    assert payload["model_name"] == "fallback"
    assert payload["error_kind"] == "connection"
    assert payload["retry_count"] == 1
    assert payload["fallback_used"] is True
    assert payload["latency_ms"] >= 0
