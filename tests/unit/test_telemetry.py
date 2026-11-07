"""模型调用日志事件测试。"""

import json
import logging

from hlw.models.telemetry import (
    ModelCallEvent,
    emit_model_call,
)


def test_primary_success_event(caplog):
    event = ModelCallEvent(
        primary_model="qwen3:8b",
        model_name="qwen3:8b",
        status="success",
        error_kind=None,
        retry_count=0,
        fallback_used=False,
        latency_ms=125.5,
    )

    with caplog.at_level(
        logging.INFO,
        logger="hlw.models.telemetry",
    ):
        emit_model_call(event)

    record = caplog.records[-1]

    assert record.levelno == logging.INFO

    payload = json.loads(record.getMessage().removeprefix("model_call "))

    assert payload["event"] == "model_call"
    assert payload["model_name"] == "qwen3:8b"
    assert payload["retry_count"] == 0
    assert payload["fallback_used"] is False


def test_fallback_success_event(caplog):
    event = ModelCallEvent(
        primary_model="qwen3:8b",
        model_name="qwen3:14b",
        status="success",
        error_kind="timeout",
        retry_count=1,
        fallback_used=True,
        latency_ms=3200.0,
    )

    with caplog.at_level(
        logging.INFO,
        logger="hlw.models.telemetry",
    ):
        emit_model_call(event)

    payload = json.loads(caplog.records[-1].getMessage().removeprefix("model_call "))

    assert payload["model_name"] == "qwen3:14b"
    assert payload["error_kind"] == "timeout"
    assert payload["retry_count"] == 1
    assert payload["fallback_used"] is True
