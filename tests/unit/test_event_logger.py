
import json
import logging

import pytest

from hlw.core.event_logger import emit_event
from hlw.core.logging_config import setup_logging
from hlw.core.request_context import request_context


@pytest.fixture(autouse=True)
def cleanup_event_logger():
    yield

    logger = logging.getLogger("hlw.events")

    for handler in list(logger.handlers):
        if getattr(handler, "_hlw_managed", False):
            logger.removeHandler(handler)
            handler.close()


def test_event_written_as_jsonl(tmp_path):
    log_file = setup_logging(
        runtime="api",
        log_dir=tmp_path / "api",
    )

    with request_context("request-001"):
        emit_event(
            "request.started",
            payload={
                "message": "测试请求",
            },
        )

    event_logger = logging.getLogger("hlw.events")

    for handler in event_logger.handlers:
        handler.flush()

    events_file = log_file.parent / "events.jsonl"

    records = [
        json.loads(line)
        for line in events_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 1

    record = records[0]

    assert record["event"] == "request.started"
    assert record["request_id"] == "request-001"
    assert record["runtime"] == "api"
    assert record["payload"]["message"] == "测试请求"


def test_sensitive_fields_are_redacted(tmp_path):
    setup_logging(
        runtime="api",
        log_dir=tmp_path / "api",
    )

    emit_event(
        "model.call",
        payload={
            "api_key": "secret-123",
            "model": "qwen3:8b",
            "nested": {
                "password": "123456",
            },
        },
    )

    event_logger = logging.getLogger("hlw.events")

    for handler in event_logger.handlers:
        handler.flush()

    content = (
        tmp_path / "api" / "events.jsonl"
    ).read_text(encoding="utf-8")

    record = json.loads(content)

    assert record["payload"]["api_key"] == "[REDACTED]"
    assert record["payload"]["nested"]["password"] == "[REDACTED]"
    assert record["payload"]["model"] == "qwen3:8b"


def test_invalid_duration_rejected():
    with pytest.raises(ValueError):
        emit_event(
            "test.event",
            duration_ms=-1,
        )
def test_safe_emit_event_handles_failure(monkeypatch):
    import hlw.core.event_logger as event_module

    def failing_writer(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        event_module,
        "emit_event",
        failing_writer,
    )

    assert (
        event_module.safe_emit_event(
            "test.event",
            payload={"message": "hello"},
        )
        is False
    )