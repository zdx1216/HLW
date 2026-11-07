"""HLW 结构化事件日志。

统一记录 API 和 LangGraph Dev 的运行事件。
每条事件写入独立 JSONL 行，便于后续数据分析。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from hlw.core.request_context import get_request_id

logger = logging.getLogger("hlw.events")

SCHEMA_VERSION = "1.0"

SENSITIVE_KEY_PATTERN = re.compile(
    r"api[_-]?key|authorization|password|passwd|secret|"
    r"access[_-]?token|refresh[_-]?token|cookie|"
    r"private[_-]?key|database[_-]?url|dsn",
    re.IGNORECASE,
)

SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|password|secret|access[_-]?token|"
    r"refresh[_-]?token)\b(\s*[:=]\s*)([^\s,;]+)"
)

BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")

API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")


def _redact_text(value: str) -> str:
    """脱敏文本中的常见凭据。"""

    value = SECRET_ASSIGNMENT_PATTERN.sub(
        r"\1\2[REDACTED]",
        value,
    )

    value = BEARER_PATTERN.sub(
        "Bearer [REDACTED]",
        value,
    )

    return API_KEY_PATTERN.sub(
        "[REDACTED_KEY]",
        value,
    )


def _json_safe(value: Any) -> Any:
    """将复杂对象转换为可 JSON 序列化的数据，并递归脱敏。"""

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, Enum):
        return _json_safe(value.value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return {
            "_type": "bytes",
            "size": len(value),
        }

    if isinstance(value, Mapping):
        result = {}

        for key, item in value.items():
            key_string = str(key)

            if SENSITIVE_KEY_PATTERN.search(key_string):
                result[key_string] = "[REDACTED]"
            else:
                result[key_string] = _json_safe(item)

        return result

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except (TypeError, ValueError):
            return _redact_text(str(value))

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    return _redact_text(str(value))


def emit_event(
    event: str,
    *,
    payload: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    run_id: str | None = None,
    duration_ms: float | None = None,
    status: str | None = None,
) -> None:
    """记录一条结构化事件。"""

    if not isinstance(event, str) or not event.strip():
        raise ValueError("event 不能为空")

    if duration_ms is not None and duration_ms < 0:
        raise ValueError("duration_ms 不能小于零")

    current_request_id = request_id or get_request_id()

    if current_request_id == "-":
        current_request_id = None

    current_trace_id = trace_id or current_request_id

    runtime = getattr(
        logger,
        "_hlw_runtime",
        "unknown",
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "runtime": runtime,
        "event": event,
        "trace_id": current_trace_id,
        "request_id": current_request_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "duration_ms": duration_ms,
        "status": status,
        "payload": _json_safe(payload or {}),
    }

    logger.info(
        "%s",
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def safe_emit_event(
    event: str,
    **kwargs: Any,
) -> bool:
    """尽力记录事件，日志故障不得影响 Agent 执行。

    True 表示事件写入过程未抛异常。
    False 表示事件采集失败，并非业务执行失败。
    """
    try:
        emit_event(event, **kwargs)
    except Exception as exc:  # noqa: BLE001
        # 不记录日志内容及异常原文，避免泄露凭证。
        # 诊断信息写入普通日志，而不是 JSONL 事件日志。
        try:
            logging.getLogger("hlw.core.event_logger").warning(
                "Event capture failed: event=%s error_type=%s",
                event,
                type(exc).__name__,
            )
        except Exception:   # noqa: BLE001, S110
            # 即使普通诊断日志也失败，仍然不能影响业务。
            pass

        return False

    return True
