"""HLW Agent 生命周期日志。

记录 Agent 开始、正常完成、用户问题、最终回答及耗时。
API 和 LangGraph Dev 共用本模块。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from threading import Lock
from time import perf_counter
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from hlw.core.event_logger import emit_event
from hlw.core.request_context import get_request_id

logger = logging.getLogger(__name__)

# 未正常结束的运行可能不会触发 after_agent。
# 定期删除过期计时状态，避免运行时间越长内存占用越大。
START_TTL_SECONDS = 3600
MAX_TRACKED_RUNS = 4096


def _as_id(value: Any) -> str | None:
    """将运行标识转换为非空字符串。"""
    if value is None:
        return None

    result = str(value).strip()
    return result or None


def _as_mapping(value: Any) -> Mapping:
    """安全获取映射结构。"""
    return value if isinstance(value, Mapping) else {}


def _runtime_fields(runtime: Runtime[Any]) -> dict[str, Any]:
    """统一提取 API / LangGraph Dev 运行标识。"""

    try:
        config = get_config()
    except RuntimeError:
        config = {}

    config = _as_mapping(config)
    configurable = _as_mapping(config.get("configurable"))
    metadata = _as_mapping(config.get("metadata"))

    execution = getattr(runtime, "execution_info", None)
    server = getattr(runtime, "server_info", None)

    request_id = get_request_id()

    if request_id == "-":
        request_id = configurable.get("langgraph_request_id") or metadata.get(
            "request_id"
        )

    run_id = (
        getattr(execution, "run_id", None)
        or config.get("run_id")
        or configurable.get("run_id")
    )

    thread_id = (
        getattr(execution, "thread_id", None)
        or configurable.get("thread_id")
        or metadata.get("thread_id")
    )

    user_id = configurable.get("user_id") or metadata.get("user_id")

    request_id = _as_id(request_id)
    run_id = _as_id(run_id)

    return {
        "request_id": request_id,
        "trace_id": request_id or run_id,
        "run_id": run_id,
        "thread_id": _as_id(thread_id),
        "user_id": _as_id(user_id),
        "assistant_id": _as_id(getattr(server, "assistant_id", None)),
        "graph_id": _as_id(getattr(server, "graph_id", None)),
    }


def _latest_message(
    state: Any,
    message_types: set[str],
) -> Any:
    """从 Agent 状态中查找最后一条指定类型的消息。"""

    if not isinstance(state, Mapping):
        return None

    messages = state.get("messages", [])

    if not isinstance(messages, (list, tuple)):
        return None

    for message in reversed(messages):
        if isinstance(message, Mapping):
            kind = message.get("type") or message.get("role")
            content = message.get("content")
        else:
            kind = getattr(message, "type", None)
            content = getattr(message, "content", None)

        if kind in message_types:
            return content

    return None


def _safe_emit(
    event_name: str,
    *,
    fields: dict[str, Any],
    payload: dict[str, Any],
    status: str,
    duration_ms: float | None = None,
) -> None:
    """日志写入失败时，不中断 Agent 的业务执行。"""

    try:
        emit_event(
            event_name,
            payload=payload,
            request_id=fields["request_id"],
            trace_id=fields["trace_id"],
            thread_id=fields["thread_id"],
            run_id=fields["run_id"],
            user_id=fields["user_id"],
            duration_ms=duration_ms,
            status=status,
        )
    except Exception as exc: # noqa: BLE001
        # 不输出业务内容或可能包含密钥的异常原文。
        logger.warning(
            "HLW event logging failed: event=%s error_type=%s",
            event_name,
            type(exc).__name__,
        )


class AgentRunAuditMiddleware(AgentMiddleware):
    """记录 Agent 级别的运行过程。"""

    def __init__(self) -> None:
        super().__init__()

        # 按 Run ID 或 Request ID 隔离计时。
        self._starts: dict[str, float] = {}
        self._lock = Lock()

    @staticmethod
    def _tracking_key(fields: Mapping[str, Any]) -> str | None:
        """优先按 Run ID 关联，不同请求不能共用计时。"""

        if fields.get("run_id"):
            return f"run:{fields['run_id']}"

        if fields.get("request_id"):
            return f"request:{fields['request_id']}"

        return None

    def _register_start(
        self,
        key: str | None,
        started: float,
    ) -> None:
        """保存开始时间并清理过期计时状态。"""

        if key is None:
            return

        with self._lock:
            expired = [
                item
                for item, timestamp in self._starts.items()
                if started - timestamp > START_TTL_SECONDS
            ]

            for item in expired:
                self._starts.pop(item, None)

            self._starts[key] = started

            # 限制最大记录数量，防止异常中断造成无限累积。
            while len(self._starts) > MAX_TRACKED_RUNS:
                oldest = min(
                    self._starts,
                    key=self._starts.get,
                )
                self._starts.pop(oldest, None)

    def _finish(
        self,
        key: str | None,
    ) -> float | None:
        """完成计时并清理当前运行状态。"""

        if key is None:
            return None

        with self._lock:
            started = self._starts.pop(key, None)

        if started is None:
            return None

        return (perf_counter() - started) * 1000

    def before_agent(
        self,
        state: Any,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Agent 开始时记录请求与输入。"""

        fields = _runtime_fields(runtime)
        key = self._tracking_key(fields)

        self._register_start(key, perf_counter())

        user_message = _latest_message(
            state,
            {"human", "user"},
        )

        _safe_emit(
            "agent.run.started",
            fields=fields,
            payload={
                "input": user_message,
                "assistant_id": fields["assistant_id"],
                "graph_id": fields["graph_id"],
            },
            status="started",
        )

        # 不修改 Agent 状态。
        return None

    def after_agent(
        self,
        state: Any,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Agent 正常结束时记录最终回答和耗时。"""

        fields = _runtime_fields(runtime)
        key = self._tracking_key(fields)

        duration_ms = self._finish(key)

        final_answer = _latest_message(
            state,
            {"ai", "assistant"},
        )

        _safe_emit(
            "agent.run.completed",
            fields=fields,
            payload={
                "output": final_answer,
                "assistant_id": fields["assistant_id"],
                "graph_id": fields["graph_id"],
            },
            duration_ms=duration_ms,
            status="success",
        )

        # 不修改 Agent 状态。
        return None
