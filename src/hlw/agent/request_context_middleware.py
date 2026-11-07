
"""LangGraph Agent 请求上下文 Middleware。"""

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langgraph.config import get_config

from hlw.core.request_context import get_request_id, request_context


class RequestContextMiddleware(AgentMiddleware):
    """统一模型调用和工具调用的请求上下文。"""

    @staticmethod
    def _resolve_request_id(request: Any) -> str | None:
        """按照优先级解析当前请求 ID。"""

        # 1. 优先使用已经建立的上下文
        current_id = get_request_id()

        if current_id != "-":
            return current_id

        # 2. 优先从 Runtime 获取运行配置
        runtime = getattr(request, "runtime", None)
        config = getattr(runtime, "config", None)

        # 3. 如果 Runtime 没有提供配置，则读取 RunnableConfig
        if not isinstance(config, Mapping) or not config:
            try:
                config = get_config()
            except RuntimeError:
                config = {}

        if not isinstance(config, Mapping):
            config = {}

        # 4. 优先读取 LangGraph Server 请求 ID
        configurable = config.get("configurable", {})

        if isinstance(configurable, Mapping):
            request_id = configurable.get("langgraph_request_id")

            if isinstance(request_id, str) and request_id.strip():
                return request_id.strip()

        # 5. 读取 metadata 中的 request_id
        metadata = config.get("metadata", {})

        if isinstance(metadata, Mapping):
            request_id = metadata.get("request_id")

            if isinstance(request_id, str) and request_id.strip():
                return request_id.strip()

        # 6. 从 LangGraph ExecutionInfo 获取 Run ID
        execution_info = getattr(runtime, "execution_info", None)
        run_id = getattr(execution_info, "run_id", None)

        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()

        # 7. 最后尝试 RunnableConfig 中的 run_id
        run_id = config.get("run_id")

        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()

        return None

    @contextmanager
    def _request_scope(
        self,
        request: Any,
    ) -> Iterator[None]:
        """建立请求上下文，并在执行结束后自动恢复。"""

        request_id = self._resolve_request_id(request)

        if request_id is None or request_id == get_request_id():
            yield
            return

        with request_context(request_id):
            yield

    # ========================================================
    # 1. 模型调用
    # ========================================================

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]],
            ModelResponse[Any],
        ],
    ) -> ModelResponse[Any]:
        """同步模型调用上下文。"""

        with self._request_scope(request):
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelResponse[Any]:
        """异步模型调用上下文。"""

        with self._request_scope(request):
            return await handler(request)

    # ========================================================
    # 2. 工具调用
    # ========================================================

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """同步工具调用上下文。"""

        with self._request_scope(request):
            return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[Any],
        ],
    ) -> Any:
        """异步工具调用上下文。"""

        with self._request_scope(request):
            return await handler(request)
