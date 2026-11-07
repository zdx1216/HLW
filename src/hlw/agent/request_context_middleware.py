"""LangGraph Agent 请求上下文 Middleware。"""

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langgraph.config import get_config

from hlw.core.request_context import get_request_id, request_context


class RequestContextMiddleware(AgentMiddleware):
    """统一 API 与 LangGraph Dev 的模型调用请求上下文。"""

    @staticmethod
    def _resolve_request_id(request: ModelRequest[Any]) -> str | None:
        """按优先级获取当前请求 ID。"""

        # 1. 优先使用 API AgentService 已经建立的请求上下文。
        current_id = get_request_id()

        if current_id != "-":
            return current_id

        # 2. 获取当前 LangGraph RunnableConfig。
        try:
            config = get_config()
        except RuntimeError:
            config = {}

        configurable = config.get("configurable", {})

        # 3. 优先读取 LangGraph Server 的 HTTP 请求 ID。
        if isinstance(configurable, Mapping):
            request_id = configurable.get("langgraph_request_id")

            if isinstance(request_id, str) and request_id.strip():
                return request_id.strip()

        # 4. 如果没有 HTTP 请求 ID，使用 Graph Run ID 兜底。
        runtime = getattr(request, "runtime", None)
        execution_info = getattr(runtime, "execution_info", None)
        run_id = getattr(execution_info, "run_id", None)

        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()

        return None

    @contextmanager
    def _request_scope(
        self,
        request: ModelRequest[Any],
    ) -> Iterator[None]:
        """在模型调用期间建立请求上下文。"""

        request_id = self._resolve_request_id(request)

        if request_id is None or request_id == get_request_id():
            yield
            return

        with request_context(request_id):
            yield

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
