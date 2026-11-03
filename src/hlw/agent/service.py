
"""HLW Agent 统一异步执行服务。"""

import asyncio
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from hlw.core.concurrency import AgentConcurrencyLimiter
from hlw.core.logging_config import get_logger
from hlw.core.request_context import request_context
from hlw.core.thread_locks import ThreadExecutionLocks

logger = get_logger(__name__)


class AgentService:
    """统一管理 Agent 请求上下文和并发执行。"""

    def __init__(
        self,
        agent_graph: Any,
        max_concurrent: int = 2,
    ):
        self._graph = agent_graph
        self._limiter = AgentConcurrencyLimiter(
            max_concurrent=max_concurrent
        )
        self._thread_locks = ThreadExecutionLocks()

    async def run(
        self,
        message: str,
        *,
        thread_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """异步执行一次 Agent 请求。"""

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message 不能为空")

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id 不能为空")

        rid = uuid4().hex if request_id is None else request_id

        if (
            not isinstance(rid, str)
            or not rid.strip()
            or "\n" in rid
            or "\r" in rid
        ):
            raise ValueError("request_id 格式不合法")

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            },
            "metadata": {
                "request_id": rid,
            },
            "recursion_limit": 30,
        }

        async def invoke_graph() -> dict[str, Any]:
            return await self._graph.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": message,
                        }
                    ]
                },
                config=config,
            )

        with request_context(rid):
            logger.info(
                "Agent request started request_id=%s thread_id=%s",
                rid,
                thread_id,
            )

            try:
                async with self._thread_locks.hold(thread_id):
                    result = await self._limiter.run(invoke_graph)

            except asyncio.CancelledError:
                logger.warning(
                    "Agent request cancelled request_id=%s",
                    rid,
                )
                raise

            except Exception:
                logger.exception(
                    "Agent request failed request_id=%s",
                    rid,
                )
                raise

            logger.info(
                "Agent request completed request_id=%s",
                rid,
            )

            return result
