"""HLW Agent 统一异步执行服务。"""

import asyncio
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from hlw.core.concurrency import AgentConcurrencyLimiter
from hlw.core.config import settings
from hlw.core.identity import normalize_user_id
from hlw.core.logging_config import get_logger
from hlw.core.request_context import request_context
from hlw.core.thread_locks import ThreadExecutionLocks

logger = get_logger(__name__)


class AgentService:
    """统一管理 Agent 请求上下文、用户身份和并发执行。"""

    def __init__(
        self,
        agent_graph: Any,
        max_concurrent: int = 2,
        *,
        user_id: str | None = None,
    ):
        self._graph = agent_graph

        # 第一版采用单用户模式。
        # 默认读取服务端配置的用户身份。
        configured_user_id = settings.user_id if user_id is None else user_id

        # 统一校验用户身份。
        self._user_id = normalize_user_id(configured_user_id)

        # 控制 Agent 最大并发请求数。
        self._limiter = AgentConcurrencyLimiter(max_concurrent=max_concurrent)

        # 防止同一个 Thread ID 同时执行多个请求。
        self._thread_locks = ThreadExecutionLocks()

    async def run(
        self,
        message: str,
        *,
        thread_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """异步执行一次 Agent 请求。"""

        # ==================================================
        # 1. 校验用户消息
        # ==================================================

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message 不能为空")

        # ==================================================
        # 2. 校验会话 ID
        # ==================================================

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id 不能为空")

        # ==================================================
        # 3. 创建或校验请求 ID
        # ==================================================

        rid = uuid4().hex if request_id is None else request_id

        if not isinstance(rid, str) or not rid.strip() or "\n" in rid or "\r" in rid:
            raise ValueError("request_id 格式不合法")

        # ==================================================
        # 4. 构建 LangGraph 运行配置
        # ==================================================

        # 每次 Agent 执行分配独立的根运行 ID。
        # request_id 对应业务请求；run_id 对应本次图执行。
        run_id = uuid4()

        config: RunnableConfig = {
            "run_id": run_id,
            "configurable": {
                "thread_id": thread_id,
                "user_id": self._user_id,
            },
            "metadata": {
                "request_id": rid,
                "user_id": self._user_id,
            },
            "recursion_limit": 30,
        }

        # ==================================================
        # 5. 定义 Agent 执行函数
        # ==================================================

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

        # ==================================================
        # 6. 执行 Agent 请求
        # ==================================================

        with request_context(rid):
            logger.info(
                "Agent request started request_id=%s user_id=%s thread_id=%s",
                rid,
                self._user_id,
                thread_id,
            )

            try:
                # 同一个 Thread ID 串行执行。
                async with self._thread_locks.hold(thread_id):
                    # 同时限制整个 Agent 服务的并发数量。
                    result = await self._limiter.run(invoke_graph)

            except asyncio.CancelledError:
                logger.warning(
                    "Agent request cancelled request_id=%s user_id=%s",
                    rid,
                    self._user_id,
                )

                raise

            except Exception:
                logger.exception(
                    "Agent request failed request_id=%s user_id=%s",
                    rid,
                    self._user_id,
                )

                raise

            logger.info(
                "Agent request completed request_id=%s user_id=%s",
                rid,
                self._user_id,
            )

            return result
