
"""HLW Agent 异步任务并发控制。"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class AgentConcurrencyLimiter:
    """限制同一事件循环内同时执行的 Agent 任务数。"""

    def __init__(self, max_concurrent: int = 2):
        if (
            not isinstance(max_concurrent, int)
            or isinstance(max_concurrent, bool)
            or max_concurrent < 1
        ):
            raise ValueError(
                "max_concurrent 必须是大于零的整数"
            )

        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(
            max_concurrent
        )

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        """取得执行许可，运行任务，并自动释放许可。"""

        async with self._semaphore:
            return await operation()
