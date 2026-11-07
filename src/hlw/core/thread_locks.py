"""HLW 同一会话的异步执行锁。"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class _LockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class ThreadExecutionLocks:
    """确保同一事件循环中相同 thread_id 的任务串行执行。"""

    def __init__(self) -> None:
        self._entries: dict[str, _LockEntry] = {}

    @asynccontextmanager
    async def hold(
        self,
        thread_id: str,
    ) -> AsyncGenerator[None, None]:
        """获取会话锁，并在无人使用时自动清理。"""

        entry = self._entries.get(thread_id)

        if entry is None:
            entry = _LockEntry()
            self._entries[thread_id] = entry

        # 同时统计持锁者与等待者
        entry.users += 1

        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1

            if entry.users == 0 and self._entries.get(thread_id) is entry:
                del self._entries[thread_id]
