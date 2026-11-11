"""HLW 长期记忆的异步执行锁。

负责：
1. 确保同一用户的同一条记忆串行修改。
2. 允许不同用户之间并发操作。
3. 允许同一用户的不同记忆并发操作。
4. 自动清理不再使用的锁对象。

注意：
本模块使用 asyncio.Lock，只能保护同一进程、
同一事件循环中的协程。
不提供跨进程、跨容器的并发保护。
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from hlw.core.identity import normalize_user_id


@dataclass
class _MemoryLockEntry:
    """保存一把 Memory 锁及其使用情况。"""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # 当前正在持锁或等待获取锁的任务数量。
    users: int = 0


class MemoryExecutionLocks:
    """管理用户长期记忆的并发访问。"""

    def __init__(self) -> None:
        # 每个 (user_id, memory_key) 对应一把独立的锁。
        self._entries: dict[
            tuple[str, str],
            _MemoryLockEntry,
        ] = {}

    @asynccontextmanager
    async def hold(
        self,
        user_id: str,
        memory_key: str,
    ) -> AsyncGenerator[None, None]:
        """获取指定用户指定记忆的执行锁。

        同一用户、同一记忆：
            必须串行执行。

        同一用户、不同记忆：
            允许并发执行。

        不同用户、相同记忆 key：
            允许并发执行。
        """

        # STEP 1：使用全局统一规则规范化用户 ID。
        normalized_user_id = normalize_user_id(user_id)

        # STEP 2：检查记忆 key。
        if not isinstance(memory_key, str) or not memory_key.strip():
            raise ValueError("memory_key 不能为空")

        normalized_memory_key = memory_key.strip()

        # STEP 3：构建锁的唯一标识。
        lock_key = (
            normalized_user_id,
            normalized_memory_key,
        )

        # STEP 4：查询是否已经存在对应的锁。
        entry = self._entries.get(lock_key)

        if entry is None:
            entry = _MemoryLockEntry()
            self._entries[lock_key] = entry

        # STEP 5：记录使用者数量。
        # 等待获取锁的任务也要计入。
        entry.users += 1

        try:
            # STEP 6：获取锁。
            async with entry.lock:
                # STEP 7：把锁交给调用方。
                yield

        finally:
            # STEP 8：无论正常完成、异常还是取消任务，
            # 都必须减少锁的使用者数量。
            entry.users -= 1

            # STEP 9：没有任务使用或等待时，清理锁。
            if entry.users == 0 and self._entries.get(lock_key) is entry:
                del self._entries[lock_key]

# ???????????
memory_execution_locks = MemoryExecutionLocks()
