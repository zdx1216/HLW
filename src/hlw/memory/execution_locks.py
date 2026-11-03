
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


@dataclass
class _MemoryLockEntry:
    """保存一把 Memory 锁及其使用情况。"""

    # 真正负责控制并发的异步锁。
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )

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

        # STEP 1：检查用户 ID。
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")

        # STEP 2：检查记忆 key。
        if (
            not isinstance(memory_key, str)
            or not memory_key.strip()
        ):
            raise ValueError("memory_key 不能为空")

        # STEP 3：构建锁的唯一标识。
        # 必须同时包含用户 ID 和记忆 key。
        lock_key = (
            user_id.strip(),
            memory_key.strip(),
        )

        # STEP 4：查询是否已经存在对应的锁。
        entry = self._entries.get(lock_key)

        # 如果不存在，就创建一把新锁。
        if entry is None:
            entry = _MemoryLockEntry()
            self._entries[lock_key] = entry

        # STEP 5：记录使用者数量。
        # 等待获取锁的任务也要计入，防止锁被提前删除。
        entry.users += 1

        try:
            # STEP 6：获取锁。
            # 如果其他任务正在使用同一把锁，
            # 当前任务就在这里等待。
            async with entry.lock:

                # STEP 7：把锁交给调用方。
                # 调用方可以在这里执行完整的读、判断、写操作。
                yield

        finally:
            # STEP 8：无论正常完成、异常还是取消任务，
            # 都必须减少锁的使用者数量。
            entry.users -= 1

            # STEP 9：没有任务使用或等待时，清理锁。
            if (
                entry.users == 0
                and self._entries.get(lock_key) is entry
            ):
                del self._entries[lock_key]
