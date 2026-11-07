"""HLW MemoryManager PostgreSQL 集成测试。"""

import asyncio
import sys

from hlw.memory.manager import MemoryManager
from hlw.memory.store_postgres import open_postgres_store


async def main() -> None:
    """验证 MemoryManager 与真实 PostgreSQL Store 的集成。"""

    user_a = "hlw_memory_manager_test_a"
    user_b = "hlw_memory_manager_test_b"
    key = "learning_style"

    async with open_postgres_store() as store:
        manager = MemoryManager(store)

        try:
            # STEP 1：保存用户 A 的记忆
            await manager.save_memory(
                user_a,
                key,
                "喜欢详细中文代码注释",
                category="preference",
            )

            print("STEP 1: Memory save PASS")

            # STEP 2：保存用户 B 的同名记忆
            await manager.save_memory(
                user_b,
                key,
                "喜欢英文代码注释",
                category="preference",
            )

            print("STEP 2: Second user save PASS")

            # STEP 3：验证用户隔离
            memory_a = await manager.get_memory(
                user_a,
                key,
            )

            memory_b = await manager.get_memory(
                user_b,
                key,
            )

            assert memory_a["content"] == "喜欢详细中文代码注释"
            assert memory_b["content"] == "喜欢英文代码注释"

            print("STEP 3: User isolation PASS")

            # STEP 4：更新用户 A 的记忆
            updated = await manager.update_memory(
                user_a,
                key,
                content="喜欢非常详细的中文代码注释",
            )

            assert updated is not None
            assert updated["content"] == "喜欢非常详细的中文代码注释"

            print("STEP 4: Memory update PASS")

            # STEP 5：按类别检索
            memories = await manager.list_memories(
                user_a,
                category="preference",
            )

            assert len(memories) == 1
            assert memories[0]["key"] == key

            print("STEP 5: Memory listing PASS")

            # STEP 6：删除并验证
            deleted = await manager.delete_memory(
                user_a,
                key,
            )

            assert deleted is True

            assert (
                await manager.get_memory(
                    user_a,
                    key,
                )
                is None
            )

            # 用户 B 的记忆不受影响
            remaining = await manager.get_memory(
                user_b,
                key,
            )

            assert remaining is not None
            assert remaining["content"] == "喜欢英文代码注释"

            print("STEP 6: Memory deletion and isolation PASS")

            print("RESULT: PostgreSQL MemoryManager integration PASS")

        finally:
            # 清理本次测试数据，避免污染正式记忆
            await manager.delete_memory(user_a, key)
            await manager.delete_memory(user_b, key)


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
