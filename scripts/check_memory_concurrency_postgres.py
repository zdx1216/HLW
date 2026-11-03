
"""HLW PostgreSQL Memory 并发安全实验。

实验一：
同一用户、同一记忆，同时提交两个不同内容。
预期：一个创建成功，另一个检测到冲突。

实验二：
不同用户、相同记忆 key，同时提交。
预期：两个请求均能独立创建，允许并发执行。
"""

import asyncio
import sys
from typing import Any
from uuid import uuid4

from langgraph.store.base import BaseStore

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.manager import MemoryManager
from hlw.memory.schemas import MemoryCandidate
from hlw.memory.store_postgres import open_postgres_store


class ObservedMemoryManager(MemoryManager):
    """用于观察数据库读取并发情况的测试管理器。"""

    def __init__(
        self,
        store: BaseStore,
        delay_seconds: float = 0.25,
    ) -> None:
        super().__init__(store)

        self.delay_seconds = delay_seconds

        # 当前正在执行 get_memory 的任务数量。
        self.active_reads = 0

        # 记录同时执行读取操作的最大数量。
        self.max_active_reads = 0

    async def get_memory(
        self,
        user_id: str,
        key: str,
    ) -> dict[str, Any] | None:
        """增加测试延迟，观察多个请求是否同时进入。"""

        self.active_reads += 1

        self.max_active_reads = max(
            self.max_active_reads,
            self.active_reads,
        )

        try:
            # 人为制造一个等待窗口。
            # 仅用于观察并发，不用于正式业务。
            await asyncio.sleep(self.delay_seconds)

            return await super().get_memory(
                user_id,
                key,
            )

        finally:
            self.active_reads -= 1

    def reset_metrics(self) -> None:
        """重置并发统计数据。"""

        self.active_reads = 0
        self.max_active_reads = 0


def make_candidate(
    key: str,
    content: str,
    evidence: str,
) -> MemoryCandidate:
    """创建测试候选记忆。"""

    return MemoryCandidate(
        key=key,
        category="preference",
        content=content,
        evidence=evidence,
    )


async def main() -> None:
    """执行 PostgreSQL 并发实验。"""

    suffix = uuid4().hex[:8]

    user_same = f"hlw_concurrency_same_{suffix}"
    user_a = f"hlw_concurrency_a_{suffix}"
    user_b = f"hlw_concurrency_b_{suffix}"

    key = "concurrency_probe"

    async with open_postgres_store() as store:

        # 创建共享数据库管理器。
        manager = ObservedMemoryManager(store)

        # 创建唯一的锁管理器。
        # 所有并发请求必须共享它。
        execution_locks = MemoryExecutionLocks()

        # 创建统一 Deduplicator。
        deduplicator = MemoryDeduplicator(
            manager,
            execution_locks,
        )

        try:

            # =====================================
            # 实验一：同一用户、同一条记忆
            # =====================================

            print("\nEXPERIMENT 1: Same user, same memory")

            candidate_a = make_candidate(
                key,
                "喜欢中文代码注释",
                "中文代码注释",
            )

            candidate_b = make_candidate(
                key,
                "喜欢英文代码注释",
                "英文代码注释",
            )

            manager.reset_metrics()

            # asyncio.gather 同时提交两个协程。
            results = await asyncio.gather(
                deduplicator.process(
                    user_same,
                    candidate_a,
                ),
                deduplicator.process(
                    user_same,
                    candidate_b,
                ),
            )

            statuses = [
                result.status
                for result in results
            ]

            print("Request A/B statuses:", statuses)

            print(
                "Maximum simultaneous reads:",
                manager.max_active_reads,
            )

            # 两个请求只有一个能创建成功。
            assert statuses.count("created") == 1
            assert statuses.count("conflict") == 1

            # 同一用户、同一记忆不能同时进入读取区域。
            assert manager.max_active_reads == 1

            saved = await manager.get_memory(
                user_same,
                key,
            )

            assert saved is not None

            assert saved["content"] in [
                "喜欢中文代码注释",
                "喜欢英文代码注释",
            ]

            print(
                "STEP 1: Same-user serialization PASS"
            )

            # =====================================
            # 实验二：不同用户、相同记忆 key
            # =====================================

            print("\nEXPERIMENT 2: Different users, same key")

            candidate_user_a = make_candidate(
                key,
                "用户 A 的记忆",
                "用户 A",
            )

            candidate_user_b = make_candidate(
                key,
                "用户 B 的记忆",
                "用户 B",
            )

            manager.reset_metrics()

            results = await asyncio.gather(
                deduplicator.process(
                    user_a,
                    candidate_user_a,
                ),
                deduplicator.process(
                    user_b,
                    candidate_user_b,
                ),
            )

            statuses = [
                result.status
                for result in results
            ]

            print("Request A/B statuses:", statuses)

            print(
                "Maximum simultaneous reads:",
                manager.max_active_reads,
            )

            # 两个用户的记忆相互独立。
            assert statuses == ["created", "created"]

            # 不同用户应当允许同时进入读取区域。
            assert manager.max_active_reads == 2

            memory_a = await manager.get_memory(
                user_a,
                key,
            )

            memory_b = await manager.get_memory(
                user_b,
                key,
            )

            assert memory_a is not None
            assert memory_b is not None

            assert memory_a["content"] == "用户 A 的记忆"
            assert memory_b["content"] == "用户 B 的记忆"

            print(
                "STEP 2: Different-user concurrency PASS"
            )

            print(
                "\nRESULT: PostgreSQL Memory concurrency PASS"
            )

        finally:

            # 清理本次实验数据。
            await manager.delete_memory(
                user_same,
                key,
            )

            await manager.delete_memory(
                user_a,
                key,
            )

            await manager.delete_memory(
                user_b,
                key,
            )

            print("CLEANUP: Test memories deleted")


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
