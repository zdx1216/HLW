"""HLW MemoryPipeline PostgreSQL 集成测试。"""

import asyncio
import sys
from uuid import uuid4

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.manager import MemoryManager
from hlw.memory.pipeline import MemoryPipeline
from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryExtractionResult,
)
from hlw.memory.store_postgres import open_postgres_store
from hlw.memory.validator import MemoryValidator


class FakeMemoryExtractor(MemoryExtractor):
    """模拟 Extractor，不调用真实 LLM。"""

    def __init__(
        self,
        candidates: list[MemoryCandidate],
    ) -> None:
        self._result = MemoryExtractionResult(candidates=candidates)

    async def extract(
        self,
        user_message: str,
    ) -> MemoryExtractionResult:
        """直接返回预设的候选记忆。"""

        return self._result


def make_candidate(
    content: str,
    evidence: str,
) -> MemoryCandidate:
    """构造测试用候选记忆。"""

    return MemoryCandidate(
        key="coding_style",
        category="preference",
        content=content,
        evidence=evidence,
    )


def build_pipeline(
    manager: MemoryManager,
    execution_locks: MemoryExecutionLocks,
    candidates: list[MemoryCandidate],
) -> MemoryPipeline:
    """构建连接真实数据库的 MemoryPipeline。

    manager:
        提供 PostgreSQL Store 的读写能力。

    execution_locks:
        所有 Pipeline 共享的并发锁管理器。

    candidates:
        当前测试使用的模拟记忆。
    """

    return MemoryPipeline(
        extractor=FakeMemoryExtractor(candidates),
        validator=MemoryValidator(),
        deduplicator=MemoryDeduplicator(
            manager,
            execution_locks,
        ),
    )


async def main() -> None:
    """验证 MemoryPipeline 与真实 PostgreSQL 的集成。"""

    # 使用唯一测试用户 ID，避免与其他测试数据冲突。
    suffix = uuid4().hex[:8]

    user_a = f"hlw_pipeline_test_a_{suffix}"
    user_b = f"hlw_pipeline_test_b_{suffix}"

    key = "coding_style"

    async with open_postgres_store() as store:
        # STEP 0：创建统一的数据库管理器和并发锁。
        manager = MemoryManager(store)

        # 关键：整个测试只创建一个锁管理器。
        # 后续所有 Pipeline 都共享这个实例。
        execution_locks = MemoryExecutionLocks()

        try:
            # STEP 1：用户 A 首次保存记忆。
            candidate_a = make_candidate(
                content="喜欢详细的中文代码注释",
                evidence="详细中文代码注释",
            )

            pipeline_a = build_pipeline(
                manager,
                execution_locks,
                [candidate_a],
            )

            result_a = await pipeline_a.run(
                user_id=user_a,
                user_message="以后请给我提供详细中文代码注释。",
            )

            assert result_a.deduplication_results[0].status == "created"

            saved_a = await manager.get_memory(
                user_a,
                key,
            )

            assert saved_a is not None
            assert saved_a["content"] == "喜欢详细的中文代码注释"

            print("STEP 1: Pipeline memory creation PASS")

            # STEP 2：再次提交相同记忆。
            unchanged_result = await pipeline_a.run(
                user_id=user_a,
                user_message="以后请给我提供详细中文代码注释。",
            )

            assert unchanged_result.deduplication_results[0].status == "unchanged"

            print("STEP 2: Duplicate memory detection PASS")

            # STEP 3：提交不同内容，验证冲突保护。
            candidate_conflict = make_candidate(
                content="喜欢简洁的英文代码注释",
                evidence="简洁英文代码注释",
            )

            conflict_pipeline = build_pipeline(
                manager,
                execution_locks,
                [candidate_conflict],
            )

            conflict_result = await conflict_pipeline.run(
                user_id=user_a,
                user_message="我现在更喜欢简洁英文代码注释。",
            )

            assert conflict_result.deduplication_results[0].status == "conflict"

            # 验证原来的记忆没有被覆盖。
            saved_a = await manager.get_memory(
                user_a,
                key,
            )

            assert saved_a is not None
            assert saved_a["content"] == "喜欢详细的中文代码注释"

            print("STEP 3: Memory conflict protection PASS")

            # STEP 4：用户 B 保存相同 key 的记忆。
            candidate_b = make_candidate(
                content="喜欢简洁的英文代码注释",
                evidence="简洁英文代码注释",
            )

            pipeline_b = build_pipeline(
                manager,
                execution_locks,
                [candidate_b],
            )

            result_b = await pipeline_b.run(
                user_id=user_b,
                user_message="我喜欢简洁英文代码注释。",
            )

            assert result_b.deduplication_results[0].status == "created"

            # 验证用户 A 和用户 B 的记忆相互隔离。
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

            assert memory_a["content"] == "喜欢详细的中文代码注释"
            assert memory_b["content"] == "喜欢简洁的英文代码注释"

            print("STEP 4: Multi-user memory isolation PASS")

            print("RESULT: PostgreSQL MemoryPipeline integration PASS")

        finally:
            # 清理本次测试产生的数据。
            await manager.delete_memory(user_a, key)
            await manager.delete_memory(user_b, key)

            print("CLEANUP: Test memories deleted")


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
