"""HLW Memory 三层身份隔离与并发集成测试。"""

import asyncio

import pytest
from langgraph.store.memory import InMemoryStore

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.manager import MemoryManager
from hlw.memory.schemas import MemoryCandidate


@pytest.fixture
def anyio_backend() -> str:
    """指定使用 asyncio 后端。"""
    return "asyncio"


def make_candidate(content: str) -> MemoryCandidate:
    """创建测试用记忆候选。"""

    return MemoryCandidate(
        key="coding_style",
        category="preference",
        content=content,
        evidence=content,
    )


def build_memory_system():
    """构建完整的三层记忆系统。"""

    store = InMemoryStore()

    manager = MemoryManager(store)

    execution_locks = MemoryExecutionLocks()

    deduplicator = MemoryDeduplicator(
        manager,
        execution_locks,
    )

    return (
        store,
        manager,
        execution_locks,
        deduplicator,
    )


class PausingMemoryManager(MemoryManager):
    """用于测试并发顺序的 MemoryManager。"""

    def __init__(self, store):
        super().__init__(store)

        self.first_read_started = asyncio.Event()
        self.release_first_read = asyncio.Event()

        self.read_count = 0

    async def get_memory(
        self,
        user_id: str,
        key: str,
    ):
        self.read_count += 1

        # 第一次读取时暂停，模拟数据库操作延迟。
        if self.read_count == 1:
            self.first_read_started.set()

            await self.release_first_read.wait()

        return await super().get_memory(
            user_id,
            key,
        )


@pytest.mark.anyio
async def test_normalized_user_ids_share_one_deduplication_lock():
    """验证同一用户不同形式的 ID 共享同一把锁。"""

    store = InMemoryStore()

    manager = PausingMemoryManager(store)

    execution_locks = MemoryExecutionLocks()

    deduplicator = MemoryDeduplicator(
        manager,
        execution_locks,
    )

    candidate_1 = make_candidate(
        "喜欢中文代码注释"
    )

    candidate_2 = make_candidate(
        "喜欢英文代码注释"
    )

    # 第一个请求使用带空格的 ID。
    first_task = asyncio.create_task(
        deduplicator.process(
            " zdx ",
            candidate_1,
        )
    )

    # 等待第一个请求进入数据库读取阶段。
    await manager.first_read_started.wait()

    second_started = asyncio.Event()

    async def second_request():
        second_started.set()

        # 第二个请求使用规范化后的 ID。
        return await deduplicator.process(
            "zdx",
            candidate_2,
        )

    second_task = asyncio.create_task(
        second_request()
    )

    try:
        await second_started.wait()

        # 让第二个任务尝试获取锁。
        await asyncio.sleep(0)

        # 如果两个 ID 使用同一把锁，
        # 第二个任务不能进入数据库读取。
        assert manager.read_count == 1

    finally:
        manager.release_first_read.set()

        results = await asyncio.gather(
            first_task,
            second_task,
            return_exceptions=True,
        )

    first_result, second_result = results

    assert first_result.status == "created"
    assert second_result.status == "conflict"

    # 最终保留第一个请求写入的内容。
    saved = await manager.get_memory(
        "zdx",
        "coding_style",
    )

    assert saved is not None
    assert saved["content"] == "喜欢中文代码注释"

    assert execution_locks._entries == {}


@pytest.mark.anyio
async def test_different_users_have_independent_memories():
    """验证不同用户的同名记忆相互隔离。"""

    (
        _store,
        manager,
        execution_locks,
        deduplicator,
    ) = build_memory_system()

    candidate_1 = make_candidate(
        "喜欢中文代码注释"
    )

    candidate_2 = make_candidate(
        "喜欢英文代码注释"
    )

    # 两个用户同时保存同一个 Key。
    results = await asyncio.gather(
        deduplicator.process(
            "zdx",
            candidate_1,
        ),
        deduplicator.process(
            "user_002",
            candidate_2,
        ),
    )

    assert results[0].status == "created"
    assert results[1].status == "created"

    memory_1 = await manager.get_memory(
        "zdx",
        "coding_style",
    )

    memory_2 = await manager.get_memory(
        "user_002",
        "coding_style",
    )

    assert memory_1 is not None
    assert memory_2 is not None

    assert memory_1["content"] == "喜欢中文代码注释"
    assert memory_2["content"] == "喜欢英文代码注释"

    assert execution_locks._entries == {}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_user_id",
    [
        "../admin",
        "-admin",
        "local user",
        "a" * 65,
        "",
        None,
        123,
    ],
)
async def test_invalid_identity_rejected_before_memory_processing(
    invalid_user_id,
):
    """验证非法身份不能进入记忆处理流程。"""

    (
        _store,
        _manager,
        execution_locks,
        deduplicator,
    ) = build_memory_system()

    candidate = make_candidate(
        "喜欢中文代码注释"
    )

    with pytest.raises(ValueError):
        await deduplicator.process(
            invalid_user_id,
            candidate,
        )

    # 非法身份不能创建执行锁。
    assert execution_locks._entries == {}
