
"""HLW MemoryExecutionLocks 单元测试。

验证：
1. 同一用户、同一记忆必须串行执行。
2. 同一用户、不同记忆允许并发。
3. 不同用户、相同记忆允许并发。
4. 异常发生后正确释放锁。
5. 等待任务取消后正确清理锁。
"""

import asyncio

import pytest

from hlw.memory.execution_locks import MemoryExecutionLocks


@pytest.fixture
def anyio_backend() -> str:
    """指定使用 asyncio 后端。"""
    return "asyncio"


@pytest.fixture
def memory_locks() -> MemoryExecutionLocks:
    """创建独立的 Memory 锁管理器。"""
    return MemoryExecutionLocks()


@pytest.mark.anyio
async def test_same_memory_serialization(memory_locks):
    """同一用户的同一条记忆必须串行执行。"""

    first_entered = asyncio.Event()
    second_attempting = asyncio.Event()
    second_entered = asyncio.Event()
    release_first = asyncio.Event()

    execution_order = []

    async def first_task():
        async with memory_locks.hold(
            "user_001",
            "coding_style",
        ):
            execution_order.append("first_start")
            first_entered.set()

            await release_first.wait()

            execution_order.append("first_end")

    async def second_task():
        await first_entered.wait()

        second_attempting.set()

        async with memory_locks.hold(
            "user_001",
            "coding_style",
        ):
            execution_order.append("second_start")
            second_entered.set()

    task_1 = asyncio.create_task(first_task())
    task_2 = asyncio.create_task(second_task())

    try:
        await first_entered.wait()
        await second_attempting.wait()

        # 让第二个任务有机会尝试获取锁。
        await asyncio.sleep(0)

        # 第一个任务没有结束，第二个任务不能进入。
        assert not second_entered.is_set()

    finally:
        release_first.set()

        await asyncio.gather(
            task_1,
            task_2,
        )

    assert execution_order == [
        "first_start",
        "first_end",
        "second_start",
    ]

    assert memory_locks._entries == {}


@pytest.mark.anyio
async def test_different_memories_allow_concurrency(
    memory_locks,
):
    """同一用户的不同记忆允许并发执行。"""

    both_entered = asyncio.Event()
    release_tasks = asyncio.Event()

    entered_count = 0

    async def worker(memory_key):
        nonlocal entered_count

        async with memory_locks.hold(
            "user_001",
            memory_key,
        ):
            entered_count += 1

            if entered_count == 2:
                both_entered.set()

            await release_tasks.wait()

    tasks = [
        asyncio.create_task(
            worker("coding_style")
        ),
        asyncio.create_task(
            worker("work_project")
        ),
    ]

    try:
        # 两个不同的记忆应该能够同时进入临界区。
        await asyncio.wait_for(
            both_entered.wait(),
            timeout=2,
        )
    finally:
        release_tasks.set()
        await asyncio.gather(*tasks)

    assert entered_count == 2
    assert memory_locks._entries == {}


@pytest.mark.anyio
async def test_different_users_allow_concurrency(
    memory_locks,
):
    """不同用户的同名记忆允许并发执行。"""

    both_entered = asyncio.Event()
    release_tasks = asyncio.Event()

    entered_count = 0

    async def worker(user_id):
        nonlocal entered_count

        async with memory_locks.hold(
            user_id,
            "coding_style",
        ):
            entered_count += 1

            if entered_count == 2:
                both_entered.set()

            await release_tasks.wait()

    tasks = [
        asyncio.create_task(
            worker("user_001")
        ),
        asyncio.create_task(
            worker("user_002")
        ),
    ]

    try:
        await asyncio.wait_for(
            both_entered.wait(),
            timeout=2,
        )
    finally:
        release_tasks.set()
        await asyncio.gather(*tasks)

    assert entered_count == 2
    assert memory_locks._entries == {}


@pytest.mark.anyio
async def test_exception_releases_memory_lock(
    memory_locks,
):
    """任务发生异常后，锁必须正确释放。"""

    with pytest.raises(RuntimeError):

        async with memory_locks.hold(
            "user_001",
            "coding_style",
        ):
            raise RuntimeError("模拟任务异常")

    assert memory_locks._entries == {}


@pytest.mark.anyio
async def test_cancelled_waiter_cleanup(
    memory_locks,
):
    """等待获取锁的任务被取消后，正确清理计数。"""

    first_entered = asyncio.Event()
    second_attempting = asyncio.Event()
    release_first = asyncio.Event()

    async def first_task():
        async with memory_locks.hold(
            "user_001",
            "coding_style",
        ):
            first_entered.set()
            await release_first.wait()

    async def second_task():
        second_attempting.set()

        async with memory_locks.hold(
            "user_001",
            "coding_style",
        ):
            pass

    task_1 = asyncio.create_task(first_task())

    await first_entered.wait()

    task_2 = asyncio.create_task(second_task())

    try:
        await second_attempting.wait()

        # 让第二个任务进入锁等待阶段。
        await asyncio.sleep(0)

        entry = memory_locks._entries[
            ("user_001", "coding_style")
        ]

        # 一个持锁，一个等待，共两个使用者。
        assert entry.users == 2

        # 取消第二个等待任务。
        task_2.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task_2

        # 第二个任务取消后，使用者数量恢复为 1。
        assert entry.users == 1

    finally:
        release_first.set()
        await task_1

    # 所有任务结束后，锁对象应该被清理。
    assert memory_locks._entries == {}

@pytest.mark.anyio
async def test_user_id_is_normalized_before_lock_key(
    memory_locks,
):
    """验证带首尾空格的用户 ID 使用规范化后的锁键。"""

    async with memory_locks.hold(
        " zdx ",
        "coding_style",
    ):
        # 锁键应该使用规范化后的用户 ID。
        assert (
            "zdx",
            "coding_style",
        ) in memory_locks._entries

        # 不应该保留带空格的原始 ID。
        assert (
            " zdx ",
            "coding_style",
        ) not in memory_locks._entries

    # 退出后锁应该被清理。
    assert memory_locks._entries == {}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_user_id",
    [
        "../admin",
        "-admin",
        "local user",
        "a" * 65,
        "",
        "   ",
        None,
        123,
    ],
)
async def test_invalid_user_id_rejected(
    memory_locks,
    invalid_user_id,
):
    """验证非法用户 ID 无法获取记忆执行锁。"""

    with pytest.raises(ValueError):
        async with memory_locks.hold(
            invalid_user_id,
            "coding_style",
        ):
            pass

    # 非法 ID 不应该创建任何锁对象。
    assert memory_locks._entries == {}
