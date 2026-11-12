
"""HLW MemoryConfirmationService 单元测试。"""

import asyncio
from uuid import uuid4

import pytest
from langgraph.store.memory import InMemoryStore

from hlw.memory.confirmation_service import MemoryConfirmationService
from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def memory_system():
    """构造完整的独立记忆测试环境。"""

    store = InMemoryStore()

    manager = MemoryManager(store)
    pending_manager = PendingMemoryManager(store)

    execution_locks = MemoryExecutionLocks()

    deduplicator = MemoryDeduplicator(
        manager,
        execution_locks,
    )

    service = MemoryConfirmationService(
        pending_manager,
        deduplicator,
        execution_locks,
    )

    return service, manager, pending_manager, deduplicator


def make_candidate() -> MemoryCandidate:
    return MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢英文代码注释",
        evidence="喜欢英文代码注释",
    )


async def create_pending_record(
    manager,
    pending_manager,
    candidate=None,
):
    """创建一条已有正式记忆及其待确认更新记录。"""

    existing = await manager.save_memory(
        user_id="user_001",
        key="coding_style",
        content="喜欢中文代码注释",
        category="preference",
    )

    candidate = candidate or make_candidate()

    return await pending_manager.create_pending(
        user_id="user_001",
        candidate=candidate,
        existing_snapshot=existing,
    )


@pytest.mark.anyio
async def test_confirm_updates_memory_and_consumes_pending(
    memory_system,
):
    """确认后应该更新正式记忆并删除待确认记录。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    result = await service.confirm(
        "user_001",
        pending["pending_id"],
    )

    assert result.status == "confirmed"
    assert result.deduplication_result.status == "updated"

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved["content"] == "喜欢英文代码注释"

    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is None


@pytest.mark.anyio
async def test_reject_does_not_modify_existing_memory(
    memory_system,
):
    """拒绝后只能删除待确认记录，不能修改正式记忆。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    result = await service.reject(
        "user_001",
        pending["pending_id"],
    )

    assert result.status == "rejected"
    assert result.deduplication_result is None

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved["content"] == "喜欢中文代码注释"

    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is None


@pytest.mark.anyio
async def test_confirm_rejects_stale_snapshot(
    memory_system,
):
    """原有记忆变化后，不能使用过期快照覆盖。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    # 模拟用户确认之前，正式记忆发生变化。
    await manager.update_memory(
        user_id="user_001",
        key="coding_style",
        content="喜欢日文代码注释",
    )

    result = await service.confirm(
        "user_001",
        pending["pending_id"],
    )

    assert result.status == "stale"
    assert result.deduplication_result.status == "conflict"

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved["content"] == "喜欢日文代码注释"

    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is None


@pytest.mark.anyio
async def test_confirm_already_applied_memory(
    memory_system,
):
    """目标记忆已经存在时，不应该重复更新。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    # 模拟目标内容已经由其他操作写入。
    await manager.update_memory(
        user_id="user_001",
        key="coding_style",
        content="喜欢英文代码注释",
    )

    result = await service.confirm(
        "user_001",
        pending["pending_id"],
    )

    assert result.status == "already_applied"
    assert result.deduplication_result.status == "unchanged"

    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is None


@pytest.mark.anyio
async def test_confirm_missing_pending_returns_not_found(
    memory_system,
):
    """不存在的待确认记录应该返回 not_found。"""

    service, _, _, _ = memory_system

    result = await service.confirm(
        "user_001",
        str(uuid4()),
    )

    assert result.status == "not_found"


@pytest.mark.anyio
async def test_reject_missing_pending_returns_not_found(
    memory_system,
):
    """拒绝不存在的记录应该返回 not_found。"""

    service, _, _, _ = memory_system

    result = await service.reject(
        "user_001",
        str(uuid4()),
    )

    assert result.status == "not_found"


@pytest.mark.anyio
async def test_confirmation_respects_user_isolation(
    memory_system,
):
    """其他用户不能确认该用户的待确认记录。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    result = await service.confirm(
        "user_002",
        pending["pending_id"],
    )

    assert result.status == "not_found"

    # 原用户的待确认记录仍然存在。
    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is not None

    # 正式记忆也没有变化。
    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved["content"] == "喜欢中文代码注释"


@pytest.mark.anyio
async def test_confirm_rejects_invalid_pending_id(
    memory_system,
):
    """非法 UUID 必须被拒绝。"""

    service, _, _, _ = memory_system

    with pytest.raises(ValueError, match="pending_id"):
        await service.confirm(
            "user_001",
            "invalid-pending-id",
        )


@pytest.mark.anyio
async def test_concurrent_confirm_is_idempotent(
    memory_system,
):
    """同一条待确认记录不能被重复执行。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    first, second = await asyncio.gather(
        service.confirm(
            "user_001",
            pending["pending_id"],
        ),
        service.confirm(
            "user_001",
            pending["pending_id"],
        ),
    )

    assert {first.status, second.status} == {
        "confirmed",
        "not_found",
    }

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved["content"] == "喜欢英文代码注释"


@pytest.mark.anyio
async def test_concurrent_confirm_and_reject_are_serialized(
    memory_system,
):
    """同一条记录的确认与拒绝必须串行执行。"""

    service, manager, pending_manager, _ = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    confirm_result, reject_result = await asyncio.gather(
        service.confirm(
            "user_001",
            pending["pending_id"],
        ),
        service.reject(
            "user_001",
            pending["pending_id"],
        ),
    )

    statuses = {
        confirm_result.status,
        reject_result.status,
    }

    assert statuses in (
        {"confirmed", "not_found"},
        {"rejected", "not_found"},
    )

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    if "confirmed" in statuses:
        assert saved["content"] == "喜欢英文代码注释"
    else:
        assert saved["content"] == "喜欢中文代码注释"

    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is None


@pytest.mark.anyio
async def test_failed_confirmation_preserves_pending_record(
    memory_system,
    monkeypatch,
):
    """更新异常时保留待确认记录，以便后续重试。"""

    service, manager, pending_manager, deduplicator = memory_system

    pending = await create_pending_record(
        manager,
        pending_manager,
    )

    async def fail_confirm_update(**kwargs):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(
        deduplicator,
        "confirm_update",
        fail_confirm_update,
    )

    with pytest.raises(RuntimeError, match="database failure"):
        await service.confirm(
            "user_001",
            pending["pending_id"],
        )

    assert await pending_manager.get_pending(
        "user_001",
        pending["pending_id"],
    ) is not None

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved["content"] == "喜欢中文代码注释"
