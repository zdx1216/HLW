"""HLW MemoryDeduplicator 单元测试。"""

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


@pytest.fixture
def memory_system() -> tuple[MemoryDeduplicator, MemoryManager]:
    """构造独立的记忆管理测试环境。"""
    store = InMemoryStore()
    manager = MemoryManager(store)
    execution_locks = MemoryExecutionLocks()
    deduplicator = MemoryDeduplicator(manager, execution_locks)
    return deduplicator, manager


def make_candidate(
    key: str,
    content: str,
    category: str = "preference",
) -> MemoryCandidate:
    """构造测试用候选记忆。"""
    return MemoryCandidate(
        key=key,
        category=category,
        content=content,
        evidence=content,
    )


async def get_snapshot(
    manager: MemoryManager,
    user_id: str,
    key: str,
) -> dict[str, str]:
    """读取一条已有正式记忆的四字段快照。"""
    saved = await manager.get_memory(user_id, key)
    assert saved is not None
    return {
        field: saved[field]
        for field in ("content", "category", "created_at", "updated_at")
    }


@pytest.mark.anyio
async def test_create_new_memory(memory_system) -> None:
    """不存在的记忆应该被新增。"""
    deduplicator, manager = memory_system

    candidate = make_candidate("python_style", "喜欢中文代码注释")
    result = await deduplicator.process("user_001", candidate)

    assert result.status == "created"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved is not None
    assert saved["content"] == "喜欢中文代码注释"


@pytest.mark.anyio
async def test_skip_unchanged_memory(memory_system) -> None:
    """完全相同的记忆不应该重复保存。"""
    deduplicator, _ = memory_system

    candidate = make_candidate("python_style", "喜欢中文代码注释")
    await deduplicator.process("user_001", candidate)
    result = await deduplicator.process("user_001", candidate)

    assert result.status == "unchanged"


@pytest.mark.anyio
async def test_detect_conflict_without_confirmation(memory_system) -> None:
    """内容不同且没有更新授权时，应该保留旧记忆。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    result = await deduplicator.process("user_001", new_candidate)

    assert result.status == "conflict"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢中文代码注释"


@pytest.mark.anyio
async def test_update_memory_with_confirmation(memory_system) -> None:
    """有原始快照的明确授权可覆盖旧记忆。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    result = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert result.status == "updated"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢英文代码注释"


@pytest.mark.anyio
async def test_updated_memory_becomes_unchanged(memory_system) -> None:
    """更新成功后，再次提交相同内容应该跳过。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    updated = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert updated.status == "updated"

    result = await deduplicator.process("user_001", new_candidate)
    assert result.status == "unchanged"


@pytest.mark.anyio
async def test_different_users_are_isolated(memory_system) -> None:
    """不同用户使用相同 key 时，记忆应该相互隔离。"""
    deduplicator, manager = memory_system

    candidate_1 = make_candidate("python_style", "喜欢中文代码注释")
    candidate_2 = make_candidate("python_style", "喜欢英文代码注释")

    result_1 = await deduplicator.process("user_001", candidate_1)
    result_2 = await deduplicator.process("user_002", candidate_2)

    assert result_1.status == "created"
    assert result_2.status == "created"

    memory_1 = await manager.get_memory("user_001", "python_style")
    memory_2 = await manager.get_memory("user_002", "python_style")

    assert memory_1["content"] == "喜欢中文代码注释"
    assert memory_2["content"] == "喜欢英文代码注释"


@pytest.mark.anyio
async def test_category_difference_is_conflict(memory_system) -> None:
    """内容相同但类别不同，也应该识别为冲突。"""
    deduplicator, _ = memory_system

    candidate_1 = make_candidate(
        "python_style",
        "喜欢中文代码注释",
        category="preference",
    )
    candidate_2 = make_candidate(
        "python_style",
        "喜欢中文代码注释",
        category="project",
    )

    await deduplicator.process("user_001", candidate_1)
    result = await deduplicator.process("user_001", candidate_2)

    assert result.status == "conflict"


@pytest.mark.anyio
async def test_confirm_update(memory_system) -> None:
    """确认更新时携带原始快照，应该修改已有记忆。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    result = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert result.status == "updated"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved is not None
    assert saved["content"] == "喜欢英文代码注释"


@pytest.mark.anyio
async def test_confirm_update_does_not_recreate_deleted_memory(
    memory_system,
) -> None:
    """旧记忆已删除时，不能通过确认更新重新创建。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    await manager.delete_memory("user_001", "python_style")

    result = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert result.status == "conflict"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved is None


@pytest.mark.anyio
async def test_confirm_update_with_matching_snapshot(memory_system) -> None:
    """原始快照一致时，允许执行授权更新。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    result = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert result.status == "updated"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢英文代码注释"


@pytest.mark.anyio
async def test_confirm_update_rejects_stale_snapshot(memory_system) -> None:
    """原始记忆发生变化后，拒绝用过期快照覆盖。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    await manager.update_memory(
        "user_001",
        "python_style",
        content="喜欢简洁的中文代码注释",
    )

    result = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert result.status == "conflict"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢简洁的中文代码注释"


@pytest.mark.anyio
async def test_confirm_update_with_same_snapshot_is_idempotent(
    memory_system,
) -> None:
    """相同授权重复执行时，不应重复修改。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    snapshot = await get_snapshot(manager, "user_001", "python_style")

    first = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert first.status == "updated"

    second = await deduplicator.confirm_update(
        "user_001",
        new_candidate,
        expected_snapshot=snapshot,
    )
    assert second.status == "unchanged"

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢英文代码注释"


@pytest.mark.anyio
async def test_conflict_returns_original_memory_snapshot(
    memory_system,
) -> None:
    """发生冲突时返回完整的原始记忆快照。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)
    result = await deduplicator.process("user_001", new_candidate)

    assert result.status == "conflict"
    assert result.existing_snapshot is not None

    saved = await manager.get_memory("user_001", "python_style")
    for field in ("content", "category", "created_at", "updated_at"):
        assert result.existing_snapshot[field] == saved[field]

    assert saved["content"] == "喜欢中文代码注释"


@pytest.mark.anyio
async def test_process_cannot_bypass_confirmation(memory_system) -> None:
    """普通处理入口不得使用旧参数直接授权更新。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)

    with pytest.raises(TypeError, match="update_confirmed"):
        await deduplicator.process(
            "user_001",
            new_candidate,
            update_confirmed=True,
        )

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢中文代码注释"


@pytest.mark.anyio
async def test_confirm_update_requires_snapshot(memory_system) -> None:
    """确认更新不允许省略原始快照。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)

    with pytest.raises(TypeError, match="expected_snapshot"):
        await deduplicator.confirm_update("user_001", new_candidate)

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢中文代码注释"


@pytest.mark.anyio
async def test_confirm_update_rejects_none_snapshot(
    memory_system,
) -> None:
    """显式传入 None 也不能绕过快照校验。"""
    deduplicator, manager = memory_system

    old_candidate = make_candidate("python_style", "喜欢中文代码注释")
    new_candidate = make_candidate("python_style", "喜欢英文代码注释")

    await deduplicator.process("user_001", old_candidate)

    with pytest.raises(TypeError, match="expected_snapshot"):
        await deduplicator.confirm_update(
            "user_001",
            new_candidate,
            expected_snapshot=None,
        )

    saved = await manager.get_memory("user_001", "python_style")
    assert saved["content"] == "喜欢中文代码注释"
