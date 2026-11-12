
"""HLW PendingMemoryManager 单元测试。"""

from uuid import UUID, uuid4

import pytest
from langgraph.store.memory import InMemoryStore

from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def manager() -> PendingMemoryManager:
    """创建独立的内存 Store。"""
    return PendingMemoryManager(InMemoryStore())


def make_candidate() -> MemoryCandidate:
    return MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢简洁的英文代码注释",
        evidence="简洁的英文代码注释",
    )


def make_snapshot() -> dict:
    return {
        "content": "喜欢详细的中文代码注释",
        "category": "preference",
        "created_at": "2026-11-01T10:00:00+00:00",
        "updated_at": "2026-11-05T10:00:00+00:00",
    }


@pytest.mark.anyio
async def test_create_and_get_pending(manager):
    """测试待确认记录创建和读取。"""

    created = await manager.create_pending(
        user_id="user_001",
        candidate=make_candidate(),
        existing_snapshot=make_snapshot(),
    )

    # 验证 UUID。
    UUID(created["pending_id"])

    assert created["status"] == "pending"

    assert created["candidate"]["key"] == "coding_style"

    assert (
        created["existing_snapshot"]["content"]
        == "喜欢详细的中文代码注释"
    )

    assert created["created_at"] == created["updated_at"]

    loaded = await manager.get_pending(
        user_id="user_001",
        pending_id=created["pending_id"],
    )

    assert loaded == created


@pytest.mark.anyio
async def test_get_missing_pending_returns_none(manager):
    """不存在的待确认记录应该返回 None。"""

    result = await manager.get_pending(
        user_id="user_001",
        pending_id=str(uuid4()),
    )

    assert result is None


@pytest.mark.anyio
async def test_same_key_creates_independent_pending_records(manager):
    """相同记忆 Key 的不同待确认记录不能互相覆盖。"""

    first = await manager.create_pending(
        user_id="user_001",
        candidate=make_candidate(),
        existing_snapshot=make_snapshot(),
    )

    second_candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢简短的英文代码注释",
        evidence="简短的英文代码注释",
    )

    second = await manager.create_pending(
        user_id="user_001",
        candidate=second_candidate,
        existing_snapshot=make_snapshot(),
    )

    assert first["pending_id"] != second["pending_id"]

    loaded_first = await manager.get_pending(
        "user_001",
        first["pending_id"],
    )

    loaded_second = await manager.get_pending(
        "user_001",
        second["pending_id"],
    )

    assert loaded_first["candidate"]["content"] == (
        "喜欢简洁的英文代码注释"
    )

    assert loaded_second["candidate"]["content"] == (
        "喜欢简短的英文代码注释"
    )


@pytest.mark.anyio
async def test_pending_records_are_isolated_by_user(manager):
    """不同用户不能读取彼此的待确认记录。"""

    created = await manager.create_pending(
        user_id="user_001",
        candidate=make_candidate(),
        existing_snapshot=make_snapshot(),
    )

    result = await manager.get_pending(
        user_id="user_002",
        pending_id=created["pending_id"],
    )

    assert result is None

@pytest.mark.anyio
async def test_list_pending(manager):
    """测试待确认记录列表和用户隔离。"""

    first = await manager.create_pending(
        "user_001",
        make_candidate(),
        make_snapshot(),
    )

    second = await manager.create_pending(
        "user_001",
        MemoryCandidate(
            key="coding_style",
            category="preference",
            content="喜欢简短的英文代码注释",
            evidence="简短的英文代码注释",
        ),
        make_snapshot(),
    )
    await manager.create_pending(
        "user_002",
        make_candidate(),
        make_snapshot(),
    )

    results = await manager.list_pending("user_001")

    assert len(results) == 2

    result_ids = {
        item["pending_id"] for item in results
    }

    assert result_ids == {
        first["pending_id"],
        second["pending_id"],
    }

    assert all(
        item["status"] == "pending"
        for item in results
    )


@pytest.mark.anyio
async def test_list_pending_pagination(manager):
    """测试待确认记录分页。"""

    created_ids = set()
    for index in range(4):
        candidate = MemoryCandidate(
            key="coding_style",
            category="preference",
            content=f"喜欢英文代码注释版本{index}",
            evidence=f"英文代码注释版本{index}",
        )

        record = await manager.create_pending(
            "user_001",
            candidate,
            make_snapshot(),
        )

        created_ids.add(record["pending_id"])
    page_1 = await manager.list_pending(
        "user_001",
        limit=2,
        offset=0,
    )

    page_2 = await manager.list_pending(
        "user_001",
        limit=2,
        offset=2,
    )

    ids_1 = {
        item["pending_id"] for item in page_1
    }

    ids_2 = {
        item["pending_id"] for item in page_2
    }

    assert len(page_1) == 2
    assert len(page_2) == 2

    assert ids_1.isdisjoint(ids_2)

    assert ids_1 | ids_2 == created_ids


@pytest.mark.anyio
async def test_list_pending_empty(manager):
    """没有待确认记录时返回空列表。"""

    results = await manager.list_pending("user_001")

    assert results == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_limit",
    [0, -1, True, 1.5, "1", None],
)
async def test_list_pending_rejects_invalid_limit(
    manager,
    invalid_limit,
):
    """非法 limit 必须被拒绝。"""

    with pytest.raises(ValueError):
        await manager.list_pending(
            "user_001",
            limit=invalid_limit,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_offset",
    [-1, True, 1.5, "1", None],
)
async def test_list_pending_rejects_invalid_offset(
    manager,
    invalid_offset,
):
    """非法 offset 必须被拒绝。"""

    with pytest.raises(ValueError):
        await manager.list_pending(
            "user_001",
            offset=invalid_offset,
        )

@pytest.mark.anyio
async def test_delete_pending(manager):
    """删除待确认记录后，不能再次读取。"""

    created = await manager.create_pending(
        "user_001",
        make_candidate(),
        make_snapshot(),
    )

    await manager.delete_pending(
        "user_001",
        created["pending_id"],
    )

    result = await manager.get_pending(
        "user_001",
        created["pending_id"],
    )

    assert result is None

    remaining = await manager.list_pending("user_001")

    assert remaining == []


@pytest.mark.anyio
async def test_delete_missing_pending_is_idempotent(manager):
    """重复删除不存在的记录不应该报错。"""

    pending_id = str(uuid4())

    await manager.delete_pending(
        "user_001",
        pending_id,
    )

    await manager.delete_pending(
        "user_001",
        pending_id,
    )

    assert await manager.get_pending(
        "user_001",
        pending_id,
    ) is None


@pytest.mark.anyio
async def test_delete_pending_respects_user_isolation(manager):
    """一个用户不能删除另一个用户的待确认记录。"""

    created = await manager.create_pending(
        "user_001",
        make_candidate(),
        make_snapshot(),
    )

    await manager.delete_pending(
        "user_002",
        created["pending_id"],
    )

    result = await manager.get_pending(
        "user_001",
        created["pending_id"],
    )

    assert result == created


@pytest.mark.anyio
async def test_delete_pending_rejects_invalid_id(manager):
    """非法 pending_id 必须被拒绝。"""

    with pytest.raises(ValueError, match="pending_id"):
        await manager.delete_pending(
            "user_001",
            "invalid-pending-id",
        )

@pytest.mark.anyio
async def test_create_pending_rejects_invalid_candidate(manager):
    """候选记忆不是 MemoryCandidate 时应拒绝创建。"""

    with pytest.raises(TypeError, match="candidate"):
        await manager.create_pending(
            user_id="user_001",
            candidate=object(),
            existing_snapshot=make_snapshot(),
        )


@pytest.mark.anyio
async def test_create_pending_rejects_non_dict_snapshot(manager):
    """原始记忆快照不是字典时应拒绝创建。"""

    with pytest.raises(TypeError, match="existing_snapshot"):
        await manager.create_pending(
            user_id="user_001",
            candidate=make_candidate(),
            existing_snapshot=None,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field",
    ["content", "category", "created_at", "updated_at"],
)
async def test_create_pending_rejects_missing_snapshot_field(
    manager,
    field,
):
    """原始记忆快照缺少必要字段时应拒绝创建。"""

    snapshot = make_snapshot()
    del snapshot[field]

    with pytest.raises(
        ValueError,
        match=rf"existing_snapshot\.{field}",
    ):
        await manager.create_pending(
            user_id="user_001",
            candidate=make_candidate(),
            existing_snapshot=snapshot,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field",
    ["content", "category", "created_at", "updated_at"],
)
async def test_create_pending_rejects_empty_snapshot_field(
    manager,
    field,
):
    """原始记忆快照字段为空时应拒绝创建。"""

    snapshot = make_snapshot()
    snapshot[field] = "   "

    with pytest.raises(
        ValueError,
        match=rf"existing_snapshot\.{field}",
    ):
        await manager.create_pending(
            user_id="user_001",
            candidate=make_candidate(),
            existing_snapshot=snapshot,
        )


@pytest.mark.anyio
async def test_get_pending_rejects_invalid_uuid(manager):
    """读取待确认记录时应拒绝非法 UUID。"""

    with pytest.raises(ValueError, match="pending_id"):
        await manager.get_pending(
            user_id="user_001",
            pending_id="invalid-pending-id",
        )


@pytest.mark.anyio
async def test_create_pending_does_not_modify_original_snapshot(
    manager,
):
    """创建待确认记录不能修改调用方传入的原始快照。"""

    snapshot = make_snapshot()
    original_content = snapshot["content"]

    await manager.create_pending(
        user_id="user_001",
        candidate=make_candidate(),
        existing_snapshot=snapshot,
    )

    assert snapshot["content"] == original_content

@pytest.mark.anyio
async def test_create_pending_reuses_identical_record(manager):
    """相同记忆目标和快照应复用已有 Pending。"""

    candidate = make_candidate()
    snapshot = make_snapshot()

    first = await manager.create_pending(
        user_id="user_001",
        candidate=candidate,
        existing_snapshot=snapshot,
    )

    # evidence 改变，但 key/category/content 不变。
    second_candidate = candidate.model_copy(
        update={"evidence": "英文代码注释"}
    )

    second = await manager.create_pending(
        user_id="user_001",
        candidate=second_candidate,
        existing_snapshot=snapshot,
    )

    assert second["pending_id"] == first["pending_id"]
    assert second["created_at"] == first["created_at"]

    records = await manager.list_pending("user_001")

    assert len(records) == 1
    assert records[0]["pending_id"] == first["pending_id"]


@pytest.mark.anyio
async def test_create_pending_after_delete_uses_new_id(manager):
    """已删除的 Pending 不得复用旧 UUID。"""

    first = await manager.create_pending(
        user_id="user_001",
        candidate=make_candidate(),
        existing_snapshot=make_snapshot(),
    )

    await manager.delete_pending(
        user_id="user_001",
        pending_id=first["pending_id"],
    )

    second = await manager.create_pending(
        user_id="user_001",
        candidate=make_candidate(),
        existing_snapshot=make_snapshot(),
    )

    assert first["pending_id"] != second["pending_id"]

    records = await manager.list_pending("user_001")

    assert len(records) == 1
    assert records[0]["pending_id"] == second["pending_id"]
