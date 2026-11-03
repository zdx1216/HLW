
"""HLW MemoryManager 单元测试。

使用 InMemoryStore 替代 PostgreSQL，
测试 MemoryManager 的核心记忆管理功能。
"""

import pytest
from langgraph.store.memory import InMemoryStore

from hlw.memory.manager import MemoryManager


def create_manager() -> MemoryManager:
    """创建使用内存 Store 的 MemoryManager。

    每次调用都会创建一个新的内存数据库，
    避免不同测试之间相互影响。
    """

    store = InMemoryStore()

    return MemoryManager(store)


@pytest.mark.anyio
async def test_save_and_get_memory():
    """测试记忆保存和读取。"""

    manager = create_manager()

    saved = await manager.save_memory(
        user_id="user_001",
        key="learning_style",
        content="喜欢详细的中文代码注释",
        category="preference",
    )

    assert saved["content"] == "喜欢详细的中文代码注释"
    assert saved["category"] == "preference"
    assert "created_at" in saved
    assert "updated_at" in saved

    result = await manager.get_memory(
        user_id="user_001",
        key="learning_style",
    )

    assert result == saved


@pytest.mark.anyio
async def test_save_existing_memory():
    """测试相同 key 的记忆覆盖更新。"""

    manager = create_manager()

    first = await manager.save_memory(
        user_id="user_001",
        key="learning_style",
        content="喜欢中文注释",
    )

    second = await manager.save_memory(
        user_id="user_001",
        key="learning_style",
        content="喜欢详细的中文注释",
    )

    # 创建时间应该保留
    assert first["created_at"] == second["created_at"]

    # 内容应该更新
    assert second["content"] == "喜欢详细的中文注释"


@pytest.mark.anyio
async def test_user_memory_isolation():
    """测试不同用户之间的记忆隔离。"""

    manager = create_manager()

    await manager.save_memory(
        user_id="user_001",
        key="learning_style",
        content="喜欢中文注释",
    )

    await manager.save_memory(
        user_id="user_002",
        key="learning_style",
        content="喜欢英文注释",
    )

    user_001 = await manager.get_memory(
        "user_001",
        "learning_style",
    )

    user_002 = await manager.get_memory(
        "user_002",
        "learning_style",
    )

    assert user_001["content"] == "喜欢中文注释"
    assert user_002["content"] == "喜欢英文注释"


@pytest.mark.anyio
async def test_list_memories():
    """测试记忆列表和类别筛选。"""

    manager = create_manager()

    await manager.save_memory(
        "user_001",
        "learning_style",
        "喜欢中文注释",
        category="preference",
    )

    await manager.save_memory(
        "user_001",
        "project",
        "正在开发 HLW",
        category="project",
    )

    all_memories = await manager.list_memories(
        "user_001"
    )

    assert len(all_memories) == 2

    preferences = await manager.list_memories(
        "user_001",
        category="preference",
    )

    assert len(preferences) == 1
    assert preferences[0]["key"] == "learning_style"


@pytest.mark.anyio
async def test_update_memory():
    """测试已有记忆的局部更新。"""

    manager = create_manager()

    original = await manager.save_memory(
        "user_001",
        "learning_style",
        "喜欢中文注释",
        category="preference",
    )

    updated = await manager.update_memory(
        "user_001",
        "learning_style",
        content="喜欢详细的中文注释",
    )

    assert updated is not None

    assert updated["content"] == "喜欢详细的中文注释"

    # 未指定修改的字段应该保持不变
    assert updated["category"] == "preference"

    # 创建时间不应该改变
    assert updated["created_at"] == original["created_at"]


@pytest.mark.anyio
async def test_delete_memory():
    """测试记忆删除。"""

    manager = create_manager()

    await manager.save_memory(
        "user_001",
        "learning_style",
        "喜欢中文注释",
    )

    deleted = await manager.delete_memory(
        "user_001",
        "learning_style",
    )

    assert deleted is True

    # 删除后应该无法读取
    result = await manager.get_memory(
        "user_001",
        "learning_style",
    )

    assert result is None

    # 再次删除应该返回 False
    deleted_again = await manager.delete_memory(
        "user_001",
        "learning_style",
    )

    assert deleted_again is False


@pytest.mark.anyio
async def test_invalid_memory_input():
    """测试非法输入是否被正确拒绝。"""

    manager = create_manager()

    with pytest.raises(ValueError):
        await manager.save_memory(
            user_id="",
            key="test",
            content="测试",
        )

    with pytest.raises(ValueError):
        await manager.save_memory(
            user_id="user_001",
            key="",
            content="测试",
        )

    with pytest.raises(ValueError):
        await manager.save_memory(
            user_id="user_001",
            key="test",
            content="",
        )

    with pytest.raises(ValueError):
        await manager.list_memories(
            "user_001",
            limit=0,
        )
