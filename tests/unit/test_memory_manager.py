
"""HLW MemoryManager 单元测试。

使用 InMemoryStore 替代 PostgreSQL，
测试 MemoryManager 的核心记忆管理功能。

测试范围：
1. 记忆保存与读取
2. 重复记忆覆盖
3. 用户记忆隔离
4. 记忆列表和分类筛选
5. 记忆分页检索
6. 记忆更新与删除
7. 用户身份规范化
8. 非法输入校验
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

    # 创建时间应该保留。
    assert first["created_at"] == second["created_at"]

    # 内容应该更新。
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

    # 未指定修改的字段应该保持不变。
    assert updated["category"] == "preference"

    # 创建时间不应该改变。
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

    # 删除后应该无法读取。
    result = await manager.get_memory(
        "user_001",
        "learning_style",
    )

    assert result is None

    # 再次删除应该返回 False。
    deleted_again = await manager.delete_memory(
        "user_001",
        "learning_style",
    )

    assert deleted_again is False


@pytest.mark.anyio
async def test_invalid_memory_input():
    """测试非法记忆输入是否被正确拒绝。"""

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


# =====================================================
# 用户身份规范化测试
# =====================================================


@pytest.mark.anyio
async def test_user_id_is_normalized_before_namespace():
    """测试用户 ID 在构建命名空间前进行规范化。"""

    manager = create_manager()

    await manager.save_memory(
        user_id=" user_001 ",
        key="learning_style",
        content="喜欢详细的中文注释",
    )

    # 规范化前后的用户 ID 应访问相同的记忆空间。
    result = await manager.get_memory(
        user_id="user_001",
        key="learning_style",
    )

    assert result is not None
    assert result["content"] == "喜欢详细的中文注释"


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
    invalid_user_id,
):
    """测试非法用户身份必须被拒绝。"""

    manager = create_manager()

    with pytest.raises(ValueError):
        await manager.save_memory(
            user_id=invalid_user_id,
            key="test",
            content="测试内容",
        )


# =====================================================
# 分页检索测试
# =====================================================


@pytest.mark.anyio
async def test_list_memories_pagination():
    """测试记忆列表分页及类别筛选。"""

    manager = create_manager()

    # 创建4条 preference 记忆。
    keys = [
        "learning_style",
        "coding_style",
        "research_style",
        "writing_style",
    ]

    for key in keys:
        await manager.save_memory(
            user_id="user_001",
            key=key,
            content=f"记忆内容：{key}",
            category="preference",
        )

    # 再创建一条其他类别的记忆。
    await manager.save_memory(
        user_id="user_001",
        key="project",
        content="正在开发 HLW",
        category="project",
    )

    # 获取 preference 第一页。
    page_1 = await manager.list_memories(
        user_id="user_001",
        category="preference",
        limit=2,
        offset=0,
    )

    # 获取 preference 第二页。
    page_2 = await manager.list_memories(
        user_id="user_001",
        category="preference",
        limit=2,
        offset=2,
    )

    keys_1 = {
        item["key"]
        for item in page_1
    }

    keys_2 = {
        item["key"]
        for item in page_2
    }

    assert len(page_1) == 2
    assert len(page_2) == 2

    # 两页不能出现重复记录。
    assert keys_1.isdisjoint(keys_2)

    # 两页合起来必须覆盖全部 preference 记忆。
    assert keys_1 | keys_2 == set(keys)

    # 第三页应该没有剩余记录。
    page_3 = await manager.list_memories(
        user_id="user_001",
        category="preference",
        limit=2,
        offset=4,
    )

    assert page_3 == []

    # project 类别不能出现在 preference 结果中。
    assert all(
        item["category"] == "preference"
        for item in page_1 + page_2
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_offset",
    [
        -1,
        True,
        1.5,
        "1",
        None,
    ],
)
async def test_list_memories_rejects_invalid_offset(
    invalid_offset,
):
    """非法 offset 必须被拒绝。"""

    manager = create_manager()

    with pytest.raises(ValueError):
        await manager.list_memories(
            user_id="user_001",
            offset=invalid_offset,
        )
