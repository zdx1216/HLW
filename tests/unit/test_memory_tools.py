
"""HLW 长期记忆工具测试。"""

import json
from types import SimpleNamespace

import pytest

from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

from hlw.memory.manager import MemoryManager
from hlw.tools.memory_tools import (
    _search_user_memories,
    search_user_memories,
)


class FakeEmbeddings(Embeddings):
    """模拟向量模型。"""

    def embed_query(self, text: str) -> list[float]:
        if "代码" in text or "注释" in text:
            return [1.0, 0.0, 0.0]

        if "项目" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self.embed_query(text)
            for text in texts
        ]


def create_store() -> InMemoryStore:
    """创建支持语义检索的内存 Store。"""

    return InMemoryStore(
        index={
            "dims": 3,
            "embed": FakeEmbeddings(),
            "fields": ["content"],
        }
    )


def create_runtime(store, user_id):
    """模拟 Agent 注入的运行时对象。"""

    return SimpleNamespace(
        config={
            "configurable": {
                "user_id": user_id,
            }
        },
        store=store,
    )


@pytest.mark.anyio
async def test_search_user_memories():
    """测试长期记忆语义检索和用户隔离。"""

    store = create_store()
    manager = MemoryManager(store)

    # 用户1的记忆
    await manager.save_memory(
        user_id="user_001",
        key="learning_style",
        content="喜欢详细的中文代码注释",
        category="preference",
    )

    # 用户2的记忆
    await manager.save_memory(
        user_id="user_002",
        key="learning_style",
        content="喜欢简洁的英文注释",
        category="preference",
    )

    runtime = create_runtime(
        store=store,
        user_id="user_001",
    )

    result = await _search_user_memories(
        query="我喜欢什么样的代码注释？",
        runtime=runtime,
    )

    data = json.loads(result)

    assert data["count"] == 1

    assert data["memories"][0]["key"] == "learning_style"

    assert data["memories"][0]["content"] == (
        "喜欢详细的中文代码注释"
    )


@pytest.mark.anyio
async def test_missing_user_id():
    """测试缺少用户身份时拒绝检索。"""

    runtime = SimpleNamespace(
        config={"configurable": {}},
        store=create_store(),
    )

    with pytest.raises(RuntimeError, match="user_id"):
        await _search_user_memories(
            query="我的偏好",
            runtime=runtime,
        )


@pytest.mark.anyio
async def test_invalid_user_id():
    """测试非法用户身份被拒绝。"""

    runtime = create_runtime(
        store=create_store(),
        user_id="../admin",
    )

    with pytest.raises(ValueError):
        await _search_user_memories(
            query="我的偏好",
            runtime=runtime,
        )


@pytest.mark.anyio
async def test_missing_store():
    """测试没有 Store 时拒绝检索。"""

    runtime = create_runtime(
        store=None,
        user_id="user_001",
    )

    with pytest.raises(RuntimeError, match="Store"):
        await _search_user_memories(
            query="我的偏好",
            runtime=runtime,
        )


def test_tool_definition():
    """验证工具注册及模型可见参数。"""

    # 工具名称正确
    assert search_user_memories.name == (
        "search_user_memories"
    )

    # 模型只能看到 query 参数
    model_fields = set(
        search_user_memories.tool_call_schema.model_fields
    )

    assert model_fields == {"query"}