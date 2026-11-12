
"""MemoryCaptureMiddleware 单元测试。

使用 InMemoryStore 和模拟提取器，
不连接 PostgreSQL，也不调用真实 LLM。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from hlw.agent.memory_capture_middleware import MemoryCaptureMiddleware
from hlw.core.config import settings
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate, MemoryExtractionResult


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeMemoryExtractor(MemoryExtractor):
    """返回固定候选记忆，不调用 LLM。"""

    def __init__(
        self,
        candidate: MemoryCandidate,
    ) -> None:
        self._candidate = candidate
        self.calls: list[str] = []

    async def extract(
        self,
        user_message: str,
    ) -> MemoryExtractionResult:
        self.calls.append(user_message)

        return MemoryExtractionResult(
            candidates=[self._candidate]
        )


class FailingMemoryExtractor(MemoryExtractor):
    """模拟记忆提取失败。"""

    async def extract(
        self,
        user_message: str,
    ) -> MemoryExtractionResult:
        raise RuntimeError("simulated extraction failure")


async def invoke_middleware(
    middleware: MemoryCaptureMiddleware,
    store: InMemoryStore | None,
    message: str,
    *,
    human: bool = True,
) -> None:
    """模拟一次 Agent 模型调用并验证 handler 正常执行。"""

    chat_message = (
        HumanMessage(content=message)
        if human
        else AIMessage(content=message)
    )

    request = SimpleNamespace(
        messages=[chat_message],
        runtime=SimpleNamespace(store=store),
    )

    response = object()
    handler_calls = []

    async def handler(current_request):
        handler_calls.append(current_request)
        return response

    actual = await middleware.awrap_model_call(
        request,
        handler,
    )

    assert actual is response
    assert handler_calls == [request]


@pytest.mark.anyio
async def test_middleware_creates_new_memory() -> None:
    """首次捕获有效偏好时，应创建正式记忆。"""

    store = InMemoryStore()

    candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢中文代码注释",
        evidence="中文代码注释",
    )

    extractor = FakeMemoryExtractor(candidate)

    middleware = MemoryCaptureMiddleware(
        extractor=extractor,
    )

    await invoke_middleware(
        middleware,
        store,
        "以后请提供中文代码注释。",
    )

    manager = MemoryManager(store)

    saved = await manager.get_memory(
        settings.user_id,
        "coding_style",
    )

    assert saved is not None
    assert saved["content"] == candidate.content

    pending_manager = PendingMemoryManager(store)

    assert await pending_manager.list_pending(
        settings.user_id
    ) == []

    assert extractor.calls == [
        "以后请提供中文代码注释。"
    ]


@pytest.mark.anyio
async def test_middleware_creates_pending_on_conflict() -> None:
    """冲突记忆应进入 Pending，不得直接覆盖正式记忆。"""

    store = InMemoryStore()
    manager = MemoryManager(store)

    await manager.save_memory(
        user_id=settings.user_id,
        key="coding_style",
        content="喜欢中文代码注释",
        category="preference",
    )

    original = await manager.get_memory(
        settings.user_id,
        "coding_style",
    )

    candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢英文代码注释",
        evidence="英文代码注释",
    )

    middleware = MemoryCaptureMiddleware(
        extractor=FakeMemoryExtractor(candidate),
    )

    await invoke_middleware(
        middleware,
        store,
        "以后请提供英文代码注释。",
    )

    # 原始正式记忆不能改变。
    saved = await manager.get_memory(
        settings.user_id,
        "coding_style",
    )
    assert saved == original

    # 应当存在一条待确认记录。
    pending_manager = PendingMemoryManager(store)
    pending_records = await pending_manager.list_pending(
        settings.user_id
    )

    assert len(pending_records) == 1

    pending = pending_records[0]

    assert pending["status"] == "pending"
    assert pending["candidate"] == candidate.model_dump(
        mode="json"
    )

    for field in (
        "content",
        "category",
        "created_at",
        "updated_at",
    ):
        assert (
            pending["existing_snapshot"][field]
            == original[field]
        )


@pytest.mark.anyio
async def test_middleware_skips_without_store() -> None:
    """没有 Runtime Store 时，仍应正常执行 handler。"""

    candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢中文代码注释",
        evidence="中文代码注释",
    )

    extractor = FakeMemoryExtractor(candidate)

    middleware = MemoryCaptureMiddleware(
        extractor=extractor,
    )

    await invoke_middleware(
        middleware,
        None,
        "以后请提供中文代码注释。",
    )

    assert extractor.calls == []


@pytest.mark.anyio
async def test_middleware_failure_does_not_block_chat() -> None:
    """记忆提取失败时，不能中断正常模型调用。"""

    middleware = MemoryCaptureMiddleware(
        extractor=FailingMemoryExtractor(),
    )

    await invoke_middleware(
        middleware,
        InMemoryStore(),
        "以后请提供中文代码注释。",
    )


@pytest.mark.anyio
async def test_middleware_skips_non_human_message() -> None:
    """最后一条不是用户消息时，不触发记忆捕获。"""

    candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢中文代码注释",
        evidence="中文代码注释",
    )

    extractor = FakeMemoryExtractor(candidate)

    middleware = MemoryCaptureMiddleware(
        extractor=extractor,
    )

    await invoke_middleware(
        middleware,
        InMemoryStore(),
        "好的，我会使用中文注释。",
        human=False,
    )

    assert extractor.calls == []

@pytest.mark.anyio
async def test_middleware_reuses_pending_on_repeated_conflict() -> None:
    """重复捕获相同冲突时不能生成多条 Pending。"""

    store = InMemoryStore()
    manager = MemoryManager(store)

    await manager.save_memory(
        user_id=settings.user_id,
        key="coding_style",
        content="喜欢中文代码注释",
        category="preference",
    )

    candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content="喜欢英文代码注释",
        evidence="英文代码注释",
    )

    middleware = MemoryCaptureMiddleware(
        extractor=FakeMemoryExtractor(candidate),
    )

    pending_manager = PendingMemoryManager(store)

    # 第一次捕获冲突。
    await invoke_middleware(
        middleware,
        store,
        "以后请提供英文代码注释。",
    )

    first_records = await pending_manager.list_pending(
        settings.user_id
    )

    assert len(first_records) == 1
    first_id = first_records[0]["pending_id"]

    # 第二次重复提交相同消息。
    await invoke_middleware(
        middleware,
        store,
        "以后请提供英文代码注释。",
    )

    second_records = await pending_manager.list_pending(
        settings.user_id
    )

    # 不得新增第二条相同 Pending。
    assert len(second_records) == 1
    assert second_records[0]["pending_id"] == first_id

    # 未确认前，正式记忆仍然保持原样。
    saved = await manager.get_memory(
        settings.user_id,
        "coding_style",
    )

    assert saved["content"] == "喜欢中文代码注释"
