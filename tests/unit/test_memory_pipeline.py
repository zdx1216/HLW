
"""HLW MemoryPipeline 单元测试。"""

import pytest
from langgraph.store.memory import InMemoryStore

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.manager import MemoryManager
from hlw.memory.pipeline import MemoryPipeline
from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryExtractionResult,
)
from hlw.memory.validator import MemoryValidator


@pytest.fixture
def anyio_backend() -> str:
    """指定使用 asyncio 后端。"""
    return "asyncio"


class FakeMemoryExtractor(MemoryExtractor):
    """模拟 Extractor，不调用真实 LLM。"""

    def __init__(
        self,
        result: MemoryExtractionResult,
    ) -> None:
        self._result = result

    async def extract(
        self,
        user_message: str,
    ) -> MemoryExtractionResult:
        return self._result


def make_candidate(
    content: str = "喜欢详细的中文代码注释",
    evidence: str = "详细的中文代码注释",
) -> MemoryCandidate:
    """构造测试用候选记忆。"""

    return MemoryCandidate(
        key="coding_style",
        category="preference",
        content=content,
        evidence=evidence,
    )


def build_pipeline(
    extraction: MemoryExtractionResult,
) -> tuple[MemoryPipeline, MemoryManager]:
    """构建使用内存数据库的测试环境。"""

    store = InMemoryStore()

    manager = MemoryManager(store)

    execution_locks = MemoryExecutionLocks()

    deduplicator = MemoryDeduplicator(
        manager,
        execution_locks,
    )

    pipeline = MemoryPipeline(
        extractor=FakeMemoryExtractor(extraction),
        validator=MemoryValidator(),
        deduplicator=deduplicator,
    )

    return pipeline, manager


@pytest.mark.anyio
async def test_pipeline_create_memory() -> None:
    """有效记忆应该经过完整流程并成功保存。"""

    candidate = make_candidate()

    pipeline, manager = build_pipeline(
        MemoryExtractionResult(
            candidates=[candidate]
        )
    )

    result = await pipeline.run(
        user_id="user_001",
        user_message="以后请提供详细的中文代码注释。",
    )

    assert len(result.extraction.candidates) == 1
    assert len(result.validation.accepted) == 1

    assert (
        result.deduplication_results[0].status
        == "created"
    )

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved is not None
    assert saved["content"] == candidate.content


@pytest.mark.anyio
async def test_pipeline_empty_extraction() -> None:
    """没有候选记忆时，不应该执行数据库写入。"""

    pipeline, _ = build_pipeline(
        MemoryExtractionResult()
    )

    result = await pipeline.run(
        user_id="user_001",
        user_message="你好。",
    )

    assert result.extraction.candidates == []
    assert result.validation.accepted == []
    assert result.deduplication_results == []


@pytest.mark.anyio
async def test_pipeline_reject_invalid_evidence() -> None:
    """证据不在原文中，应该被 Validator 拒绝。"""

    candidate = make_candidate(
        evidence="我喜欢英文注释"
    )

    pipeline, _ = build_pipeline(
        MemoryExtractionResult(
            candidates=[candidate]
        )
    )

    result = await pipeline.run(
        user_id="user_001",
        user_message="以后请提供中文代码注释。",
    )

    assert len(result.validation.accepted) == 0
    assert len(result.validation.rejected) == 1

    assert (
        result.validation.rejected[0].reason
        == "evidence_not_found"
    )

    assert result.deduplication_results == []


@pytest.mark.anyio
async def test_pipeline_unchanged_memory() -> None:
    """数据库已经存在相同记忆时，不重复保存。"""

    candidate = make_candidate()

    pipeline, manager = build_pipeline(
        MemoryExtractionResult(
            candidates=[candidate]
        )
    )

    await manager.save_memory(
        user_id="user_001",
        key="coding_style",
        content=candidate.content,
        category=candidate.category,
    )

    result = await pipeline.run(
        user_id="user_001",
        user_message="以后请提供详细的中文代码注释。",
    )

    assert (
        result.deduplication_results[0].status
        == "unchanged"
    )


@pytest.mark.anyio
async def test_pipeline_memory_conflict() -> None:
    """新旧记忆不一致时，不应该擅自覆盖。"""

    candidate = make_candidate(
        content="喜欢详细且规范的中文代码注释"
    )

    pipeline, manager = build_pipeline(
        MemoryExtractionResult(
            candidates=[candidate]
        )
    )

    await manager.save_memory(
        user_id="user_001",
        key="coding_style",
        content="喜欢中文代码注释",
        category="preference",
    )

    result = await pipeline.run(
        user_id="user_001",
        user_message="我喜欢详细的中文代码注释。",
    )

    assert (
        result.deduplication_results[0].status
        == "conflict"
    )

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved is not None
    assert saved["content"] == "喜欢中文代码注释"

@pytest.mark.anyio
async def test_pipeline_confirm_update() -> None:
    """用户授权后，Pipeline 应该成功更新已有记忆。"""

    candidate = make_candidate(
        content="喜欢简洁的英文代码注释",
        evidence="简洁的英文代码注释",
    )

    pipeline, manager = build_pipeline(
        MemoryExtractionResult(
            candidates=[candidate]
        )
    )

    # 先保存旧记忆。
    await manager.save_memory(
        user_id="user_001",
        key="coding_style",
        content="喜欢详细的中文代码注释",
        category="preference",
    )

    # 用户确认更新。
    result = await pipeline.confirm_update(
        user_id="user_001",
        candidate=candidate,
    )

    assert result.status == "updated"

    # 验证数据库中的内容确实改变。
    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved is not None
    assert saved["content"] == "喜欢简洁的英文代码注释"


@pytest.mark.anyio
async def test_pipeline_confirm_update_missing_memory() -> None:
    """旧记忆不存在时，Pipeline 不应该重新创建。"""

    candidate = make_candidate(
        content="喜欢简洁的英文代码注释",
        evidence="简洁的英文代码注释",
    )

    pipeline, manager = build_pipeline(
        MemoryExtractionResult(
            candidates=[candidate]
        )
    )

    result = await pipeline.confirm_update(
        user_id="user_001",
        candidate=candidate,
    )

    assert result.status == "conflict"

    saved = await manager.get_memory(
        "user_001",
        "coding_style",
    )

    assert saved is None
