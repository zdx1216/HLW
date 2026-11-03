
"""HLW MemoryValidator 单元测试。"""

import pytest

from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryExtractionResult,
)
from hlw.memory.validator import MemoryValidator


def make_candidate(
    key: str,
    content: str,
    evidence: str,
) -> MemoryCandidate:
    """构造测试用候选记忆。"""

    return MemoryCandidate(
        key=key,
        category="preference",
        content=content,
        evidence=evidence,
    )


def test_accept_valid_memory() -> None:
    """有效记忆应该通过校验。"""

    message = "以后请给 Python 代码添加中文注释。"

    extraction = MemoryExtractionResult(
        candidates=[
            make_candidate(
                "python_style",
                "喜欢中文代码注释",
                "添加中文注释",
            )
        ]
    )

    result = MemoryValidator().validate(
        extraction,
        message,
    )

    assert len(result.accepted) == 1
    assert len(result.rejected) == 0


def test_reject_missing_evidence() -> None:
    """证据不存在于原文，应拒绝。"""

    extraction = MemoryExtractionResult(
        candidates=[
            make_candidate(
                "python_style",
                "喜欢中文代码注释",
                "我喜欢英文注释",
            )
        ]
    )

    result = MemoryValidator().validate(
        extraction,
        "以后请给 Python 代码添加中文注释。",
    )

    assert len(result.accepted) == 0
    assert result.rejected[0].reason == "evidence_not_found"


def test_reject_possible_secret() -> None:
    """包含疑似 API Key 的记忆应拒绝。"""

    extraction = MemoryExtractionResult(
        candidates=[
            make_candidate(
                "api_config",
                "API_KEY=sk_test_123456",
                "API_KEY=sk_test_123456",
            )
        ]
    )

    result = MemoryValidator().validate(
        extraction,
        "API_KEY=sk_test_123456",
    )

    assert len(result.accepted) == 0
    assert result.rejected[0].reason == "possible_secret"


def test_reject_duplicate_key() -> None:
    """同一批次重复 key，只保留第一条。"""

    extraction = MemoryExtractionResult(
        candidates=[
            make_candidate(
                "python_style",
                "喜欢中文注释",
                "中文注释",
            ),
            make_candidate(
                "python_style",
                "喜欢详细注释",
                "详细注释",
            ),
        ]
    )

    result = MemoryValidator().validate(
        extraction,
        "我喜欢中文注释，也喜欢详细注释。",
    )

    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == "duplicate_key"


def test_reject_empty_message() -> None:
    """不允许使用空的原始消息进行校验。"""

    extraction = MemoryExtractionResult()

    with pytest.raises(ValueError):
        MemoryValidator().validate(
            extraction,
            " ",
        )
