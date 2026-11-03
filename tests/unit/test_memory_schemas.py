
"""HLW 记忆提取 Schema 测试。"""

import pytest
from pydantic import ValidationError

from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryExtractionResult,
)


def test_valid_candidate() -> None:
    """有效候选记忆应当通过验证。"""

    candidate = MemoryCandidate(
        key="learning_style",
        category="preference",
        content="喜欢中文代码注释",
        evidence="以后请添加中文代码注释",
    )

    assert candidate.key == "learning_style"


def test_empty_extraction() -> None:
    """没有值得记忆的信息时允许返回空列表。"""

    result = MemoryExtractionResult()

    assert result.candidates == []


def test_invalid_key() -> None:
    """记忆 key 不允许包含空格等非法字符。"""

    with pytest.raises(ValidationError):
        MemoryCandidate(
            key="learning style",
            category="preference",
            content="喜欢中文注释",
            evidence="以后请添加中文注释",
        )


def test_extra_field_rejected() -> None:
    """拒绝未声明的额外字段。"""

    with pytest.raises(ValidationError):
        MemoryCandidate.model_validate(
            {
                "key": "learning_style",
                "category": "preference",
                "content": "喜欢中文注释",
                "evidence": "以后请添加中文注释",
                "user_id": "admin",
            }
        )
