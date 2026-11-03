
"""HLW MemoryExtractor 单元测试。

使用 FakeStructuredModel 模拟 LLM，
不需要启动 Ollama，也不需要连接 PostgreSQL。
"""

import asyncio
from typing import Any

import pytest

from hlw.memory.extractor import MemoryExtractor


class FakeStructuredModel:
    """模拟结构化 LLM 的返回结果。"""

    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.calls = 0
        self.messages = None

    async def ainvoke(self, messages):
        """模拟一次异步模型调用。"""

        self.calls += 1
        self.messages = messages

        return self.result


def test_extract_valid_candidate() -> None:
    """测试：正常提取长期偏好。"""

    message = "以后请给我的 Python 代码添加详细的中文注释。"

    fake_model = FakeStructuredModel(
        {
            "candidates": [
                {
                    "key": "python_comment_style",
                    "category": "preference",
                    "content": "喜欢详细的中文代码注释",
                    "evidence": "添加详细的中文注释",
                }
            ]
        }
    )

    extractor = MemoryExtractor(
        structured_model=fake_model
    )

    result = asyncio.run(
        extractor.extract(message)
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].key == "python_comment_style"
    assert result.candidates[0].category == "preference"
    assert fake_model.calls == 1


def test_reject_invalid_evidence() -> None:
    """测试：证据不在用户原文中，应当丢弃。"""

    message = "以后请给我的 Python 代码添加详细注释。"

    fake_model = FakeStructuredModel(
        {
            "candidates": [
                {
                    "key": "python_comment_style",
                    "category": "preference",
                    "content": "喜欢中文代码注释",
                    "evidence": "我非常喜欢中文注释",
                }
            ]
        }
    )

    extractor = MemoryExtractor(
        structured_model=fake_model
    )

    result = asyncio.run(
        extractor.extract(message)
    )

    assert result.candidates == []


def test_remove_duplicate_keys() -> None:
    """测试：相同 key 只保留第一条。"""

    message = "以后请给我的 Python 代码添加详细的中文注释。"

    fake_model = FakeStructuredModel(
        {
            "candidates": [
                {
                    "key": "python_comment_style",
                    "category": "preference",
                    "content": "喜欢详细的中文注释",
                    "evidence": "详细的中文注释",
                },
                {
                    "key": "python_comment_style",
                    "category": "preference",
                    "content": "喜欢 Python 代码",
                    "evidence": "Python 代码",
                },
            ]
        }
    )

    extractor = MemoryExtractor(
        structured_model=fake_model
    )

    result = asyncio.run(
        extractor.extract(message)
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].content == "喜欢详细的中文注释"


def test_empty_extraction() -> None:
    """测试：模型没有发现长期记忆。"""

    message = "今天下午我要去超市买东西。"

    fake_model = FakeStructuredModel(
        {
            "candidates": []
        }
    )

    extractor = MemoryExtractor(
        structured_model=fake_model
    )

    result = asyncio.run(
        extractor.extract(message)
    )

    assert result.candidates == []


def test_reject_empty_user_message() -> None:
    """测试：空消息不允许调用模型。"""

    fake_model = FakeStructuredModel(
        {
            "candidates": []
        }
    )

    extractor = MemoryExtractor(
        structured_model=fake_model
    )

    with pytest.raises(ValueError):
        asyncio.run(
            extractor.extract("   ")
        )

    assert fake_model.calls == 0
