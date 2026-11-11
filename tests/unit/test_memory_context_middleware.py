"""长期记忆自动加载 Middleware 单元测试。"""

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from hlw.agent.memory_context_middleware import (
    MemoryContextMiddleware,
)
from hlw.memory.manager import MemoryManager


@dataclass(frozen=True)
class FakeRequest:
    """模拟 LangChain ModelRequest。"""

    messages: list[Any]
    runtime: Any
    system_message: SystemMessage | None

    def override(self, **overrides: Any) -> "FakeRequest":
        return replace(self, **overrides)


def make_request(store: Any) -> FakeRequest:
    """构造测试请求。"""

    return FakeRequest(
        messages=[
            HumanMessage(
                content="请回忆我之前关于 HLW 长期记忆的设计"
            )
        ],
        runtime=SimpleNamespace(store=store),
        system_message=SystemMessage(
            content="你是 HLW 助手。"
        ),
    )


def run_middleware(request, handler):
    """运行异步 Middleware。"""

    middleware = MemoryContextMiddleware()

    return asyncio.run(
        middleware.awrap_model_call(
            request,
            handler,
        )
    )


def test_memory_is_loaded_into_system_message(monkeypatch):
    """有相关记忆时，应自动检索并注入模型请求。"""

    async def fake_search(
        self,
        *,
        user_id,
        query,
        limit,
    ):
        assert user_id
        assert "HLW" in query
        assert limit == 5

        return [
            {
                "key": "project_goal",
                "category": "project",
                "content": "用户希望实现长期记忆自动增删改查与加载。",
            }
        ]

    monkeypatch.setattr(
        MemoryManager,
        "search_memories",
        fake_search,
    )

    request = make_request(object())
    captured = []

    async def handler(received_request):
        captured.append(received_request)
        return "MODEL_RESPONSE"

    result = run_middleware(request, handler)

    assert result == "MODEL_RESPONSE"
    assert len(captured) == 1

    # 原始请求不能被修改。
    assert captured[0] is not request

    assert request.system_message.text == "你是 HLW 助手。"

    # 新请求必须保留原有 SystemMessage。
    assert "你是 HLW 助手。" in captured[0].system_message.text

    # 新请求必须包含检索到的记忆。
    assert "长期记忆使用规则" in captured[0].system_message.text
    assert "长期记忆自动增删改查与加载" in (
        captured[0].system_message.text
    )


def test_missing_store_skips_memory_loading(monkeypatch):
    """没有 Store 时应跳过记忆检索。"""

    async def forbidden_search(self, **kwargs):
        raise AssertionError("不应该执行记忆检索")

    monkeypatch.setattr(
        MemoryManager,
        "search_memories",
        forbidden_search,
    )

    request = make_request(None)
    captured = []

    async def handler(received_request):
        captured.append(received_request)
        return "MODEL_RESPONSE"

    result = run_middleware(request, handler)

    assert result == "MODEL_RESPONSE"
    assert captured == [request]


def test_empty_memories_do_not_modify_request(monkeypatch):
    """没有检索到记忆时不修改请求。"""

    async def fake_search(self, **kwargs):
        return []

    monkeypatch.setattr(
        MemoryManager,
        "search_memories",
        fake_search,
    )

    request = make_request(object())
    captured = []

    async def handler(received_request):
        captured.append(received_request)
        return "MODEL_RESPONSE"

    result = run_middleware(request, handler)

    assert result == "MODEL_RESPONSE"
    assert captured == [request]


def test_retrieval_failure_does_not_interrupt_agent(monkeypatch):
    """记忆检索异常时应降级，不影响正常对话。"""

    async def fake_search(self, **kwargs):
        raise RuntimeError("模拟数据库连接失败")

    monkeypatch.setattr(
        MemoryManager,
        "search_memories",
        fake_search,
    )

    request = make_request(object())
    captured = []

    async def handler(received_request):
        captured.append(received_request)
        return "MODEL_RESPONSE"

    result = run_middleware(request, handler)

    assert result == "MODEL_RESPONSE"
    assert captured == [request]


def test_build_search_query_extracts_user_messages():
    """检索问题应只提取用户消息。"""

    from hlw.agent.memory_context_middleware import (
        _build_search_query,
    )

    messages = [
        HumanMessage(content="第一条用户消息"),
        HumanMessage(content="第二条用户消息"),
    ]

    query = _build_search_query(messages)

    assert query == "第一条用户消息\n第二条用户消息"
