
"""HLW Persistent Runtime 生命周期测试。"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from hlw.agent import persistent_runtime as runtime


def test_runtime_injects_user_identity_and_releases_checkpointer(
    monkeypatch,
):
    """验证身份注入、Graph 构建及正常退出时的资源释放。"""

    events = []

    fake_saver = object()
    fake_graph = object()
    fake_service = object()

    # 模拟 PostgreSQL Checkpointer 上下文管理器
    @asynccontextmanager
    async def fake_open_checkpointer():
        events.append("checkpointer_opened")

        try:
            yield fake_saver

        finally:
            events.append("checkpointer_closed")

    # 模拟 LangGraph 构建过程
    def fake_build_graph(*, checkpointer):
        assert checkpointer is fake_saver

        events.append("graph_built")

        return fake_graph

    # 模拟 AgentService 创建过程
    def fake_agent_service(
        *,
        agent_graph,
        max_concurrent,
        user_id,
    ):
        assert agent_graph is fake_graph
        assert max_concurrent == 3
        assert user_id == runtime.settings.user_id

        events.append("service_created")

        return fake_service

    # 替换真实依赖
    monkeypatch.setattr(
        runtime,
        "open_postgres_checkpointer",
        fake_open_checkpointer,
    )

    monkeypatch.setattr(
        runtime,
        "build_persistent_graph",
        fake_build_graph,
    )

    monkeypatch.setattr(
        runtime,
        "AgentService",
        fake_agent_service,
    )

    async def scenario():

        async with runtime.open_persistent_agent_service(
            max_concurrent=3,
        ) as service:

            assert service is fake_service

            assert events == [
                "checkpointer_opened",
                "graph_built",
                "service_created",
            ]

        # 退出上下文后，Checkpointer 必须关闭
        assert events == [
            "checkpointer_opened",
            "graph_built",
            "service_created",
            "checkpointer_closed",
        ]

    asyncio.run(scenario())


def test_runtime_releases_checkpointer_after_exception(
    monkeypatch,
):
    """验证调用方发生异常时，Checkpointer 仍然关闭。"""

    events = []

    @asynccontextmanager
    async def fake_open_checkpointer():
        events.append("checkpointer_opened")

        try:
            yield object()

        finally:
            events.append("checkpointer_closed")

    monkeypatch.setattr(
        runtime,
        "open_postgres_checkpointer",
        fake_open_checkpointer,
    )

    monkeypatch.setattr(
        runtime,
        "build_persistent_graph",
        lambda *, checkpointer: object(),
    )

    monkeypatch.setattr(
        runtime,
        "AgentService",
        lambda **kwargs: object(),
    )

    async def scenario():

        with pytest.raises(
            RuntimeError,
            match="simulated failure",
        ):

            async with runtime.open_persistent_agent_service():

                events.append("caller_entered")

                raise RuntimeError("simulated failure")

        assert events == [
            "checkpointer_opened",
            "caller_entered",
            "checkpointer_closed",
        ]

    asyncio.run(scenario())


def test_runtime_releases_checkpointer_when_graph_build_fails(
    monkeypatch,
):
    """验证 Graph 构建失败时，Checkpointer 仍然关闭。"""

    events = []

    @asynccontextmanager
    async def fake_open_checkpointer():
        events.append("checkpointer_opened")

        try:
            yield object()

        finally:
            events.append("checkpointer_closed")

    def fake_build_graph(*, checkpointer):
        raise RuntimeError("graph build failed")

    monkeypatch.setattr(
        runtime,
        "open_postgres_checkpointer",
        fake_open_checkpointer,
    )

    monkeypatch.setattr(
        runtime,
        "build_persistent_graph",
        fake_build_graph,
    )

    async def scenario():

        with pytest.raises(
            RuntimeError,
            match="graph build failed",
        ):

            async with runtime.open_persistent_agent_service():
                pass

        assert events == [
            "checkpointer_opened",
            "checkpointer_closed",
        ]

    asyncio.run(scenario())
