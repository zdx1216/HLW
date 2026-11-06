
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
    fake_store = object()
    fake_graph = object()
    fake_service = object()

    # ============================================================
    # 1. 模拟 PostgreSQL Checkpointer
    # ============================================================

    @asynccontextmanager
    async def fake_open_checkpointer():
        events.append("checkpointer_opened")

        try:
            yield fake_saver
        finally:
            events.append("checkpointer_closed")

    # ============================================================
    # 2. 模拟 PostgreSQL Store
    # ============================================================

    @asynccontextmanager
    async def fake_open_store():
        events.append("store_opened")

        try:
            yield fake_store
        finally:
            events.append("store_closed")

    # ============================================================
    # 3. 模拟 LangGraph 构建过程
    # ============================================================

    def fake_build_graph(*, checkpointer, store):
        assert checkpointer is fake_saver
        assert store is fake_store

        events.append("graph_built")

        return fake_graph

    # ============================================================
    # 4. 模拟 AgentService 创建过程
    # ============================================================

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

    # ============================================================
    # 5. 替换真实依赖
    # ============================================================

    monkeypatch.setattr(
        runtime,
        "open_postgres_checkpointer",
        fake_open_checkpointer,
    )

    monkeypatch.setattr(
        runtime,
        "open_postgres_store",
        fake_open_store,
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

    # ============================================================
    # 6. 执行测试
    # ============================================================

    async def scenario():

        async with runtime.open_persistent_agent_service(
            max_concurrent=3,
        ) as service:

            assert service is fake_service

            # 两个资源均已打开，Graph 和 Service 已创建
            assert events == [
                "checkpointer_opened",
                "store_opened",
                "graph_built",
                "service_created",
            ]

        # 退出上下文后，按逆序释放资源
        assert events == [
            "checkpointer_opened",
            "store_opened",
            "graph_built",
            "service_created",
            "store_closed",
            "checkpointer_closed",
        ]

    asyncio.run(scenario())


def test_runtime_releases_checkpointer_after_exception(
    monkeypatch,
):
    """验证调用方发生异常时，两个资源仍然正确关闭。"""

    events = []

    fake_saver = object()
    fake_store = object()

    # ============================================================
    # 1. 模拟 Checkpointer
    # ============================================================

    @asynccontextmanager
    async def fake_open_checkpointer():
        events.append("checkpointer_opened")

        try:
            yield fake_saver
        finally:
            events.append("checkpointer_closed")

    # ============================================================
    # 2. 模拟 Store
    # ============================================================

    @asynccontextmanager
    async def fake_open_store():
        events.append("store_opened")

        try:
            yield fake_store
        finally:
            events.append("store_closed")

    # ============================================================
    # 3. 模拟 Graph 构建
    # ============================================================

    def fake_build_graph(*, checkpointer, store):
        assert checkpointer is fake_saver
        assert store is fake_store

        events.append("graph_built")

        return object()

    # ============================================================
    # 4. 模拟 Service 创建
    # ============================================================

    def fake_agent_service(**kwargs):
        events.append("service_created")

        return object()

    # ============================================================
    # 5. 注入模拟依赖
    # ============================================================

    monkeypatch.setattr(
        runtime,
        "open_postgres_checkpointer",
        fake_open_checkpointer,
    )

    monkeypatch.setattr(
        runtime,
        "open_postgres_store",
        fake_open_store,
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

    # ============================================================
    # 6. 模拟调用方异常
    # ============================================================

    async def scenario():

        with pytest.raises(
            RuntimeError,
            match="simulated failure",
        ):

            async with runtime.open_persistent_agent_service():

                events.append("caller_entered")

                raise RuntimeError("simulated failure")

        # 即使调用方发生异常，两个资源也必须释放
        assert events == [
            "checkpointer_opened",
            "store_opened",
            "graph_built",
            "service_created",
            "caller_entered",
            "store_closed",
            "checkpointer_closed",
        ]

    asyncio.run(scenario())


def test_runtime_releases_checkpointer_when_graph_build_fails(
    monkeypatch,
):
    """验证 Graph 构建失败时，两个资源仍然正确关闭。"""

    events = []

    fake_saver = object()
    fake_store = object()

    # ============================================================
    # 1. 模拟 Checkpointer
    # ============================================================

    @asynccontextmanager
    async def fake_open_checkpointer():
        events.append("checkpointer_opened")

        try:
            yield fake_saver
        finally:
            events.append("checkpointer_closed")

    # ============================================================
    # 2. 模拟 Store
    # ============================================================

    @asynccontextmanager
    async def fake_open_store():
        events.append("store_opened")

        try:
            yield fake_store
        finally:
            events.append("store_closed")

    # ============================================================
    # 3. 模拟 Graph 构建失败
    # ============================================================

    def fake_build_graph(*, checkpointer, store):
        assert checkpointer is fake_saver
        assert store is fake_store

        events.append("graph_build_failed")

        raise RuntimeError("graph build failed")

    # ============================================================
    # 4. 注入模拟依赖
    # ============================================================

    monkeypatch.setattr(
        runtime,
        "open_postgres_checkpointer",
        fake_open_checkpointer,
    )

    monkeypatch.setattr(
        runtime,
        "open_postgres_store",
        fake_open_store,
    )

    monkeypatch.setattr(
        runtime,
        "build_persistent_graph",
        fake_build_graph,
    )

    # ============================================================
    # 5. 执行测试
    # ============================================================

    async def scenario():

        with pytest.raises(
            RuntimeError,
            match="graph build failed",
        ):

            async with runtime.open_persistent_agent_service():
                pass

        # Graph 构建失败后，两个资源必须释放
        assert events == [
            "checkpointer_opened",
            "store_opened",
            "graph_build_failed",
            "store_closed",
            "checkpointer_closed",
        ]

    asyncio.run(scenario())
