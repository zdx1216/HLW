
"""HLW 常驻 Agent 服务与 PostgreSQL Store 生命周期管理。"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langgraph.store.base import BaseStore

from hlw.agent.graph import build_agent_graph
from hlw.agent.service import AgentService
from hlw.core.config import settings
from hlw.memory.checkpoints_postgres import (
    open_postgres_checkpointer,
)
from hlw.memory.store_postgres import open_postgres_store


@dataclass(frozen=True)
class PersistentAgentRuntime:
    """向 API 暴露当前有效的 AgentService 与 Store。"""

    service: AgentService
    store: BaseStore


@asynccontextmanager
async def open_persistent_agent_runtime(
    max_concurrent: int = 2,
) -> AsyncGenerator[PersistentAgentRuntime, None]:
    """统一创建、共享并释放持久化运行时资源。"""

    # STEP 1：打开 Checkpointer 和 Store。
    async with (
        open_postgres_checkpointer() as saver,
        open_postgres_store() as store,
    ):
        # STEP 2：创建共享持久化资源的 Agent Graph。
        agent_graph = build_agent_graph(
            checkpointer=saver,
            store=store,
        )

        # STEP 3：创建单用户 AgentService。
        service = AgentService(
            agent_graph=agent_graph,
            max_concurrent=max_concurrent,
            user_id=settings.user_id,
        )

        # STEP 4：向 API 提供仍在生命周期内的资源。
        yield PersistentAgentRuntime(
            service=service,
            store=store,
        )


@asynccontextmanager
async def open_persistent_agent_service(
    max_concurrent: int = 2,
) -> AsyncGenerator[AgentService, None]:
    """保留已有 AgentService 接口，兼容旧调用方。"""

    async with open_persistent_agent_runtime(
        max_concurrent=max_concurrent,
    ) as runtime:
        yield runtime.service
