
"""HLW 常驻 Agent 服务的生命周期管理。"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from hlw.agent.graph import build_persistent_graph
from hlw.agent.service import AgentService
from hlw.core.config import settings
from hlw.memory.checkpoints_postgres import (
    open_postgres_checkpointer,
)


@asynccontextmanager
async def open_persistent_agent_service(
    max_concurrent: int = 2,
) -> AsyncGenerator[AgentService, None]:
    """打开一个常驻的 Agent 服务运行时。

    单用户模式下，从全局配置读取固定 user_id。

    使用示例：
        async with open_persistent_agent_service() as service:
            await service.run("你好", thread_id="demo")

    注意：
        PostgreSQL 连接在服务使用期间保持打开。
        退出 async with 后自动释放数据库连接。
    """

    # 1. 打开 PostgreSQL Checkpointer
    async with open_postgres_checkpointer() as saver:

        # 2. 使用 Checkpointer 构建持久化 Agent Graph
        persistent_graph = build_persistent_graph(
            checkpointer=saver
        )

        # 3. 创建 AgentService
        # 显式传入服务端配置的用户身份
        service = AgentService(
            agent_graph=persistent_graph,
            max_concurrent=max_concurrent,
            user_id=settings.user_id,
        )

        # 4. 将服务交给调用方
        yield service
