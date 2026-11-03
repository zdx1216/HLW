
"""HLW 常驻 Agent 服务的生命周期管理。"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from hlw.agent.graph import build_persistent_graph
from hlw.agent.service import AgentService
from hlw.memory.checkpoints_postgres import (
    open_postgres_checkpointer,
)


@asynccontextmanager
async def open_persistent_agent_service(
    max_concurrent: int = 2,
) -> AsyncGenerator[AgentService, None]:
    """打开一个常驻的 Agent 服务运行时。

    使用示例：
        async with open_persistent_agent_service() as service:
            await service.run("你好", thread_id="demo")

    注意：
        PostgreSQL 连接在作用域内保持打开，退出 ``async with`` 时自动关闭。
        ``max_concurrent`` 限制该服务同时处理的请求数。
    """

    # 打开 PostgreSQL Checkpointer（保存对话/图状态的持久化组件），
    # 它会随上下文退出而自动释放数据库连接。
    async with open_postgres_checkpointer() as saver:
        # 用该 Checkpointer 构建可持久化的 Agent 图。
        persistent_graph = build_persistent_graph(
            checkpointer=saver
        )

        # 把图包装成对外提供服务的 AgentService。
        service = AgentService(
            agent_graph=persistent_graph,
            max_concurrent=max_concurrent,
        )

        # 将服务交给调用方使用；使用结束后回到这里继续执行收尾逻辑。
        yield service
