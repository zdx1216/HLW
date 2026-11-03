
"""HLW PostgreSQL Store 连接与生命周期管理。

Store 用于保存跨会话的长期记忆。
本模块负责创建 Store、初始化数据库表和管理连接生命周期。
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.store.postgres.aio import AsyncPostgresStore

from hlw.memory.checkpoints_postgres import (
    build_postgres_conninfo,
)


@asynccontextmanager
async def open_postgres_store(
) -> AsyncGenerator[AsyncPostgresStore, None]:
    """创建、初始化并正确关闭 PostgreSQL Store。

    使用示例：

        async with open_postgres_store() as store:
            await store.aput(
                namespace=("users", "user_001", "memories"),
                key="learning_style",
                value={
                    "content": "喜欢详细的中文代码注释"
                },
            )

    注意：
        Store 必须在 async with 作用域内使用。
        退出作用域后，数据库连接自动关闭。
    """

    # 1. 复用 Checkpointer 已有的数据库连接配置。
    # 不重复读取 .env，也不重复编写数据库连接逻辑。
    conninfo = build_postgres_conninfo()

    # 2. 创建 PostgreSQL Store。
    # from_conn_string 返回异步上下文管理器。
    async with AsyncPostgresStore.from_conn_string(
        conninfo
    ) as store:

        # 3. 初始化 Store 所需的数据表。
        # 首次运行时创建表，后续运行时检查已有结构。
        await store.setup()

        # 4. 将 Store 交给调用方使用。
        # 调用方退出 async with 后，自动关闭连接。
        yield store
