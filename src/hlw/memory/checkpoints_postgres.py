"""HLW PostgreSQL Checkpointer 连接与生命周期管理。"""
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import make_conninfo

from hlw.core.config import PROJECT_ROOT


def build_postgres_conninfo() -> str:
    """根据 HLW 环境配置构建 PostgreSQL 连接信息。"""

    # 加载项目根目录下的 .env
    load_dotenv(PROJECT_ROOT / ".env")

    password = os.getenv("HLW_DB_PASSWORD")

    if not password:
        raise ValueError(
            "缺少 HLW_DB_PASSWORD，请检查项目根目录的 .env"
        )

    return make_conninfo(
        host=os.getenv("HLW_DB_HOST", "127.0.0.1"),
        port=os.getenv("HLW_DB_PORT", "5432"),
        dbname=os.getenv("HLW_DB_NAME", "hlw"),
        user=os.getenv("HLW_DB_USER", "hlw"),
        password=password,
    )


@asynccontextmanager
async def open_postgres_checkpointer(
) -> AsyncGenerator[AsyncPostgresSaver, None]:
    """创建、初始化并正确关闭 PostgreSQL Checkpointer。

    使用示例：
        async with open_postgres_checkpointer() as saver:
            graph = builder.compile(checkpointer=saver)
            await graph.ainvoke(...)

    注意：
        Checkpointer 必须在 async with 作用域内使用。
        离开作用域后，数据库连接会关闭。
    """

    conninfo = build_postgres_conninfo()

    async with AsyncPostgresSaver.from_conn_string(
        conninfo
    ) as saver:
        # 初始化 Checkpointer 所需的数据库表。
        # 当前实验阶段执行 setup()；正式部署时统一管理初始化。
        await saver.setup()

        yield saver
