
"""HLW PostgreSQL Store 连接与生命周期管理。

Store 用于保存跨会话长期记忆。

通过 LangGraph 内置向量索引能力，
实现记忆自动向量化和语义检索。
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres.aio import AsyncPostgresStore

from hlw.core.config import settings
from hlw.memory.checkpoints_postgres import (
    build_postgres_conninfo,
)


@asynccontextmanager
async def open_postgres_store(
) -> AsyncGenerator[AsyncPostgresStore, None]:
    """创建、初始化并正确关闭 PostgreSQL Store。

    支持：
    1. 长期记忆持久化。
    2. 自动生成文本 Embedding。
    3. pgvector 向量索引。
    4. 语义相似度检索。

    注意：
        Store 必须在 async with 作用域内使用。
    """
    # ========================================================
    # STEP 1：获取 PostgreSQL 连接信息
    # ========================================================
    conninfo = build_postgres_conninfo()
    # ========================================================
    # STEP 2：创建 LangChain 官方 Embedding 实例
    # ========================================================
    embeddings = OllamaEmbeddings(

        # 使用统一配置中的模型名称
        model=settings.embedding_model,

        # Ollama 原生服务地址
        base_url=settings.embedding_base_url,

        # 输出向量维度
        dimensions=settings.embedding_dimensions,
    )
    # ========================================================
    # STEP 3：定义 LangGraph 向量索引配置
    # ========================================================
    index_config = {
        # PostgreSQL pgvector 向量维度
        "dims": settings.embedding_dimensions,
        # LangChain Embedding 实例
        "embed": embeddings,
        # 仅对 content 字段进行向量化
        "fields": ["content"],
    }

    # ========================================================
    # STEP 4：创建支持向量检索的 PostgreSQL Store
    # ========================================================

    async with AsyncPostgresStore.from_conn_string(
        conninfo,
        index=index_config,
    ) as store:

        # 初始化数据库表和向量索引迁移
        await store.setup()

        # ====================================================
        # STEP 5：将 Store 交给调用方
        # ====================================================

        yield store
