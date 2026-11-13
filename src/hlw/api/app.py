
"""HLW HTTP API entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from hlw.agent.persistent_runtime import (
    open_persistent_agent_runtime,
)
from hlw.api.memory_routes import router as memory_router
from hlw.core.logging_config import get_logger, setup_logging
from hlw.memory.confirmation_service import MemoryConfirmationService
from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import memory_execution_locks
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager


class ChatRequest(BaseModel):
    """Incoming Agent request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)
    request_id: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    """Final Agent response."""

    thread_id: str
    request_id: str
    answer: str


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    """在应用生命周期内共享 AgentService 和 Store。"""

    setup_logging(runtime="api")
    logger = get_logger(__name__)

    logger.info("HLW API startup")

    # STEP 1：创建共享的持久化运行时。
    async with open_persistent_agent_runtime(
        max_concurrent=2,) as runtime:
        # STEP 2：创建共享的长期记忆管理组件。
        manager = MemoryManager(runtime.store)

        pending_manager = PendingMemoryManager(
            runtime.store
        )

        # STEP 3：与 MemoryCaptureMiddleware 共享进程内锁。
        execution_locks = memory_execution_locks

        deduplicator = MemoryDeduplicator(
            manager=manager,
            execution_locks=execution_locks,
        )

        # STEP 4：组装记忆确认服务。
        confirmation_service = MemoryConfirmationService(
            pending_manager=pending_manager,
            deduplicator=deduplicator,
            execution_locks=execution_locks,
        )

        # STEP 5：将组件注册到 FastAPI 生命周期。
        app.state.agent_service = runtime.service
        app.state.pending_memory_manager = pending_manager
        app.state.memory_confirmation_service = (
            confirmation_service
        )

        try:
            yield

        finally:
            # STEP 6：退出时清理组件引用。
            app.state.agent_service = None
            app.state.pending_memory_manager = None
            app.state.memory_confirmation_service = None



app = FastAPI(
    title="Hermes Local Workbench API",
    version="0.1.0",
    lifespan=lifespan,
)

# 注册长期记忆管理路由。
app.include_router(memory_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic application liveness check."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Execute a request using the shared AgentService."""

    thread_id = body.thread_id or uuid4().hex
    request_id = body.request_id or uuid4().hex

    service = request.app.state.agent_service

    result = await service.run(
        body.message,
        thread_id=thread_id,
        request_id=request_id,
    )

    answer = ""

    for message in reversed(result["messages"]):
        if getattr(message, "type", None) != "ai":
            continue

        content = getattr(message, "content", "")

        if isinstance(content, str) and content.strip():
            answer = content
            break

    if not answer:
        raise RuntimeError(
            "Agent returned no final text answer"
        )

    return ChatResponse(
        thread_id=thread_id,
        request_id=request_id,
        answer=answer,
    )
