
"""HLW HTTP API entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from hlw.agent.persistent_runtime import (
    open_persistent_agent_service,
)


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
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Keep one persistent AgentService alive across requests."""

    async with open_persistent_agent_service(
        max_concurrent=2
    ) as service:
        app.state.agent_service = service

        try:
            yield
        finally:
            app.state.agent_service = None


app = FastAPI(
    title="Hermes Local Workbench API",
    version="0.1.0",
    lifespan=lifespan,
)


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
        raise RuntimeError("Agent returned no final text answer")

    return ChatResponse(
        thread_id=thread_id,
        request_id=request_id,
        answer=answer,
    )
