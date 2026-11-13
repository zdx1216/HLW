
"""HLW 长期记忆 HTTP 接口。

Step 8.5.1：
1. 校验 Memory API Key。
2. 查询当前服务端用户的待确认记忆。
3. 不允许 HTTP 请求自行指定 user_id。
"""

import os
from secrets import compare_digest
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Security,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict

from hlw.core.config import settings
from hlw.memory.confirmation_service import (
    MemoryConfirmationResult,
    MemoryConfirmationService,
)
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate

memory_api_key_header = APIKeyHeader(
    name="X-HLW-API-Key",
    auto_error=False,
)


def require_memory_api_key(
    provided_key: str | None = Security(memory_api_key_header),
) -> None:
    """校验服务端配置的记忆管理访问密钥。

    未配置密钥时默认拒绝访问。
    不接受用户自行指定的 user_id。
    """

    expected_key = os.getenv("HLW_MEMORY_API_KEY", "")

    # 必须显式配置足够长的访问密钥。
    if len(expected_key) < 32:
        raise HTTPException(
            status_code=503,
            detail="Memory API authentication is not configured",
        )

    # 使用常量时间字符串比较。
    if (
        provided_key is None
        or not compare_digest(provided_key, expected_key)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Memory API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


router = APIRouter(
    prefix="/api/v1/memories",
    tags=["memories"],
    dependencies=[Depends(require_memory_api_key)],
)


class PendingMemoryItem(BaseModel):
    """HTTP 返回的待确认记忆结构。"""

    model_config = ConfigDict(extra="forbid")

    pending_id: str
    candidate: MemoryCandidate
    existing_snapshot: dict[str, str]
    status: Literal["pending"]
    created_at: str
    updated_at: str


@router.get(
    "/pending",
    response_model=list[PendingMemoryItem],
)
async def list_pending_memories(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    """查询服务端当前用户尚未确认的长期记忆。

    使用共享 Store，不建立新的数据库连接。
    """

    manager = getattr(
        request.app.state,
        "pending_memory_manager",
        None,
    )

    if not isinstance(manager, PendingMemoryManager):
        raise HTTPException(
            status_code=503,
            detail="Pending memory service is unavailable",
        )

    return await manager.list_pending(
        user_id=settings.user_id,
        limit=limit,
        offset=offset,
    )



def get_memory_confirmation_service(
    request: Request,
) -> MemoryConfirmationService:
    """获取应用生命周期中创建的共享记忆确认服务。"""

    service = getattr(
        request.app.state,
        "memory_confirmation_service",
        None,
    )

    if not isinstance(service, MemoryConfirmationService):
        raise HTTPException(
            status_code=503,
            detail="Memory confirmation service is unavailable",
        )

    return service


def _handle_confirmation_result(
    result: MemoryConfirmationResult,
) -> MemoryConfirmationResult:
    """将确认服务的业务状态转换为 HTTP 响应。"""

    # Pending 不存在，或此前已处理完成。
    if result.status == "not_found":
        raise HTTPException(
            status_code=404,
            detail={
                "status": result.status,
                "message": result.message,
            },
        )

    # 快照过期或待确认记录状态异常。
    if result.status in {"stale", "not_pending"}:
        raise HTTPException(
            status_code=409,
            detail={
                "status": result.status,
                "message": result.message,
            },
        )

    return result


@router.post(
    "/pending/{pending_id}/confirm",
    response_model=MemoryConfirmationResult,
)
async def confirm_pending_memory(
    pending_id: UUID,
    service: Annotated[
        MemoryConfirmationService,
        Depends(get_memory_confirmation_service),
    ],
) -> MemoryConfirmationResult:
    """确认更新待确认记忆。

    根据服务端保存的 Pending ID 和原始快照
    执行安全更新，不接受客户端指定 user_id。
    """

    result = await service.confirm(
        user_id=settings.user_id,
        pending_id=str(pending_id),
    )

    return _handle_confirmation_result(result)


@router.post(
    "/pending/{pending_id}/reject",
    response_model=MemoryConfirmationResult,
)
async def reject_pending_memory(
    pending_id: UUID,
    service: Annotated[
        MemoryConfirmationService,
        Depends(get_memory_confirmation_service),
    ],
) -> MemoryConfirmationResult:
    """拒绝待确认记忆更新。

    删除 Pending，不修改正式长期记忆。
    """

    result = await service.reject(
        user_id=settings.user_id,
        pending_id=str(pending_id),
    )

    return _handle_confirmation_result(result)
