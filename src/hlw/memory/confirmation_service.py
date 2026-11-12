
"""HLW 长期记忆确认服务。

负责：
1. 用户确认待更新记忆。
2. 用户拒绝待更新记忆。
3. 校验待确认记录及原始记忆快照。
4. 防止重复确认和确认、拒绝之间的并发竞争。

本模块不直接访问数据库。
所有数据操作通过 PendingMemoryManager 和 MemoryDeduplicator 完成。
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryDeduplicationResult,
)


class MemoryConfirmationResult(BaseModel):
    """长期记忆确认操作的统一返回结果。"""

    model_config = ConfigDict(extra="forbid")

    pending_id: str

    status: Literal[
        "confirmed",
        "already_applied",
        "rejected",
        "stale",
        "not_found",
        "not_pending",
    ]

    message: str

    deduplication_result: MemoryDeduplicationResult | None = None


class MemoryConfirmationService:
    """统一处理长期记忆的确认与拒绝。"""

    def __init__(
        self,
        pending_manager: PendingMemoryManager,
        deduplicator: MemoryDeduplicator,
        execution_locks: MemoryExecutionLocks,
    ) -> None:
        """通过依赖注入获得待确认管理器、去重器和执行锁。"""

        self._pending_manager = pending_manager
        self._deduplicator = deduplicator
        self._execution_locks = execution_locks

    @staticmethod
    def _normalize_pending_id(pending_id: str) -> str:
        """校验并规范化待确认记录 UUID。"""

        if not isinstance(pending_id, str) or not pending_id.strip():
            raise ValueError("pending_id 必须是有效 UUID")

        try:
            return str(UUID(pending_id.strip()))
        except ValueError as exc:
            raise ValueError(
                "pending_id 必须是有效 UUID"
            ) from exc

    async def confirm(
        self,
        user_id: str,
        pending_id: str,
    ) -> MemoryConfirmationResult:
        """用户确认后，执行原有记忆的安全更新。

        处理顺序：
        1. 获取待确认记录专用锁。
        2. 查询待确认记录。
        3. 校验记录状态。
        4. 恢复 MemoryCandidate。
        5. 使用原始快照执行授权更新。
        6. 根据执行结果消费待确认记录。

        如果数据库操作异常，不删除待确认记录，
        以便后续重试。
        """

        # STEP 1：规范化 UUID。
        normalized_id = self._normalize_pending_id(pending_id)

        # STEP 2：获取待确认记录专用锁。
        async with self._execution_locks.hold(
            user_id,
            f"pending:{normalized_id}",
        ):
            # STEP 3：读取待确认记录。
            record = await self._pending_manager.get_pending(
                user_id=user_id,
                pending_id=normalized_id,
            )

            if record is None:
                return MemoryConfirmationResult(
                    pending_id=normalized_id,
                    status="not_found",
                    message="待确认记录不存在或已经处理",
                )

            if not isinstance(record, dict):
                raise TypeError("待确认记录必须是字典")

            # STEP 4：检查记录 ID。
            if record.get("pending_id") != normalized_id:
                raise ValueError("待确认记录 ID 不一致")

            # STEP 5：检查记录状态。
            if record.get("status") != "pending":
                return MemoryConfirmationResult(
                    pending_id=normalized_id,
                    status="not_pending",
                    message="该记录已经不处于待确认状态",
                )

            # STEP 6：恢复候选记忆。
            candidate = MemoryCandidate.model_validate(
                record.get("candidate")
            )

            # STEP 7：必须存在原始快照。
            # 不允许快照缺失时绕过快照校验。
            existing_snapshot = record.get("existing_snapshot")

            if not isinstance(existing_snapshot, dict):
                raise TypeError(
                    "existing_snapshot 必须是字典"
                )

            # STEP 8：执行明确授权的安全更新。
            deduplication_result = (
                await self._deduplicator.confirm_update(
                    user_id=user_id,
                    candidate=candidate,
                    expected_snapshot=existing_snapshot,
                )
            )

            # STEP 9：根据更新结果确定最终状态。
            if deduplication_result.status == "updated":
                final_status = "confirmed"
                message = "记忆已经根据用户确认完成更新"

            elif deduplication_result.status == "unchanged":
                final_status = "already_applied"
                message = "目标记忆已经存在，无需重复更新"

            elif deduplication_result.status == "conflict":
                final_status = "stale"
                message = (
                    "原有记忆已经变化或删除，"
                    "本次确认失效，请重新检查"
                )

            else:
                raise RuntimeError(
                    "确认更新返回了非预期状态："
                    f"{deduplication_result.status}"
                )

            # STEP 10：更新成功、目标已存在或快照过期，
            # 均消费这条待确认记录。
            await self._pending_manager.delete_pending(
                user_id=user_id,
                pending_id=normalized_id,
            )

            return MemoryConfirmationResult(
                pending_id=normalized_id,
                status=final_status,
                message=message,
                deduplication_result=deduplication_result,
            )

    async def reject(
        self,
        user_id: str,
        pending_id: str,
    ) -> MemoryConfirmationResult:
        """用户拒绝后，删除待确认记录。

        不调用 MemoryDeduplicator。
        不修改、不删除正式长期记忆。
        """

        # STEP 1：规范化 UUID。
        normalized_id = self._normalize_pending_id(pending_id)

        # STEP 2：获取与 confirm 相同的待确认记录专用锁。
        async with self._execution_locks.hold(
            user_id,
            f"pending:{normalized_id}",
        ):
            # STEP 3：查询待确认记录。
            record = await self._pending_manager.get_pending(
                user_id=user_id,
                pending_id=normalized_id,
            )

            if record is None:
                return MemoryConfirmationResult(
                    pending_id=normalized_id,
                    status="not_found",
                    message="待确认记录不存在或已经处理",
                )

            if not isinstance(record, dict):
                raise TypeError("待确认记录必须是字典")

            if record.get("pending_id") != normalized_id:
                raise ValueError("待确认记录 ID 不一致")

            # STEP 4：检查状态。
            if record.get("status") != "pending":
                return MemoryConfirmationResult(
                    pending_id=normalized_id,
                    status="not_pending",
                    message="该记录已经不处于待确认状态",
                )

            # STEP 5：只删除待确认记录。
            await self._pending_manager.delete_pending(
                user_id=user_id,
                pending_id=normalized_id,
            )

            return MemoryConfirmationResult(
                pending_id=normalized_id,
                status="rejected",
                message="已经拒绝本次记忆更新，原有记忆保持不变",
            )
