
"""HLW 长期记忆去重与冲突处理。

负责：
1. 查询已有记忆。
2. 判断新记忆是否重复。
3. 识别同一 key 下的内容冲突。
4. 根据明确的更新授权决定是否覆盖。
5. 防止确认更新时重新创建已经删除的记忆。
6. 校验待确认记忆对应的原始快照，防止覆盖更新后的记忆。
7. 使用 MemoryExecutionLocks 保证同一条记忆的操作串行执行。

本模块不直接访问数据库。
所有数据库操作通过 MemoryManager 完成。
"""

from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.manager import MemoryManager
from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryDeduplicationResult,
)


class MemoryDeduplicator:
    """统一管理长期记忆的新增、去重和更新判断。"""

    def __init__(
        self,
        manager: MemoryManager,
        execution_locks: MemoryExecutionLocks,
    ) -> None:
        """通过依赖注入获得数据库管理器和并发锁。

        Args:
            manager:
                负责长期记忆的数据库读写。

            execution_locks:
                负责控制同一条记忆的并发访问。
                必须由外部统一创建并共享。
        """

        self._manager = manager
        self._execution_locks = execution_locks

    async def process(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        update_confirmed: bool = False,
        require_existing: bool = False,
    ) -> MemoryDeduplicationResult:
        """处理一条候选记忆。

        整个读取、判断和写入过程都处于同一把锁的保护下。

        Args:
            user_id:
                当前用户 ID。

            candidate:
                Extractor 提取并经过 Validator 校验的记忆。

            update_confirmed:
                是否已经明确确认更新旧记忆。

            require_existing:
                是否要求原有记忆必须存在。
                用于防止确认更新时重新创建已经删除的记忆。

        Returns:
            MemoryDeduplicationResult:
                created / unchanged / conflict / updated
        """

        # STEP 1：获取当前用户、当前记忆对应的锁。
        async with self._execution_locks.hold(
            user_id,
            candidate.key,
        ):
            # STEP 2：进入受保护的记忆处理过程。
            return await self._process_locked(
                user_id=user_id,
                candidate=candidate,
                update_confirmed=update_confirmed,
                require_existing=require_existing,
                expected_snapshot=None,
            )

    async def _process_locked(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        update_confirmed: bool,
        require_existing: bool,
        expected_snapshot: dict[str, str] | None = None,
    ) -> MemoryDeduplicationResult:
        """在已经获得记忆锁的情况下执行读、判断和写操作。

        注意：
            本方法不再获取锁。
            只能由已经持有对应锁的内部流程调用。

        Args:
            expected_snapshot:
                待确认操作创建时保存的原始记忆快照。
                如果提供，则要求当前记忆与原始快照一致。
        """

        # STEP 1：查询用户是否已经保存过相同 key 的记忆。
        existing = await self._manager.get_memory(
            user_id=user_id,
            key=candidate.key,
        )

        # STEP 2：如果不存在，判断是否允许新增。
        if existing is None:
            # 确认更新时，原有记忆必须仍然存在。
            if require_existing:
                return MemoryDeduplicationResult(
                    key=candidate.key,
                    status="conflict",
                    message="原有记忆已不存在，取消本次更新",
                )

            # 正常情况下，不存在的记忆可以直接新增。
            await self._manager.save_memory(
                user_id=user_id,
                key=candidate.key,
                content=candidate.content,
                category=candidate.category,
            )

            return MemoryDeduplicationResult(
                key=candidate.key,
                status="created",
                message="首次发现该记忆，已经保存",
            )

        # STEP 3：获取旧记忆的内容和类别。
        existing_content = existing.get("content")
        existing_category = existing.get("category")

        # STEP 4：如果当前记忆已经等于目标内容，
        # 说明目标状态已经实现，不需要重复更新。
        if (
            existing_content == candidate.content
            and existing_category == candidate.category
        ):
            return MemoryDeduplicationResult(
                key=candidate.key,
                status="unchanged",
                message="记忆内容没有变化，跳过保存",
                existing_content=existing_content,
            )

        # STEP 5：校验原始记忆快照。
        # 防止用户确认期间，原有记忆被其他操作修改。
        if expected_snapshot is not None:

            snapshot_fields = (
                "content",
                "category",
                "created_at",
                "updated_at",
            )

            current_snapshot = {
                field: existing.get(field)
                for field in snapshot_fields
            }

            # 当前记忆与待确认记录保存的原始快照不一致。
            if current_snapshot != expected_snapshot:
                return MemoryDeduplicationResult(
                    key=candidate.key,
                    status="conflict",
                    message="原有记忆已发生变化，待确认记录已过期",
                    existing_content=existing_content,
                )

        # STEP 6：内容不同，但是没有明确更新授权。
        if not update_confirmed:

            # 在持有记忆锁期间保存完整原始快照。
            # 避免 Pipeline 发现冲突后再次读取数据库。
            snapshot_fields = (
                "content",
                "category",
                "created_at",
                "updated_at",
            )

            existing_snapshot = {
                field: existing[field]
                for field in snapshot_fields
            }

            return MemoryDeduplicationResult(
                key=candidate.key,
                status="conflict",
                message="新旧记忆存在差异，等待明确更新确认",
                existing_content=existing_content,
                existing_snapshot=existing_snapshot,
            )


        # STEP 7：明确确认更新后，才修改已有记忆。
        updated = await self._manager.update_memory(
            user_id=user_id,
            key=candidate.key,
            content=candidate.content,
            category=candidate.category,
        )

        # 如果更新过程中记忆已经不存在，不擅自重新创建。
        if updated is None:
            return MemoryDeduplicationResult(
                key=candidate.key,
                status="conflict",
                message="原有记忆已不存在，需要重新检查",
                existing_content=existing_content,
            )

        return MemoryDeduplicationResult(
            key=candidate.key,
            status="updated",
            message="已经根据明确授权更新记忆",
            existing_content=existing_content,
        )

    async def confirm_update(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        *,
        expected_snapshot: dict[str, str] | None = None,
    ) -> MemoryDeduplicationResult:
        """用户明确授权后，执行已有记忆的安全更新。

        自动启用：
        1. update_confirmed=True
        2. require_existing=True

        可选启用原始快照校验：
        1. 检查原始记忆是否发生变化。
        2. 防止过期授权覆盖新记忆。
        3. 支持重复确认的幂等处理。

        注意：
            本方法直接获取执行锁，
            不通过 process() 再次获取锁。
        """

        # STEP 1：检查候选记忆类型。
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError(
                "candidate 必须是 MemoryCandidate 类型"
            )

        # STEP 2：检查并规范化原始快照。
        normalized_snapshot = None

        if expected_snapshot is not None:

            if not isinstance(expected_snapshot, dict):
                raise TypeError(
                    "expected_snapshot 必须是字典"
                )

            required_fields = (
                "content",
                "category",
                "created_at",
                "updated_at",
            )

            normalized_snapshot = {}

            for field in required_fields:
                value = expected_snapshot.get(field)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"expected_snapshot.{field} 不能为空"
                    )

                normalized_snapshot[field] = value.strip()

        # STEP 3：获取当前用户、当前记忆对应的锁。
        async with self._execution_locks.hold(
            user_id,
            candidate.key,
        ):
            # STEP 4：在锁保护下检查快照并执行授权更新。
            return await self._process_locked(
                user_id=user_id,
                candidate=candidate,
                update_confirmed=True,
                require_existing=True,
                expected_snapshot=normalized_snapshot,
            )
