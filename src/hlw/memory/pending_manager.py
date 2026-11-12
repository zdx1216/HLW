
"""HLW 待确认长期记忆管理器。

负责：
1. 创建待确认记忆。
2. 根据 pending_id 读取待确认记忆。
3. 保存原有记忆快照。
4. 隔离不同用户的待确认记忆。

本模块不负责：
1. 调用 LLM。
2. 判断记忆是否冲突。
3. 执行正式记忆更新。
4. 创建数据库连接。
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from langgraph.store.base import BaseStore

from hlw.core.identity import normalize_user_id
from hlw.memory.schemas import MemoryCandidate


class PendingMemoryManager:
    """统一管理用户待确认的长期记忆。"""

    def __init__(self, store: BaseStore) -> None:
        """通过依赖注入接收 Store。"""
        self._store = store

    @staticmethod
    def _build_namespace(user_id: str) -> tuple[str, ...]:
        """构建用户独立的待确认记忆命名空间。"""

        normalized_user_id = normalize_user_id(user_id)

        return (
            "users",
            normalized_user_id,
            "pending_memories",
        )

    @staticmethod
    def _validate_pending_id(pending_id: str) -> str:
        """校验并规范化待确认记录 UUID。"""

        if not isinstance(pending_id, str) or not pending_id.strip():
            raise ValueError("pending_id 不能为空")

        try:
            return str(UUID(pending_id.strip()))
        except ValueError as exc:
            raise ValueError("pending_id 必须是有效 UUID") from exc

    @staticmethod
    def _validate_snapshot(
        existing_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """检查并复制原有记忆快照。"""

        if not isinstance(existing_snapshot, dict):
            raise TypeError("existing_snapshot 必须是字典")

        snapshot = dict(existing_snapshot)

        required_fields = (
            "content",
            "category",
            "created_at",
            "updated_at",
        )

        for field in required_fields:
            value = snapshot.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"existing_snapshot.{field} 不能为空"
                )

            snapshot[field] = value.strip()

        return snapshot

    async def create_pending(
        self,
        user_id: str,
        candidate: MemoryCandidate,
        existing_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """创建一条独立的待确认记忆记录。

        不会修改原有长期记忆。
        每次创建都会分配独立 UUID。
        """

        # STEP 1：构建用户命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：检查候选记忆类型。
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError(
                "candidate 必须是 MemoryCandidate 类型"
            )

        # STEP 3：校验并复制原记忆快照。
        snapshot = self._validate_snapshot(existing_snapshot)

        # STEP 4：生成唯一待确认 ID。
        pending_id = str(uuid4())

        # STEP 5：记录 UTC 时间。
        now = datetime.now(UTC).isoformat()

        # STEP 6：构建待确认记录。
        record = {
            "pending_id": pending_id,
            "candidate": candidate.model_dump(mode="json"),
            "existing_snapshot": snapshot,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }

        # STEP 7：保存到独立命名空间。
        await self._store.aput(
            namespace,
            pending_id,
            record,
        )

        return record

    async def get_pending(
        self,
        user_id: str,
        pending_id: str,
    ) -> dict[str, Any] | None:
        """根据用户 ID 和 pending_id 读取记录。"""

        # STEP 1：构建并校验用户命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：校验待确认 ID。
        pending_id = self._validate_pending_id(pending_id)

        # STEP 3：读取 Store。
        item = await self._store.aget(
            namespace,
            pending_id,
        )

        if item is None:
            return None

        # STEP 4：转换成普通字典。
        return {
            **dict(item.value),
            "pending_id": item.key,
        }

    async def list_pending(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """获取用户尚未处理的待确认记忆列表。

        支持：
        1. 只查询 pending 状态的记录。
        2. 限制返回数量。
        3. 使用 offset 实现分页。
        4. 隔离不同用户的待确认记录。
        """

        # STEP 1：构建用户命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：校验 limit。
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("limit 必须是正整数")

        # STEP 3：校验 offset。
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
        ):
            raise ValueError("offset 必须是非负整数")

        # STEP 4：查询待确认记录。
        items = await self._store.asearch(
            namespace,
            filter={"status": "pending"},
            limit=limit,
            offset=offset,
        )

        # STEP 5：转换为统一字典结构。
        return [
            {
                **dict(item.value),
                "pending_id": item.key,
            }
            for item in items
        ]

    async def delete_pending(
        self,
        user_id: str,
        pending_id: str,
    ) -> None:
        """删除指定用户的待确认记忆。

        特性：
        1. 只能删除当前用户命名空间中的记录。
        2. 不影响正式长期记忆。
        3. 重复删除不会产生异常。
        """

        # STEP 1：构建用户独立命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：校验待确认记录 ID。
        pending_id = self._validate_pending_id(pending_id)

        # STEP 3：删除待确认记录。
        await self._store.adelete(
            namespace,
            pending_id,
        )
