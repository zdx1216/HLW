
"""HLW 统一长期记忆管理器。

负责长期记忆的保存、读取、列表检索、更新和删除。

注意：
    本模块不负责创建或关闭数据库连接。
    Store 对象由外部注入，保证资源生命周期统一管理。
"""

from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore


class MemoryManager:
    """统一管理用户的长期记忆。"""

    def __init__(self, store: BaseStore) -> None:
        """接收外部传入的 Store。

        Args:
            store: LangGraph Store 实例。
        """
        self._store = store

    @staticmethod
    def _build_namespace(
        user_id: str,
    ) -> tuple[str, ...]:
        """根据用户 ID 构建长期记忆命名空间。

        不同用户拥有独立的记忆空间。
        同一用户的不同 thread_id 共用该记忆空间。
        """

        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")

        return (
            "users",
            user_id.strip(),
            "memories",
        )

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
    ) -> str:
        """检查字符串是否有效，并去除首尾空格。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} 不能为空"
            )

        return value.strip()

    async def save_memory(
        self,
        user_id: str,
        key: str,
        content: str,
        category: str = "general",
    ) -> dict[str, Any]:
        """保存一条长期记忆。

        如果相同 namespace 下已经存在相同 key，
        则覆盖原有内容，同时保留最初的创建时间。
        """

        namespace = self._build_namespace(user_id)

        key = self._validate_text(key, "key")
        content = self._validate_text(content, "content")
        category = self._validate_text(
            category,
            "category",
        )

        # 记录当前 UTC 时间
        now = datetime.now(UTC).isoformat()

        # 检查是否已经存在相同的记忆
        existing = await self._store.aget(
            namespace,
            key,
        )

        # 已存在则保留原始创建时间
        created_at = (
            existing.value.get("created_at", now)
            if existing is not None
            else now
        )

        # 统一记忆的数据结构
        memory = {
            "content": content,
            "category": category,
            "created_at": created_at,
            "updated_at": now,
        }

        # 写入 PostgreSQL Store
        await self._store.aput(
            namespace,
            key,
            memory,
        )

        return memory

    async def get_memory(
        self,
        user_id: str,
        key: str,
    ) -> dict[str, Any] | None:
        """根据用户 ID 和记忆 key 获取一条记忆。"""

        namespace = self._build_namespace(user_id)
        key = self._validate_text(key, "key")

        item = await self._store.aget(
            namespace,
            key,
        )

        if item is None:
            return None

        return dict(item.value)

    async def list_memories(
        self,
        user_id: str,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """获取用户的记忆列表。

        可以根据 category 筛选。
        """

        namespace = self._build_namespace(user_id)

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("limit 必须是正整数")

        # 如果指定类别，则只获取该类别的记忆
        filters = None

        if category is not None:
            category = self._validate_text(
                category,
                "category",
            )
            filters = {
                "category": category,
            }

        items = await self._store.asearch(
            namespace,
            filter=filters,
            limit=limit,
        )

        return [
            {
                "key": item.key,
                **dict(item.value),
            }
            for item in items
        ]

    async def update_memory(
        self,
        user_id: str,
        key: str,
        *,
        content: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any] | None:
        """更新已有记忆。

        如果记忆不存在，返回 None。
        只更新传入的字段，不改变其他字段。
        """

        namespace = self._build_namespace(user_id)
        key = self._validate_text(key, "key")

        if content is None and category is None:
            raise ValueError("至少提供一个需要更新的字段")

        # 首先读取原有记忆
        existing = await self._store.aget(
            namespace,
            key,
        )

        if existing is None:
            return None

        # 复制原有数据，避免丢失其他字段
        memory = dict(existing.value)

        # 只修改用户明确提供的字段
        if content is not None:
            memory["content"] = self._validate_text(
                content,
                "content",
            )

        if category is not None:
            memory["category"] = self._validate_text(
                category,
                "category",
            )

        # 更新修改时间
        memory["updated_at"] = (
            datetime.now(UTC).isoformat()
        )

        await self._store.aput(
            namespace,
            key,
            memory,
        )

        return memory

    async def delete_memory(
        self,
        user_id: str,
        key: str,
    ) -> bool:
        """删除一条记忆。

        返回 True 表示成功删除。
        返回 False 表示记忆原本不存在。
        """

        namespace = self._build_namespace(user_id)
        key = self._validate_text(key, "key")

        # 先确认记忆是否存在
        existing = await self._store.aget(
            namespace,
            key,
        )

        if existing is None:
            return False

        # 执行删除
        await self._store.adelete(
            namespace,
            key,
        )

        return True
