"""HLW 统一长期记忆管理器。

负责长期记忆的：
1. 保存
2. 读取
3. 列表检索
4. 分类筛选
5. 分页检索
6. 更新
7. 删除

安全要求：
- 不同用户的记忆必须隔离。
- user_id 必须经过统一规范化。
- 不允许非法身份访问记忆。
- 数据库连接由外部统一管理。
"""

from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore

from hlw.core.identity import normalize_user_id


class MemoryManager:
    """统一管理用户的长期记忆。"""

    def __init__(self, store: BaseStore) -> None:
        """通过依赖注入接收 Store。

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

        user_id 必须经过统一身份校验和规范化。
        """

        normalized_user_id = normalize_user_id(user_id)

        return (
            "users",
            normalized_user_id,
            "memories",
        )

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
    ) -> str:
        """检查字符串是否有效，并去除首尾空格。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 不能为空")

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

        # STEP 1：构建并校验用户命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：检查输入内容。
        key = self._validate_text(
            key,
            "key",
        )

        content = self._validate_text(
            content,
            "content",
        )

        category = self._validate_text(
            category,
            "category",
        )

        # STEP 3：记录当前 UTC 时间。
        now = datetime.now(UTC).isoformat()

        # STEP 4：检查是否已经存在相同记忆。
        existing = await self._store.aget(
            namespace,
            key,
        )

        # STEP 5：已存在则保留原始创建时间。
        created_at = (
            existing.value.get("created_at", now) if existing is not None else now
        )

        # STEP 6：统一记忆数据结构。
        memory = {
            "content": content,
            "category": category,
            "created_at": created_at,
            "updated_at": now,
        }

        # STEP 7：写入 Store。
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
        """根据用户 ID 和记忆 Key 获取一条记忆。"""

        # STEP 1：构建用户命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：检查 Key。
        key = self._validate_text(
            key,
            "key",
        )

        # STEP 3：读取记忆。
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
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """获取用户的记忆列表。

        支持：
        1. 根据 category 筛选记忆。
        2. 根据 limit 限制返回数量。
        3. 根据 offset 实现分页。

        Args:
            user_id: 用户身份。
            category: 可选的记忆类别。
            limit: 最大返回数量。
            offset: 跳过的记录数量。

        Returns:
            符合条件的记忆列表。
        """

        # STEP 1：构建并校验用户命名空间。
        namespace = self._build_namespace(user_id)

        # STEP 2：检查 limit。
        # 必须是大于等于 1 的整数。
        # bool 是 int 的子类，因此需要单独排除。
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit 必须是正整数")

        # STEP 3：检查 offset。
        # offset 可以为 0，但不能是负数。
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset 必须是非负整数")

        # STEP 4：构建类别筛选条件。
        filters = None

        if category is not None:
            category = self._validate_text(
                category,
                "category",
            )

            filters = {
                "category": category,
            }

        # STEP 5：执行数据库分页检索。
        items = await self._store.asearch(
            namespace,
            filter=filters,
            limit=limit,
            offset=offset,
        )

        # STEP 6：转换为统一的字典结构。
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

        # STEP 1：构建用户命名空间。
        namespace = self._build_namespace(user_id)

        key = self._validate_text(
            key,
            "key",
        )

        if content is None and category is None:
            raise ValueError("至少提供一个需要更新的字段")

        # STEP 2：读取原有记忆。
        existing = await self._store.aget(
            namespace,
            key,
        )

        if existing is None:
            return None

        # STEP 3：复制原有数据。
        memory = dict(existing.value)

        # STEP 4：只修改用户明确提供的字段。
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

        # STEP 5：更新修改时间。
        memory["updated_at"] = datetime.now(UTC).isoformat()

        # STEP 6：写入 Store。
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

        # STEP 1：构建用户命名空间。
        namespace = self._build_namespace(user_id)

        key = self._validate_text(
            key,
            "key",
        )

        # STEP 2：确认记忆是否存在。
        existing = await self._store.aget(
            namespace,
            key,
        )

        if existing is None:
            return False

        # STEP 3：执行删除。
        await self._store.adelete(
            namespace,
            key,
        )

        return True

    async def search_memories(
        self,
        user_id: str,
        query: str,
        category: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """通过自然语言语义检索用户长期记忆。

        支持：
        1. 根据自然语言查询。
        2. 根据记忆类别筛选。
        3. 限制返回数量。
        4. 分页检索。
        5. 返回语义相似度分数。
        """

        # STEP 1：构建用户命名空间。
        # 保证不同用户的记忆相互隔离。
        namespace = self._build_namespace(user_id)

        # STEP 2：检查查询文本。
        query = self._validate_text(
            query,
            "query",
        )

        # STEP 3：检查 limit。
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit 必须是正整数")

        # STEP 4：检查 offset。
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset 必须是非负整数")

        # STEP 5：构建类别过滤条件。
        filters = None

        if category is not None:
            category = self._validate_text(
                category,
                "category",
            )

            filters = {
                "category": category,
            }

        # STEP 6：执行语义检索。
        items = await self._store.asearch(
            namespace,
            query=query,
            filter=filters,
            limit=limit,
            offset=offset,
        )

        # STEP 7：转换为统一结果结构。
        return [
            {
                "key": item.key,
                **dict(item.value),
                "score": item.score,
            }
            for item in items
        ]
