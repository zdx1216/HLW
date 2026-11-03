
"""HLW PostgreSQL Store 集成测试。"""

import asyncio
import sys
from uuid import uuid4

from hlw.memory.store_postgres import open_postgres_store


async def main() -> None:
    # 每次运行生成独立测试用户，避免污染历史数据。
    run_id = uuid4().hex[:12]

    namespace_a = (
        "users",
        f"test_user_a_{run_id}",
        "memories",
    )

    namespace_b = (
        "users",
        f"test_user_b_{run_id}",
        "memories",
    )

    # 两个用户使用相同的 key。
    key = "learning_style"

    value_a = {
        "category": "learning_preference",
        "content": "喜欢详细解释",
    }

    value_b = {
        "category": "learning_preference",
        "content": "喜欢简短回答",
    }

    async with open_postgres_store() as store:

        print("STEP 1: PostgreSQL Store connected and initialized")

        # 1. 分别保存两名用户的记忆。
        await store.aput(namespace_a, key, value_a)
        await store.aput(namespace_b, key, value_b)

        print("STEP 2: Two users' memories saved")

        # 2. 分别读取记忆。
        item_a = await store.aget(namespace_a, key)
        item_b = await store.aget(namespace_b, key)

        assert item_a is not None
        assert item_b is not None

        assert item_a.value["content"] == "喜欢详细解释"
        assert item_b.value["content"] == "喜欢简短回答"

        print("STEP 3: Namespace isolation PASS")

        # 3. 更新用户 A 的记忆。
        updated_value = {
            "category": "learning_preference",
            "content": "喜欢详细的中文代码注释",
        }

        await store.aput(
            namespace_a,
            key,
            updated_value,
        )

        updated_item = await store.aget(
            namespace_a,
            key,
        )

        assert updated_item is not None
        assert (
            updated_item.value["content"]
            == "喜欢详细的中文代码注释"
        )

        print("STEP 4: Memory update PASS")

        # 4. 使用 filter 查询用户 A 的记忆。
        results = await store.asearch(
            namespace_a,
            filter={
                "category": "learning_preference"
            },
            limit=10,
        )

        assert any(
            item.key == key
            and item.value["content"]
            == "喜欢详细的中文代码注释"
            for item in results
        )

        print("STEP 5: Memory search PASS")

        # 5. 删除用户 A 的记忆。
        await store.adelete(namespace_a, key)

        deleted_item = await store.aget(
            namespace_a,
            key,
        )

        assert deleted_item is None

        # 确认用户 B 的记忆仍然存在。
        remaining_item = await store.aget(
            namespace_b,
            key,
        )

        assert remaining_item is not None
        assert (
            remaining_item.value["content"]
            == "喜欢简短回答"
        )

        # 清理用户 B 的测试数据。
        await store.adelete(namespace_b, key)

        print("STEP 6: Memory deletion and isolation PASS")

    print("RESULT: PostgreSQL Store integration PASS")


if __name__ == "__main__":

    # Windows 下 psycopg 异步连接需要兼容的事件循环。
    if sys.platform == "win32":
        with asyncio.Runner(
                    loop_factory=asyncio.SelectorEventLoop
                ) as runner:

            runner.run(main())
    else:
            asyncio.run(main())
    
