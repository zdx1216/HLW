"""HLW PostgreSQL + pgvector 长期记忆语义检索集成测试。

验证：
1. MemoryManager 能否保存记忆。
2. 能否使用自然语言进行语义检索。
3. 能否通过 category 过滤。
4. 能否返回向量相似度分数。
5. 能否正确清理测试数据。
"""

import asyncio
import sys
from uuid import uuid4

from hlw.memory.manager import MemoryManager
from hlw.memory.store_postgres import open_postgres_store


async def main() -> None:
    """执行真实 PostgreSQL 语义检索测试。"""

    # ========================================================
    # STEP 1：创建独立测试用户
    # ========================================================

    run_id = uuid4().hex[:12]
    user_id = f"semantic_test_{run_id}"

    manager = None

    test_memories = {
        "learning_style": {
            "content": "喜欢详细的中文代码注释",
            "category": "preference",
        },
        "explanation_style": {
            "content": "喜欢先了解学习目标，再看完整代码和原理解释",
            "category": "preference",
        },
        "project": {
            "content": "正在开发 HLW 长期记忆管理系统",
            "category": "project",
        },
        "device": {
            "content": "使用 Windows 电脑进行 Python 开发",
            "category": "environment",
        },
    }

    # ========================================================
    # STEP 2：连接真实 PostgreSQL Store
    # ========================================================

    async with open_postgres_store() as store:
        manager = MemoryManager(store)

        try:
            print("=" * 60)
            print("STEP 1: PostgreSQL Store 连接成功")
            print("测试用户:", user_id)

            # =================================================
            # STEP 3：写入测试记忆
            # =================================================

            for key, memory in test_memories.items():
                await manager.save_memory(
                    user_id=user_id,
                    key=key,
                    content=memory["content"],
                    category=memory["category"],
                )

            print("STEP 2: 四条长期记忆写入成功")

            # =================================================
            # STEP 4：执行自然语言语义检索
            # =================================================

            results = await manager.search_memories(
                user_id=user_id,
                query="我写程序时希望代码注释是什么语言，解释得多详细？",
                limit=5,
            )

            print("STEP 3: 语义检索完成")
            print("返回数量:", len(results))

            if not results:
                raise AssertionError("语义检索没有返回任何结果")

            # 输出所有检索结果
            for index, result in enumerate(results, start=1):
                print(f"\n第 {index} 条结果")
                print("Key:", result["key"])
                print("内容:", result["content"])
                print("类别:", result["category"])
                print("Score:", result.get("score"))

            # 检查最相关的结果
            best_result = results[0]

            assert best_result["key"] == "learning_style", (
                f"最相关记忆不符合预期，实际结果为：{best_result['key']}"
            )

            assert best_result.get("score") is not None, "没有返回向量相似度分数"

            print("\nSTEP 4: 语义排序 PASS")

            # =================================================
            # STEP 5：测试类别过滤
            # =================================================

            preference_results = await manager.search_memories(
                user_id=user_id,
                query="我的学习和代码习惯是什么？",
                category="preference",
                limit=5,
            )

            assert len(preference_results) > 0

            assert all(item["category"] == "preference" for item in preference_results)

            print("STEP 5: Category 过滤 PASS")

            # =================================================
            # STEP 6：测试返回数量限制
            # =================================================

            limited_results = await manager.search_memories(
                user_id=user_id,
                query="用户的个人背景和项目信息",
                limit=2,
            )

            assert len(limited_results) <= 2

            print("STEP 6: Limit 数量限制 PASS")

            print("\n" + "=" * 60)
            print("RESULT: PostgreSQL Semantic Search PASS")
            print("=" * 60)

        finally:
            # =================================================
            # STEP 7：清理测试数据
            # =================================================

            for key in test_memories:
                await manager.delete_memory(
                    user_id=user_id,
                    key=key,
                )

            print("STEP 7: 测试数据清理完成")


def run() -> None:
    """兼容 Windows 异步数据库连接。"""

    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())

    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
