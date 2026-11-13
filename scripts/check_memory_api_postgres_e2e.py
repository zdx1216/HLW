
"""HLW 真实 PostgreSQL + FastAPI 记忆确认集成测试。

验证：
1. 正式记忆写入 PostgreSQL。
2. Pending 写入并通过 HTTP GET 查询。
3. HTTP POST confirm 更新正式记忆。
4. 重复确认返回 404。
5. HTTP POST reject 保留正式记忆。
6. 重新打开数据库连接验证持久化。
7. 自动清理本次创建的测试数据。

注意：
- 需要提前启动 HLW API。
- 不调用 LLM，不通过 /chat 生成候选。
- 使用随机测试 key，不覆盖现有记忆。
"""

import asyncio
import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from hlw.core.config import settings
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate
from hlw.memory.store_postgres import open_postgres_store

BASE_URL = "http://127.0.0.1:8000"
SNAPSHOT_FIELDS = (
    "content",
    "category",
    "created_at",
    "updated_at",
)


def call_api(
    method: str,
    path: str,
    api_key: str,
) -> tuple[int, object]:
    """使用真实 HTTP 请求访问正在运行的 API。"""

    request = Request(
        url=f"{BASE_URL}{path}",
        method=method,
        headers={
            "X-HLW-API-Key": api_key,
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            return (
                response.status,
                json.loads(response.read().decode("utf-8")),
            )

    except HTTPError as exc:
        with exc:
            return (
                exc.code,
                json.loads(exc.read().decode("utf-8")),
            )


async def main() -> None:
    """执行完整端到端测试。"""

    api_key = os.getenv("HLW_MEMORY_API_KEY", "")

    if len(api_key) < 32:
        raise RuntimeError(
            "请先在 .env 中配置 HLW_MEMORY_API_KEY"
        )

    user_id = settings.user_id

    # 独立测试键，每次运行都不同。
    memory_key = f"hlw_e2e_{uuid4().hex[:16]}"

    old_content = f"HLW E2E original {memory_key}"
    confirmed_content = f"HLW E2E confirmed {memory_key}"
    rejected_content = f"HLW E2E rejected {memory_key}"

    pending_ids: list[str] = []

    print(f"[START] Test memory key: {memory_key}")

    try:
        # =====================================================
        # 1. 在真实 PostgreSQL 创建测试记忆和 Pending
        # =====================================================
        async with open_postgres_store() as store:
            manager = MemoryManager(store)
            pending_manager = PendingMemoryManager(store)

            existing = await manager.get_memory(
                user_id,
                memory_key,
            )

            if existing is not None:
                raise RuntimeError(
                    "测试 key 已存在，拒绝覆盖原有记忆"
                )

            await manager.save_memory(
                user_id=user_id,
                key=memory_key,
                content=old_content,
                category="preference",
            )

            original = await manager.get_memory(
                user_id,
                memory_key,
            )

            assert original is not None

            snapshot = {
                field: original[field]
                for field in SNAPSHOT_FIELDS
            }

            candidate = MemoryCandidate(
                key=memory_key,
                category="preference",
                content=confirmed_content,
                evidence=confirmed_content,
            )

            pending = await pending_manager.create_pending(
                user_id=user_id,
                candidate=candidate,
                existing_snapshot=snapshot,
            )

            confirm_id = pending["pending_id"]
            pending_ids.append(confirm_id)

        print("[PASS] PostgreSQL memory and Pending created")

        # =====================================================
        # 2. HTTP GET 查询 Pending
        # =====================================================
        status, records = call_api(
            "GET",
            "/api/v1/memories/pending?limit=100",
            api_key,
        )

        assert status == 200, (status, records)
        assert isinstance(records, list)

        assert any(
            item["pending_id"] == confirm_id
            for item in records
        )

        print("[PASS] HTTP GET found pending record")

        # =====================================================
        # 3. HTTP POST 确认更新
        # =====================================================
        confirm_url = (
            f"/api/v1/memories/pending/{confirm_id}/confirm"
        )

        status, result = call_api(
            "POST",
            confirm_url,
            api_key,
        )

        assert status == 200, (status, result)
        assert result["status"] == "confirmed", result

        print("[PASS] HTTP confirm succeeded")

        # =====================================================
        # 4. 重复确认必须失败
        # =====================================================
        status, result = call_api(
            "POST",
            confirm_url,
            api_key,
        )

        assert status == 404, (status, result)

        print("[PASS] Repeated confirmation returns 404")

        # =====================================================
        # 5. 重新连接数据库，验证更新持久化
        # =====================================================
        async with open_postgres_store() as store:
            manager = MemoryManager(store)
            pending_manager = PendingMemoryManager(store)

            updated = await manager.get_memory(
                user_id,
                memory_key,
            )

            assert updated is not None
            assert updated["content"] == confirmed_content

            assert await pending_manager.get_pending(
                user_id,
                confirm_id,
            ) is None

            print("[PASS] Confirmed content persisted in PostgreSQL")

            # 再创建一条需要拒绝的 Pending。
            snapshot = {
                field: updated[field]
                for field in SNAPSHOT_FIELDS
            }

            reject_candidate = MemoryCandidate(
                key=memory_key,
                category="preference",
                content=rejected_content,
                evidence=rejected_content,
            )

            reject_pending = await pending_manager.create_pending(
                user_id=user_id,
                candidate=reject_candidate,
                existing_snapshot=snapshot,
            )

            reject_id = reject_pending["pending_id"]
            pending_ids.append(reject_id)

        # =====================================================
        # 6. HTTP POST 拒绝更新
        # =====================================================
        status, result = call_api(
            "POST",
            f"/api/v1/memories/pending/{reject_id}/reject",
            api_key,
        )

        assert status == 200, (status, result)
        assert result["status"] == "rejected", result

        print("[PASS] HTTP reject succeeded")

        # =====================================================
        # 7. 重新连接，验证拒绝未修改正式记忆
        # =====================================================
        async with open_postgres_store() as store:
            manager = MemoryManager(store)
            pending_manager = PendingMemoryManager(store)

            current = await manager.get_memory(
                user_id,
                memory_key,
            )

            assert current is not None
            assert current["content"] == confirmed_content

            assert await pending_manager.get_pending(
                user_id,
                reject_id,
            ) is None

        print("[PASS] Rejected update left memory unchanged")
        print("[SUCCESS] All PostgreSQL API E2E checks passed")

    finally:
        # =====================================================
        # 8. 清理测试数据
        # =====================================================
        # 只清理本次随机测试 key 和已记录的 Pending ID。
        # 不删除其他用户记忆。
        try:
            async with open_postgres_store() as store:
                manager = MemoryManager(store)
                pending_manager = PendingMemoryManager(store)

                for pending_id in pending_ids:
                    await pending_manager.delete_pending(
                        user_id,
                        pending_id,
                    )

                await manager.delete_memory(
                    user_id,
                    memory_key,
                )

            print("[CLEANUP] Test data removed")

        except Exception:
            print(
                "[CLEANUP FAILED] Manual cleanup may be "
                f"required for key: {memory_key}"
            )
            raise


if __name__ == "__main__":
    # Windows 下 psycopg 异步连接使用 SelectorEventLoop。
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop,
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
