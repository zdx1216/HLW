
"""HLW Middleware + PostgreSQL + HTTP 集成测试。

验证：
1. Middleware 捕获冲突并创建 Pending。
2. 重复并发捕获不会创建重复 Pending。
3. 真实 HTTP GET 可以查询 Pending。
4. HTTP confirm 更新正式记忆。
5. HTTP confirm/reject 竞争时只有一个操作成功。
6. 验证数据库持久化并清理测试数据。

使用固定提取器，不调用真实 LLM。
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from langchain_core.messages import HumanMessage

from hlw.agent.memory_capture_middleware import MemoryCaptureMiddleware
from hlw.core.config import settings
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate, MemoryExtractionResult
from hlw.memory.store_postgres import open_postgres_store

BASE_URL = "http://127.0.0.1:8000"
API_PATH = "/api/v1/memories/pending"


class FixedExtractor(MemoryExtractor):
    """测试专用提取器，不调用 LLM。"""

    def __init__(self, candidate: MemoryCandidate):
        self._candidate = candidate

    async def extract(self, user_message: str):
        assert self._candidate.evidence in user_message
        return MemoryExtractionResult(
            candidates=[self._candidate]
        )


async def capture(middleware, store, message):
    """通过真实 Middleware 处理模拟用户消息。"""

    request = SimpleNamespace(
        messages=[HumanMessage(content=message)],
        runtime=SimpleNamespace(store=store),
    )

    response = object()

    async def handler(current_request):
        assert current_request is request
        return response

    result = await middleware.awrap_model_call(
        request,
        handler,
    )

    assert result is response


def call_api(method, path, api_key):
    """访问真实运行中的 FastAPI 服务。"""

    request = Request(
        f"{BASE_URL}{path}",
        method=method,
        headers={"X-HLW-API-Key": api_key},
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


async def find_test_pending(pending_manager, user_id, key):
    """分页查询属于本次测试键的 Pending。"""

    offset = 0
    matches = []

    while True:
        records = await pending_manager.list_pending(
            user_id=user_id,
            limit=100,
            offset=offset,
        )

        matches.extend(
            item
            for item in records
            if item.get("candidate", {}).get("key") == key
        )

        if len(records) < 100:
            break

        offset += 100

    return matches


async def main():
    api_key = os.getenv("HLW_MEMORY_API_KEY", "")

    if len(api_key) < 32:
        raise RuntimeError("HLW_MEMORY_API_KEY 未配置")

    user_id = settings.user_id
    key = f"hlw_e2e_{uuid4().hex[:16]}"

    old_content = f"HLW E2E old {key}"
    new_content = f"HLW E2E new {key}"
    race_content = f"HLW E2E race {key}"

    created = False

    print(f"[START] {key}")

    try:
        async with open_postgres_store() as store:
            manager = MemoryManager(store)
            pending_manager = PendingMemoryManager(store)

            # 1. 创建独立测试记忆。
            assert await manager.get_memory(user_id, key) is None

            await manager.save_memory(
                user_id=user_id,
                key=key,
                content=old_content,
                category="preference",
            )
            created = True

            # 2. 通过 Middleware 自动捕获冲突。
            candidate = MemoryCandidate(
                key=key,
                category="preference",
                content=new_content,
                evidence=new_content,
            )

            middleware = MemoryCaptureMiddleware(
                extractor=FixedExtractor(candidate),
            )

            message = f"本次测试的新偏好是 {new_content}"

            await capture(middleware, store, message)

            # 3. 并发重复捕获。
            await asyncio.gather(
                capture(middleware, store, message),
                capture(middleware, store, message),
            )

            records = await find_test_pending(
                pending_manager, user_id, key
            )

            assert len(records) == 1
            pending_id = records[0]["pending_id"]

            original = await manager.get_memory(user_id, key)
            assert original["content"] == old_content

        print("[PASS] Middleware created one Pending")
        print("[PASS] Concurrent repeated capture deduplicated")

        # 4. 真实 HTTP 查询。
        status, data = await asyncio.to_thread(
            call_api, "GET", f"{API_PATH}?limit=100", api_key
        )

        assert status == 200, (status, data)
        assert any(
            item["pending_id"] == pending_id for item in data
        )

        print("[PASS] HTTP GET found captured Pending")

        # 5. 真实 HTTP 确认。
        status, result = await asyncio.to_thread(
            call_api,
            "POST",
            f"{API_PATH}/{pending_id}/confirm",
            api_key,
        )

        assert status == 200, (status, result)
        assert result["status"] == "confirmed"

        async with open_postgres_store() as store:
            manager = MemoryManager(store)
            pending_manager = PendingMemoryManager(store)

            updated = await manager.get_memory(user_id, key)
            assert updated["content"] == new_content

            assert await pending_manager.get_pending(
                user_id, pending_id
            ) is None

            # 6. 创建第二条 Pending，准备并发确认/拒绝。
            snapshot = {
                field: updated[field]
                for field in (
                    "content",
                    "category",
                    "created_at",
                    "updated_at",
                )
            }

            candidate = MemoryCandidate(
                key=key,
                category="preference",
                content=race_content,
                evidence=race_content,
            )

            race_pending = await pending_manager.create_pending(
                user_id=user_id,
                candidate=candidate,
                existing_snapshot=snapshot,
            )

            race_id = race_pending["pending_id"]

        print("[PASS] HTTP confirm persisted updated memory")

        # 7. 对同一 Pending 同时发起确认与拒绝。
        confirm_response, reject_response = await asyncio.gather(
            asyncio.to_thread(
                call_api,
                "POST",
                f"{API_PATH}/{race_id}/confirm",
                api_key,
            ),
            asyncio.to_thread(
                call_api,
                "POST",
                f"{API_PATH}/{race_id}/reject",
                api_key,
            ),
        )

        results = [confirm_response, reject_response]

        assert sorted(status for status, _ in results) == [
            200, 404
        ], results

        winner = next(
            body for status, body in results if status == 200
        )

        assert winner["status"] in {"confirmed", "rejected"}

        # 8. 重新打开连接检查最终状态。
        async with open_postgres_store() as store:
            manager = MemoryManager(store)
            pending_manager = PendingMemoryManager(store)

            current = await manager.get_memory(user_id, key)

            expected = (
                race_content
                if winner["status"] == "confirmed"
                else new_content
            )

            assert current["content"] == expected

            assert await pending_manager.get_pending(
                user_id, race_id
            ) is None

        print(
            "[PASS] Concurrent confirm/reject: "
            f"winner={winner['status']}"
        )
        print("[PASS] Final database state is consistent")
        print("[SUCCESS] Middleware PostgreSQL E2E passed")

    finally:
        # 只清理本次创建的随机记忆键。
        if created:
            try:
                async with open_postgres_store() as store:
                    manager = MemoryManager(store)
                    pending_manager = PendingMemoryManager(store)

                    records = await find_test_pending(
                        pending_manager, user_id, key
                    )

                    for record in records:
                        await pending_manager.delete_pending(
                            user_id, record["pending_id"]
                        )

                    await manager.delete_memory(user_id, key)

                print("[CLEANUP] Test data removed")
            except Exception:
                print(
                    "[CLEANUP FAILED] Check test key:",
                    key,
                )
                raise


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop,
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
