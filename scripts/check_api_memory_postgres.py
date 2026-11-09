
"""HLW FastAPI 长期记忆真实 HTTP 集成测试。"""

import asyncio
import json
import sys
import urllib.request
from uuid import uuid4

import uvicorn

from hlw.core.config import settings
from hlw.memory.manager import MemoryManager
from hlw.memory.store_postgres import open_postgres_store


def post_chat(port: int, payload: dict) -> dict:
    """通过真实 HTTP 请求调用 FastAPI。"""

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


async def main():
    run_id = uuid4().hex[:10]
    user_id = settings.user_id

    key = f"p1_api_memory_{run_id}"
    marker = f"HLW_API_MEMORY_{run_id.upper()}"

    saved = False
    server = None
    server_task = None

    try:
        # STEP 1：写入真实 PostgreSQL 测试记忆
        async with open_postgres_store() as store:
            await MemoryManager(store).save_memory(
                user_id=user_id,
                key=key,
                content=(
                    "这是 HLW FastAPI 集成测试记忆，"
                    f"唯一标记为 {marker}。"
                ),
                category="fact",
            )

        saved = True
        print("STEP 1: PostgreSQL memory saved PASS")

        # STEP 2：启动真实 Uvicorn + FastAPI
        config = uvicorn.Config(
            "hlw.api.app:app",
            host="127.0.0.1",
            port=0,
            log_level="warning",
            access_log=False,
        )

        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())

        # 等待 FastAPI 完成启动及数据库初始化
        for _ in range(300):
            if server.started:
                break

            if server_task.done():
                await server_task

            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("FastAPI 启动超时")

        sockets = [
            sock
            for instance in (server.servers or [])
            for sock in (instance.sockets or [])
        ]

        if not sockets:
            raise RuntimeError("无法获取 FastAPI 监听端口")

        port = sockets[0].getsockname()[1]

        print(f"STEP 2: FastAPI startup PASS (port={port})")

        # STEP 3：发送真实 HTTP 请求
        payload = {
            "message": (
                "请必须调用 search_user_memories 工具，"
                "查询我之前保存的 HLW FastAPI 集成测试标记。"
                "只能根据工具结果回答，不要猜测。"
            ),
            "thread_id": f"p1-api-{run_id}",
            "request_id": f"p1-api-{run_id}",
        }

        response = await asyncio.to_thread(
            post_chat,
            port,
            payload,
        )

        assert response["thread_id"] == payload["thread_id"]
        assert response["request_id"] == payload["request_id"]

        print("STEP 3: HTTP /chat request PASS")

        # STEP 4：检查最终回答
        answer = response["answer"]

        assert marker in answer, (
            "API 返回结果没有包含真实检索到的记忆"
        )

        print("API ANSWER:", answer)
        print("STEP 4: API memory retrieval PASS")

        print("=" * 60)
        print("RESULT: FastAPI Memory Integration PASS")
        print("=" * 60)

    finally:
        # STEP 5：关闭 API 服务
        if server_task is not None and not server_task.done():
            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=30)

        # STEP 6：清理测试数据
        if saved:
            async with open_postgres_store() as store:
                await MemoryManager(store).delete_memory(
                    user_id=user_id,
                    key=key,
                )

            print("STEP 6: Test memory cleanup PASS")


def run():
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
