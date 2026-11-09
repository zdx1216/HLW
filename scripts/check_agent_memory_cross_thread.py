
"""HLW 跨 Thread 长期记忆集成测试。"""

import asyncio
import sys
from uuid import uuid4

from hlw.agent.persistent_runtime import open_persistent_agent_service
from hlw.core.config import settings
from hlw.memory.manager import MemoryManager
from hlw.memory.store_postgres import open_postgres_store


async def verify_thread(service, thread_id, request_id, marker):
    """验证指定 Thread 能通过 Agent 检索长期记忆。"""

    result = await service.run(
        (
            "请必须调用 search_user_memories 工具，"
            "查询我此前保存的 HLW 跨会话测试标记。"
            "只能根据工具检索结果回答，不要猜测。"
        ),
        thread_id=thread_id,
        request_id=request_id,
    )

    messages = result["messages"]

    # 检查工具调用
    tool_calls = [
        call.get("name")
        for message in messages
        for call in (getattr(message, "tool_calls", []) or [])
    ]

    assert "search_user_memories" in tool_calls, (
        f"{thread_id}: Agent 未调用记忆工具"
    )

    # 检查真实工具返回
    tool_outputs = [
        str(message.content)
        for message in messages
        if getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "search_user_memories"
    ]

    assert tool_outputs, f"{thread_id}: 没有工具返回结果"

    assert marker in "\n".join(tool_outputs), (
        f"{thread_id}: 未检索到目标长期记忆"
    )

    # 检查最终回答
    answer = ""

    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue

        content = getattr(message, "content", "")

        if isinstance(content, str) and content.strip():
            answer = content
            break

    assert marker in answer, (
        f"{thread_id}: 最终回答没有使用长期记忆"
    )

    print(f"{thread_id}: CROSS-THREAD RETRIEVAL PASS")


async def main():
    run_id = uuid4().hex[:10]
    user_id = settings.user_id

    key = f"p1_cross_thread_{run_id}"
    marker = f"HLW_CROSS_THREAD_{run_id.upper()}"

    thread_a = f"p1-cross-a-{run_id}"
    thread_b = f"p1-cross-b-{run_id}"

    saved = False

    try:
        # STEP 1：写入真实长期记忆
        async with open_postgres_store() as store:
            manager = MemoryManager(store)

            await manager.save_memory(
                user_id=user_id,
                key=key,
                content=(
                    "这是 HLW 跨 Thread 测试记忆，"
                    f"唯一标记为 {marker}。"
                ),
                category="fact",
            )

            saved = True

        print("STEP 1: PostgreSQL memory saved PASS")

        # STEP 2：使用同一个 AgentService
        async with open_persistent_agent_service() as service:

            # 新会话 A
            await verify_thread(
                service,
                thread_a,
                f"req-a-{run_id}",
                marker,
            )

            # 新会话 B
            await verify_thread(
                service,
                thread_b,
                f"req-b-{run_id}",
                marker,
            )

        assert thread_a != thread_b

        print("STEP 2: Different Thread IDs PASS")
        print("STEP 3: Shared User Memory Namespace PASS")

        print("=" * 60)
        print("RESULT: Cross-Thread Memory Integration PASS")
        print("=" * 60)

    finally:
        # STEP 4：清理测试数据
        if saved:
            async with open_postgres_store() as store:
                await MemoryManager(store).delete_memory(
                    user_id=user_id,
                    key=key,
                )

            print("STEP 4: Test memory cleanup PASS")


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
