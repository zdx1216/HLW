
"""HLW Agent 长期记忆真实集成测试。"""

import asyncio
import sys
from uuid import uuid4

from hlw.agent.persistent_runtime import open_persistent_agent_service
from hlw.core.config import settings
from hlw.memory.manager import MemoryManager
from hlw.memory.store_postgres import open_postgres_store


async def main():
    run_id = uuid4().hex[:10]
    user_id = settings.user_id

    key = f"p1_agent_probe_{run_id}"
    marker = f"HLW_P1_MEMORY_{run_id.upper()}"

    saved = False

    try:
        # STEP 1：向真实 PostgreSQL 写入测试记忆
        async with open_postgres_store() as store:
            manager = MemoryManager(store)

            await manager.save_memory(
                user_id=user_id,
                key=key,
                content=(
                    "HLW P1.1 端到端集成测试的"
                    f"唯一标记是 {marker}。"
                ),
                category="fact",
            )

            saved = True

        print("STEP 1: Test memory saved PASS")

        # STEP 2：启动真实持久化 Agent
        async with open_persistent_agent_service() as service:

            result = await service.run(
                (
                    "请必须调用 search_user_memories 工具，"
                    "查询我此前保存的关于 "
                    "HLW P1.1 端到端长期记忆集成测试的唯一标记。"
                    "只根据工具检索结果回答，不要猜测。"
                ),
                thread_id=f"p1-agent-{run_id}",
                request_id=f"p1-agent-{run_id}",
            )

        print("STEP 2: Persistent Agent execution PASS")

        # STEP 3：检查 Agent 是否真正调用了记忆工具
        messages = result["messages"]

        tool_calls = [
            call.get("name")
            for message in messages
            for call in (getattr(message, "tool_calls", []) or [])
        ]

        print("TOOL CALLS:", tool_calls)

        assert "search_user_memories" in tool_calls, (
            "Agent 没有调用 search_user_memories"
        )

        print("STEP 3: Memory tool invocation PASS")

        # STEP 4：检查工具实际返回的数据
        tool_outputs = [
            str(message.content)
            for message in messages
            if getattr(message, "type", None) == "tool"
            and getattr(message, "name", None) == "search_user_memories"
        ]

        assert tool_outputs, "没有找到记忆工具的返回结果"

        tool_result = "\n".join(tool_outputs)

        assert marker in tool_result, (
            "工具没有检索到刚才保存的测试记忆"
        )

        print("STEP 4: Real memory retrieval PASS")

        # STEP 5：检查 Agent 最终回答
        final_answer = ""

        for message in reversed(messages):
            if getattr(message, "type", None) != "ai":
                continue

            content = getattr(message, "content", "")

            if isinstance(content, str) and content.strip():
                final_answer = content
                break

        print("AGENT ANSWER:", final_answer)

        assert marker in final_answer, (
            "Agent 最终回答没有正确使用检索到的记忆"
        )

        print("STEP 5: Memory-grounded answer PASS")

        print("=" * 60)
        print("RESULT: Agent Memory Integration PASS")
        print("=" * 60)

    finally:
        # STEP 6：清理测试记忆
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
