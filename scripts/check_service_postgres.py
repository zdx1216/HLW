"""冒烟测试：验证 AgentService 的 PostgreSQL 持久化能力。

流程：先在第一个服务生命周期内写入一轮对话，退出后连接关闭；
再用第二个生命周期连回同一个 thread_id，检查历史消息是否被恢复。
"""

import asyncio
import sys

from hlw.agent.persistent_runtime import (
    open_persistent_agent_service,
)


async def main() -> None:
    """执行两段式持久化测试：先写入，再重连读取。"""

    # 固定的会话线程标识：两次生命周期都使用同一个 thread_id，
    # 才能确认状态确实来自数据库，而不是残留在进程内存中。
    thread_id = "hlw-service-pg-123"

    print("STEP 1: New thread")
    print("THREAD_ID:", thread_id)

    first_message = "Use get_project_status and tell me the current milestone."

    # 第一个生命周期：打开服务并写入首轮对话。
    # 退出 with 后 PostgreSQL 连接关闭，内存中的状态不再保留。
    async with open_persistent_agent_service() as service:
        first_result = await service.run(
            first_message,
            thread_id=thread_id,
        )

        print(
            "STEP 2: First run messages:",
            len(first_result["messages"]),
        )

    print("STEP 3: First database context closed")

    # 第二个生命周期：新建连接，但沿用同一个 thread_id，
    # 用于检查上一轮对话是否已从数据库恢复。
    async with open_persistent_agent_service() as service:
        second_result = await service.run(
            (
                "Which tool did I ask you to use "
                "in my previous message? "
                "Reply with only the tool name."
            ),
            thread_id=thread_id,
        )

        messages = second_result["messages"]

        # 在恢复出的消息里查找第一轮的原始提问；
        # 找不到就说明持久化或恢复链路有问题。
        previous_message_found = any(
            getattr(msg, "type", None) == "human"
            and getattr(msg, "content", None) == first_message
            for msg in messages
        )

        if not previous_message_found:
            raise RuntimeError("Previous conversation was not restored")

        # 最后一条消息即模型对上一轮提问的回答。
        last_message = messages[-1]

        print("STEP 4: Restored messages:", len(messages))
        print("LAST_REPLY:", last_message.content)

    # 两段生命周期均正常结束，且历史对话成功恢复，测试通过。
    print("RESULT: PostgreSQL service persistence PASS")


if __name__ == "__main__":
    if sys.platform == "win32":
        # Windows 默认的 ProactorEventLoop 不支持 add_reader，
        # 而 psycopg 异步连接依赖它，因此改用 SelectorEventLoop。
        # Runner 负责创建循环，并在退出时清理未完成的任务。
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
