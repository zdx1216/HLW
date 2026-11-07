"""HLW PostgreSQL Checkpoint 持久化实验。"""

import argparse
import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from hlw.memory.checkpoints_postgres import (
    open_postgres_checkpointer,
)


def create_test_graph(checkpointer):
    """创建用于验证持久化的最小 LangGraph。"""

    def reply(state: MessagesState):
        return {"messages": [AIMessage(content="HLW 状态持久化成功")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)

    return builder.compile(checkpointer=checkpointer)


async def main() -> None:
    """执行写入或读取实验。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["write", "read"])
    parser.add_argument("--thread", required=True)
    args = parser.parse_args()

    config = {
        "configurable": {
            "thread_id": args.thread,
        }
    }

    async with open_postgres_checkpointer() as saver:
        graph = create_test_graph(saver)

        if args.action == "write":
            await graph.ainvoke(
                {"messages": [HumanMessage(content="这是 HLW 的持久化测试")]},
                config=config,
            )
            print(f"写入成功: {args.thread}")

        else:
            snapshot = await graph.aget_state(config)
            messages = snapshot.values.get("messages", [])

            if not messages:
                print(f"当前线程没有历史状态: {args.thread}")
                return

            print(f"恢复线程: {args.thread}")
            print(f"消息数量: {len(messages)}")

            for message in messages:
                print(f"{message.type}: {message.content}")


def run() -> None:
    """在 Windows 上为 psycopg 使用兼容的事件循环。"""

    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
