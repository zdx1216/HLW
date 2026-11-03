
"""检查 HLW 与 PostgreSQL Checkpointer 的异步连接。"""

import asyncio
import sys

from hlw.memory.checkpoints_postgres import (
    open_postgres_checkpointer,
)


async def main() -> None:
    """验证 PostgreSQL 连接和 Checkpointer 初始化。"""

    # 打开 Checkpointer 上下文即完成一次真实的数据库连接与初始化；
    # 退出 with 块后连接会被自动关闭。
    async with open_postgres_checkpointer() as saver:
        print("PostgreSQL Checkpointer 连接成功")
        # 打印实际使用的 Saver 类型，便于确认走的是 PostgreSQL 实现。
        print("Saver:", type(saver).__name__)


def run() -> None:
    """启动异步测试，兼容 Windows 上的 psycopg。"""

    if sys.platform == "win32":
        # Windows 默认的 ProactorEventLoop 不支持 add_reader，
        # 而 psycopg 异步连接依赖它，因此改用 SelectorEventLoop。
        # Runner 负责创建循环，并在退出时清理未完成的任务。
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
