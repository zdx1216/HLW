"""Start the HLW API with Windows-compatible asyncio."""

import asyncio
import sys

import uvicorn

from hlw.core.logging_config import get_logger, setup_logging


async def main() -> None:
    # 先初始化统一日志（控制台 + logs/hlw.log），再启动 uvicorn，
    # 保证下面这条启动日志也按统一格式输出。
    setup_logging()
    logger = get_logger(__name__)
    logger.info("启动 Chat API 服务")

    config = uvicorn.Config(
        "hlw.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
    )

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
