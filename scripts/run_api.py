
"""Start the HLW API with Windows-compatible asyncio."""

import asyncio
import sys

import uvicorn


async def main() -> None:
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
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
