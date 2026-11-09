import asyncio
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

    print(
        f"[HLW asyncio bootstrap] "
        f"PID={os.getpid()} "
        f"Policy={type(asyncio.get_event_loop_policy()).__name__}",
        file=sys.stderr,
        flush=True,
    )
