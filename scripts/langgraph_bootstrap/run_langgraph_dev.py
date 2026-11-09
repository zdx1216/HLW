import asyncio
import sys
from pathlib import Path

# ==================================================
# 1. 加载 HLW 原有虚拟环境依赖
# ==================================================
root = Path.cwd()

site_packages = root / ".venv" / "Lib" / "site-packages"
src = root / "src"

if not site_packages.exists():
    raise RuntimeError(f"依赖目录不存在: {site_packages}")

if not src.exists():
    raise RuntimeError(f"源码目录不存在: {src}")

sys.path.insert(0, str(site_packages))
sys.path.insert(0, str(src))

# ==================================================
# 2. Windows Selector 事件循环
# ==================================================
asyncio.set_event_loop_policy(
    asyncio.WindowsSelectorEventLoopPolicy()
)

# ==================================================
# 3. 修正 Uvicorn 事件循环工厂
# ==================================================
import uvicorn.loops.asyncio as uvicorn_asyncio

def selector_loop_factory(use_subprocess=False):
    print(
        "[HLW] Using SelectorEventLoop",
        flush=True,
    )
    return asyncio.SelectorEventLoop

uvicorn_asyncio.asyncio_loop_factory = selector_loop_factory

print("[HLW] Python:", sys.executable)
print("[HLW] Site packages:", site_packages)
print("[HLW] Event loop policy:",
      type(asyncio.get_event_loop_policy()).__name__)

# ==================================================
# 4. 启动 LangGraph Dev
# ==================================================
from langgraph_cli.cli import cli

cli()
