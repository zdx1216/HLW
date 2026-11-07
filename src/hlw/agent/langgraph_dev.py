"""LangGraph Dev 专用启动入口。"""

from importlib import import_module

from hlw.core.logging_config import (
    get_logger,
    setup_logging,
)

# STEP 1：初始化 LangGraph Dev 专属日志
setup_logging(runtime="langgraph_dev")

# STEP 2：记录 Graph 模块启动
logger = get_logger(__name__)
logger.info("HLW LangGraph Dev graph module initialized")

# STEP 3：初始化日志之后，再加载 Agent Graph
graph = import_module("hlw.agent.graph").graph

__all__ = ["graph"]
