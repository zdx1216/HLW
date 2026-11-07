"""HLW LangGraph Runtime 上下文探针。

临时用于研究 LangGraph Dev 注入的运行时信息。
"""

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from hlw.core.logging_config import get_logger

logger = get_logger(__name__)


class RuntimeContextProbeMiddleware(AgentMiddleware):
    """检查 Agent 运行时上下文。"""

    def before_agent(
        self,
        state: Any,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """在 Agent 执行开始时记录运行上下文。"""

        # STEP 1：获取 LangGraph ExecutionInfo
        execution = runtime.execution_info

        execution_data = None

        if execution is not None:
            execution_data = {
                "thread_id": execution.thread_id,
                "run_id": execution.run_id,
                "task_id": execution.task_id,
                "node_attempt": execution.node_attempt,
                "checkpoint_ns": execution.checkpoint_ns,
            }

        # STEP 2：获取 LangGraph Server 信息
        server = runtime.server_info

        server_data = None

        if server is not None:
            server_data = {
                "assistant_id": server.assistant_id,
                "graph_id": server.graph_id,
                "user_present": server.user is not None,
            }

        # STEP 3：获取 RunnableConfig
        try:
            config = get_config()
            config_error = None
        except RuntimeError as exc:
            config = {}
            config_error = str(exc)

        metadata = config.get("metadata") or {}
        configurable = config.get("configurable") or {}

        # STEP 4：只提取与身份追踪有关的字段
        config_data = {
            "run_id": str(config.get("run_id")) if config.get("run_id") else None,
            "run_name": config.get("run_name"),
            "metadata_keys": sorted(metadata.keys()),
            "configurable_keys": sorted(configurable.keys()),
            "metadata_ids": {
                key: metadata[key]
                for key in ("request_id", "thread_id", "user_id")
                if key in metadata
            },
            "configurable_ids": {
                key: configurable[key]
                for key in ("request_id", "thread_id", "user_id")
                if key in configurable
            },
            "error": config_error,
            "langgraph_request_id": configurable.get("langgraph_request_id"),
        }

        # STEP 5：输出诊断信息
        payload = {
            "event": "runtime_context_probe",
            "execution": execution_data,
            "server": server_data,
            "config": config_data,
        }

        logger.info(
            "runtime_context_probe %s",
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
        )

        return None
