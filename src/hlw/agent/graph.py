
"""Hermes Local Workbench 主 Agent 图入口。"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from hlw.models.gateway import build_model
from hlw.models.resilience import ControlledFallbackMiddleware
from hlw.tools.project_tools import get_project_status

SYSTEM_PROMPT = """
你是 Hermes Local Workbench（HLW）的核心 AI 助手。
当前项目的实际版本、开发阶段和已实现功能，
必须通过 get_project_status 获取。
不得依据系统提示词中的静态信息推断开发进度。
工作规则：
1. 用户询问项目版本、开发阶段、已实现功能或规划时，
   必须调用 get_project_status 工具获取事实。

2. 严格区分工具返回的 implemented 和 planned：
   implemented 表示当前代码中已实现的能力；
   planned 表示未来规划，不能声称已经完成。

3. 不得编造工具调用结果或项目开发进度。
4. 如果工具执行失败，明确说明失败情况。
5. 只根据已经获得的证据描述系统能力。
6. 默认使用中文回答，表达清晰、准确。
你是完整 Hermes Local Workbench 的基础组件，
后续将逐步接入 Memory、Skills、MCP、RAG、
Multi-Agent、安全审批及可观测性等能力。
"""


model = build_model("primary")
fallback_model = build_model("fallback")
fallback_middleware = ControlledFallbackMiddleware(
    fallback_model = fallback_model,
    max_retries=2
)

graph = create_deep_agent(
    model=model,
    backend=StateBackend(),
    tools=[get_project_status],
    middleware=[fallback_middleware],
    system_prompt=SYSTEM_PROMPT,
)

def build_persistent_graph(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
):
    """构建使用外部 Checkpointer 的 HLW Agent Graph。"""

    persistent_fallback = ControlledFallbackMiddleware(
        fallback_model=fallback_model,
        max_retries=2,
    )

    return create_deep_agent(
        model=model,
        backend=StateBackend(),
        tools=[get_project_status],
        middleware=[persistent_fallback],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer, # 短期记忆：持久化当前thread的对话状态
        store=store,               # 长期记忆：支持跨thread的记忆存储和检索
    )
