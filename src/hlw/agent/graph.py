
"""Hermes Local Workbench 主 Agent 图入口。"""

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_openai import ChatOpenAI

from hlw.core.config import settings
from hlw.tools.project_tools import get_project_status

SYSTEM_PROMPT = """
你是 Hermes Local Workbench（HLW）的核心 AI 助手。
当前项目处于 P0-01 工程初始化阶段。
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


def create_model() -> ChatOpenAI:
    """创建当前配置的本地 Ollama 聊天模型。"""

    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        api_key="ollama",
        temperature=0,
        timeout=120,
        max_retries=0,
    )


model = create_model()


graph = create_deep_agent(
    model=model,
    backend=StateBackend(),
    tools=[get_project_status],
    system_prompt=SYSTEM_PROMPT,
)
