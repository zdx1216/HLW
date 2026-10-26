
"""HLW 主 Agent 结构测试。"""

from hlw.agent.graph import graph, model
from hlw.core.config import settings
from hlw.tools.project_tools import get_project_status


def test_graph_created():
    """检查 Deep Agent 图已创建。"""

    assert graph is not None
    assert callable(getattr(graph, "invoke", None))
    assert callable(getattr(graph, "ainvoke", None))
    assert callable(getattr(graph, "astream", None))


def test_model_config():
    """检查模型与 HLW 配置保持一致。"""

    assert model.model_name == settings.chat_model


def test_business_tool_registered_definition():
    """检查业务工具可以作为 LangChain Tool 使用。"""

    assert get_project_status.name == "get_project_status"
    assert callable(get_project_status.invoke)
