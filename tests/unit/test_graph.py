"""HLW 主 Agent 结构测试。"""

from hlw.agent.graph import graph, model
from hlw.core.config import settings
from hlw.tools.memory_tools import search_user_memories
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


def test_tools_registered_definition():
    """检查业务工具和长期记忆工具定义。"""

    assert get_project_status.name == "get_project_status"
    assert callable(get_project_status.invoke)

    assert search_user_memories.name == "search_user_memories"
    assert callable(search_user_memories.invoke)


def test_build_agent_graph_injects_store_and_checkpointer(
    monkeypatch,
):
    """验证统一工厂能够注入持久化组件。"""

    import hlw.agent.graph as graph_module

    fake_store = object()
    fake_checkpointer = object()

    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        graph_module,
        "create_deep_agent",
        fake_create_deep_agent,
    )

    result = graph_module.build_agent_graph(
        checkpointer=fake_checkpointer,
        store=fake_store,
    )

    assert result is not None

    assert captured["checkpointer"] is fake_checkpointer
    assert captured["store"] is fake_store

    assert captured["tools"] == [
        get_project_status,
        search_user_memories,
    ]


def test_build_agent_graph_without_persistence(
    monkeypatch,
):
    """验证普通模式不主动注入持久化组件。"""

    import hlw.agent.graph as graph_module

    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        graph_module,
        "create_deep_agent",
        fake_create_deep_agent,
    )

    result = graph_module.build_agent_graph()

    assert result is not None

    assert "checkpointer" not in captured
    assert "store" not in captured

    assert captured["tools"] == [
        get_project_status,
        search_user_memories,
    ]


def test_graph_modes_share_business_configuration(monkeypatch):
    """验证普通模式和持久化模式共享相同业务配置。"""

    import hlw.agent.graph as graph_module

    captured = []

    def fake_create_deep_agent(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(
        graph_module,
        "create_deep_agent",
        fake_create_deep_agent,
    )

    # 创建普通 Graph
    graph_module.build_agent_graph()

    # 创建持久化 Graph
    fake_checkpointer = object()
    fake_store = object()

    graph_module.build_agent_graph(
        checkpointer=fake_checkpointer,
        store=fake_store,
    )

    normal_config, persistent_config = captured

    # 1. 模型一致
    assert normal_config["model"] is persistent_config["model"]

    # 2. Backend 类型一致
    assert type(normal_config["backend"]) is type(persistent_config["backend"])

    # 3. Tools 一致
    assert normal_config["tools"] == persistent_config["tools"]

    # 4. System Prompt 一致
    assert normal_config["system_prompt"] == persistent_config["system_prompt"]

    # 5. Middleware 类型和核心配置一致
    normal_middleware = normal_config["middleware"][0]
    persistent_middleware = persistent_config["middleware"][0]

    assert type(normal_middleware) is type(persistent_middleware)

    assert normal_middleware.fallback_model is persistent_middleware.fallback_model

    assert normal_middleware.max_retries == (persistent_middleware.max_retries)

    # 6. 普通模式没有持久化组件
    assert "checkpointer" not in normal_config
    assert "store" not in normal_config

    # 7. 持久化模式正确注入组件
    assert persistent_config["checkpointer"] is fake_checkpointer
    assert persistent_config["store"] is fake_store


def test_build_agent_graph_rejects_partial_persistence(monkeypatch):
    """验证不允许只注入一个持久化组件。"""

    import pytest

    import hlw.agent.graph as graph_module

    with pytest.raises(
        ValueError,
        match="必须同时提供或同时省略",
    ):
        graph_module.build_agent_graph(
            checkpointer=object(),
        )

    with pytest.raises(
        ValueError,
        match="必须同时提供或同时省略",
    ):
        graph_module.build_agent_graph(
            store=object(),
        )
