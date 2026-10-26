
import json

from hlw.tools.project_tools import get_project_status


def test_project_status():
    """检查项目状态内容是否正确。"""

    raw = get_project_status.invoke({})

    result = json.loads(raw)

    assert result["project_id"] == "HLW"
    assert result["phase"] == "P0"
    assert result["milestone"] == "P0-01"

    assert result["version"] == "0.1.0-dev"


def test_project_capabilities():
    """检查已实现和规划中的能力。"""

    result = json.loads(
        get_project_status.invoke({})
    )

    assert "Environment configuration" in (
        result["implemented"]
    )

    assert "Model Gateway" in (
        result["planned"]
    )

    assert "Multi-Agent Engine" in (
        result["planned"]
    )


def test_tool_returns_valid_json():
    """检查工具是否返回合法的 JSON。"""

    raw = get_project_status.invoke({})

    result = json.loads(raw)

    assert isinstance(result, dict)
    assert isinstance(result["implemented"], list)
    assert isinstance(result["planned"], list)
