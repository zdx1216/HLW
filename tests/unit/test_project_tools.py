
"""Unit tests for the HLW project status tool."""

import json

from hlw.tools.project_tools import get_project_status


def load_status() -> dict:
    """Invoke the tool and parse its JSON response."""
    raw = get_project_status.invoke({})
    return json.loads(raw)


def test_project_status() -> None:
    """Verify the project's identity and current milestone."""
    result = load_status()

    assert result["project"] == "Hermes Local Workbench"
    assert result["project_id"] == "HLW"
    assert result["version"] == "0.1.0-dev"
    assert result["phase"] == "P0"
    assert result["milestone"] == "P0-02"


def test_project_capabilities() -> None:
    """Verify capability classification and prevent status conflicts."""
    result = load_status()

    implemented = set(result["implemented"])
    in_progress = set(result["in_progress"])
    planned = set(result["planned"])
    verified = result["verified"]

    # Implemented capabilities.
    assert "Python project structure" in implemented
    assert "Environment configuration" in implemented
    assert "Deep Agent Core" in implemented
    assert "Model Gateway" in implemented
    assert "Controlled Retry and Fallback" in implemented
    assert "Async Agent Service" in implemented
    assert "PostgreSQL Checkpointer adapter" in implemented

    # Verified behavior.
    assert any("PostgreSQL connection" in item for item in verified)
    assert any("checkpoint write/read" in item for item in verified)
    assert any("Agent Service" in item for item in verified)

    # Work that has started but is not yet fully integrated.
    assert (
        "Integrate PostgreSQL Checkpointer into the "
        "LangGraph API Server runtime"
    ) in in_progress
    assert "Dockerize the HLW application service" in in_progress
    assert (
        "Per-thread concurrency safety for requests sharing "
        "the same thread_id"
    ) in in_progress

    # Planned capabilities.
    assert "Knowledge Engine" in planned
    assert "Skills Hub" in planned
    assert "MCP Gateway" in planned
    assert "Multi-Agent Engine" in planned
    assert "Secure Execution" in planned
    assert "Evaluation Center" in planned

    # Implemented, in-progress, and planned must be disjoint.
    assert implemented.isdisjoint(in_progress)
    assert implemented.isdisjoint(planned)
    assert in_progress.isdisjoint(planned)

    # Implemented capabilities must not be reported as planned.
    assert "Model Gateway" not in planned
    assert "Deep Agent Core" not in planned


def test_tool_returns_valid_json() -> None:
    """Verify the response schema and field types."""
    raw = get_project_status.invoke({})
    result = json.loads(raw)

    required_fields = {
        "project",
        "project_id",
        "version",
        "phase",
        "milestone",
        "status_policy",
        "implemented",
        "verified",
        "in_progress",
        "planned",
    }

    assert isinstance(result, dict)
    assert required_fields.issubset(result.keys())

    assert isinstance(result["status_policy"], dict)

    for category in (
        "implemented",
        "verified",
        "in_progress",
        "planned",
    ):
        values = result[category]

        assert isinstance(values, list)
        assert len(values) > 0
        assert all(
            isinstance(item, str) and item.strip()
            for item in values
        )
