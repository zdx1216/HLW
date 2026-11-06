
"""Unit tests for the HLW project status tool."""

import json

from hlw.tools.project_tools import get_project_status


def load_status() -> dict:
    """Invoke the tool and parse its JSON response."""
    raw = get_project_status.invoke({})
    return json.loads(raw)


def test_project_status() -> None:
    """Verify project identity and milestone."""

    result = load_status()

    assert result["project"] == "Hermes Local Workbench"
    assert result["project_id"] == "HLW"
    assert result["version"] == "0.1.0-dev"
    assert result["phase"] == "P0"
    assert result["milestone"] == "P0-02"
    assert result["as_of"] == "2026-11-06"


def test_implemented_capabilities() -> None:
    """Verify implemented capabilities."""

    result = load_status()
    implemented = set(result["implemented"])

    expected = {
        "Deep Agent Core",
        "Model Gateway",
        "Async Agent Service",
        "PostgreSQL Checkpointer adapter",
        "PostgreSQL Store adapter",
        "MemoryManager CRUD and user namespace isolation",
        "pgvector semantic search",
        "MemoryExtractor standalone component",
        "MemoryPipeline standalone component",
        "FastAPI health and chat endpoints",
    }

    assert expected.issubset(implemented)


def test_verified_capabilities() -> None:
    """Verify recorded verification evidence."""

    result = load_status()
    verified = result["verified"]

    assert any(
        "154 passed" in item
        for item in verified
    )

    assert any(
        "Ruff lint" in item
        for item in verified
    )

    assert any(
        "semantic search integration test passed" in item
        for item in verified
    )

    assert result["verification_scope"]


def test_in_progress_capabilities() -> None:
    """Verify capabilities awaiting integration."""

    result = load_status()
    in_progress = set(result["in_progress"])

    expected = {
        "Inject PostgreSQL Store into persistent Agent runtime",
        "Integrate Store into DeepAgent graph",
        "Develop Agent long-term memory retrieval tools",
        "Integrate automatic memory retrieval into Agent requests",
        "Integrate MemoryPipeline into Agent request lifecycle",
    }

    assert expected.issubset(in_progress)


def test_planned_capabilities() -> None:
    """Verify future capabilities."""

    result = load_status()
    planned = set(result["planned"])

    expected = {
        "Knowledge Engine and RAG",
        "Skills Hub",
        "MCP Gateway",
        "Multi-Agent Engine",
        "Secure Execution",
        "Evaluation Center",
        "Web UI",
    }

    assert expected.issubset(planned)


def test_capability_categories_are_disjoint() -> None:
    """Prevent conflicting capability classifications."""

    result = load_status()

    implemented = set(result["implemented"])
    in_progress = set(result["in_progress"])
    planned = set(result["planned"])

    assert implemented.isdisjoint(in_progress)
    assert implemented.isdisjoint(planned)
    assert in_progress.isdisjoint(planned)


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
        "as_of",
        "status_policy",
        "implemented",
        "verified",
        "verification_scope",
        "in_progress",
        "planned",
    }

    assert isinstance(result, dict)
    assert required_fields.issubset(result.keys())

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
