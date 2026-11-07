"""Unit tests for the HLW project status tool."""

import json

from hlw.tools.project_tools import get_project_status


def load_status() -> dict:
    """Invoke the tool and parse its JSON response."""
    return json.loads(get_project_status.invoke({}))


def test_project_status() -> None:
    """Verify project identity and active memory-integration milestone."""
    result = load_status()
    assert result["project"] == "Hermes Local Workbench"
    assert result["project_id"] == "HLW"
    assert result["version"] == "0.1.0-dev"
    assert result["phase"] == "P1"
    assert result["milestone"] == "P1-01"
    assert result["as_of"] == "2026-11-07"
    assert "memory" in result["current_focus"].lower()


def test_implemented_capabilities() -> None:
    """Verify already-implemented core, observability, and memory components."""
    implemented = set(load_status()["implemented"])
    expected = {
        "Deep Agent Core",
        "Model Gateway",
        "Async Agent Service",
        "PostgreSQL Checkpointer adapter",
        "PostgreSQL Store adapter",
        "PostgreSQL Store injected into persistent Agent runtime",
        "MemoryManager CRUD and user namespace isolation",
        "pgvector semantic search",
        "MemoryExtractor standalone component",
        "MemoryPipeline standalone component",
        "FastAPI health and chat endpoints",
        "Unified JSONL structured event logging for API and LangGraph Dev",
        "Agent search_user_memories tool registered",
    }
    assert expected.issubset(implemented)


def test_verified_capabilities() -> None:
    """Verify recorded historical evidence without implying live CI status."""
    result = load_status()
    verified = result["verified"]
    assert any("205 passed" in item for item in verified)
    assert any("Ruff lint" in item for item in verified)
    assert any("semantic search integration test passed" in item for item in verified)
    assert any("JSONL" in item for item in verified)
    assert result["verification_scope"]


def test_in_progress_capabilities() -> None:
    """Verify persistent-memory Agent integration remains the active focus."""
    in_progress = set(load_status()["in_progress"])
    expected = {
        "Cross-thread retrieval with one trusted user identity",
        "Integrate automatic memory retrieval into Agent requests",
        "Integrate MemoryPipeline into Agent request lifecycle",
        "Implement explicit confirmation for conflicting memory updates",
        "Implement cross-thread memory integration regression tests",
    }
    assert expected.issubset(in_progress)


def test_planned_capabilities() -> None:
    """Verify future platform extensions stay clearly marked as planned."""
    planned = set(load_status()["planned"])
    expected = {
        "Knowledge Engine and RAG",
        "Skills Hub",
        "MCP Gateway",
        "Multi-Agent Engine",
        "Secure Execution",
        "Evaluation Center (future phase; out of scope for P1)",
        "Web UI",
    }
    assert expected.issubset(planned)


def test_capability_categories_are_disjoint() -> None:
    """Ensure an identical capability is never in conflicting categories."""
    result = load_status()
    implemented = set(result["implemented"])
    in_progress = set(result["in_progress"])
    planned = set(result["planned"])
    assert implemented.isdisjoint(in_progress)
    assert implemented.isdisjoint(planned)
    assert in_progress.isdisjoint(planned)


def test_tool_returns_valid_json() -> None:
    """Verify the existing response fields and the optional P1 planning fields."""
    result = load_status()
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
        "current_focus",
        "next_steps",
    }
    assert required_fields.issubset(result)
    assert isinstance(result["status_policy"], dict)
    for category in ("implemented", "verified", "in_progress", "planned", "next_steps"):
        values = result[category]
        assert isinstance(values, list)
        assert values
        assert all(isinstance(item, str) and item.strip() for item in values)
