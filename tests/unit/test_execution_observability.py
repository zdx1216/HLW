"""HLW Agent 生命周期日志测试。"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import hlw.agent.execution_observability as observability


def make_runtime(run_id="run-001"):
    """创建模拟 LangGraph Runtime。"""

    return SimpleNamespace(
        execution_info=SimpleNamespace(
            run_id=run_id,
            thread_id="thread-001",
        ),
        server_info=SimpleNamespace(
            assistant_id="assistant-001",
            graph_id="hlw_core",
        ),
    )


@pytest.fixture
def events(monkeypatch):
    """拦截结构化事件写入。"""

    records = []

    def fake_emit(event, **kwargs):
        records.append(
            {
                "event": event,
                **kwargs,
            }
        )

    monkeypatch.setattr(
        observability,
        "emit_event",
        fake_emit,
    )

    monkeypatch.setattr(
        observability,
        "get_config",
        lambda: {
            "configurable": {
                "langgraph_request_id": "request-001",
                "user_id": "zdx",
                "thread_id": "thread-001",
            }
        },
    )

    return records


def test_agent_lifecycle_success(events):
    """验证开始、完成、问题、回答及耗时。"""

    middleware = observability.AgentRunAuditMiddleware()
    runtime = make_runtime()

    start_state = {
        "messages": [
            HumanMessage(content="查询 HLW 当前开发进度"),
        ]
    }

    end_state = {
        "messages": [
            HumanMessage(content="查询 HLW 当前开发进度"),
            AIMessage(content="HLW 当前处于基础建设阶段"),
        ]
    }

    assert middleware.before_agent(start_state, runtime) is None
    assert middleware.after_agent(end_state, runtime) is None

    assert len(events) == 2

    started, completed = events

    assert started["event"] == "agent.run.started"
    assert started["payload"]["input"] == ("查询 HLW 当前开发进度")

    assert completed["event"] == "agent.run.completed"
    assert completed["payload"]["output"] == ("HLW 当前处于基础建设阶段")

    assert completed["duration_ms"] >= 0
    assert completed["status"] == "success"

    assert started["request_id"] == completed["request_id"]
    assert started["run_id"] == completed["run_id"]


def test_multiple_runs_do_not_mix(events):
    """交错执行的请求不能使用错误的计时状态。"""

    middleware = observability.AgentRunAuditMiddleware()

    runtime_a = make_runtime("run-A")
    runtime_b = make_runtime("run-B")

    middleware.before_agent(
        {"messages": [HumanMessage(content="问题 A")]},
        runtime_a,
    )

    middleware.before_agent(
        {"messages": [HumanMessage(content="问题 B")]},
        runtime_b,
    )

    middleware.after_agent(
        {"messages": [AIMessage(content="回答 B")]},
        runtime_b,
    )

    middleware.after_agent(
        {"messages": [AIMessage(content="回答 A")]},
        runtime_a,
    )

    completed = [event for event in events if event["event"] == "agent.run.completed"]

    assert len(completed) == 2

    assert completed[0]["run_id"] == "run-B"
    assert completed[0]["payload"]["output"] == "回答 B"

    assert completed[1]["run_id"] == "run-A"
    assert completed[1]["payload"]["output"] == "回答 A"

    assert all(event["duration_ms"] >= 0 for event in completed)


def test_missing_messages_are_safe(events):
    """没有消息时，生命周期日志不应报错。"""

    middleware = observability.AgentRunAuditMiddleware()
    runtime = make_runtime()

    middleware.before_agent({}, runtime)
    middleware.after_agent({}, runtime)

    assert events[0]["payload"]["input"] is None
    assert events[1]["payload"]["output"] is None


def test_event_write_failure_does_not_break_agent(
    monkeypatch,
):
    """日志系统异常不能让 Agent 执行失败。"""

    def failing_emit(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(
        observability,
        "emit_event",
        failing_emit,
    )

    middleware = observability.AgentRunAuditMiddleware()
    runtime = make_runtime()

    assert (
        middleware.before_agent(
            {"messages": [HumanMessage(content="你好")]},
            runtime,
        )
        is None
    )

    assert (
        middleware.after_agent(
            {"messages": [AIMessage(content="你好")]},
            runtime,
        )
        is None
    )
