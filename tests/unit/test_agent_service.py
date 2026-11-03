
"""HLW Agent Service 集成逻辑测试。"""

import asyncio

import pytest

from hlw.agent.service import AgentService
from hlw.core.request_context import get_request_id


class FakeGraph:
    """不调用 Ollama 的测试图。"""

    async def ainvoke(self, inputs, config):
        await asyncio.sleep(0)

        return {
            "message": inputs["messages"][0]["content"],
            "thread_id": config["configurable"]["thread_id"],
            "request_id": get_request_id(),
            "metadata_id": config["metadata"]["request_id"],
        }


def test_service_passes_context():
    """检查请求标识和线程标识是否正确传递。"""

    async def scenario():
        service = AgentService(
            agent_graph=FakeGraph(),
            max_concurrent=2,
        )

        return await service.run(
            "hello HLW",
            thread_id="thread-001",
            request_id="req-001",
        )

    result = asyncio.run(scenario())

    assert result["message"] == "hello HLW"
    assert result["thread_id"] == "thread-001"
    assert result["request_id"] == "req-001"
    assert result["metadata_id"] == "req-001"
    assert get_request_id() == "-"


def test_service_limits_concurrency():
    """验证真正的 Agent Service 并发限制。"""

    class TrackingGraph:
        def __init__(self):
            self.active = 0
            self.peak = 0

        async def ainvoke(self, inputs, config):
            self.active += 1
            self.peak = max(self.peak, self.active)

            try:
                await asyncio.sleep(0.01)

                return {
                    "request_id": get_request_id(),
                }
            finally:
                self.active -= 1

    async def scenario():
        fake_graph = TrackingGraph()

        service = AgentService(
            agent_graph=fake_graph,
            max_concurrent=2,
        )

        results = await asyncio.gather(
            *(
                service.run(
                    f"task-{i}",
                    thread_id=f"thread-{i}",
                    request_id=f"req-{i}",
                )
                for i in range(8)
            )
        )

        return fake_graph.peak, results

    peak, results = asyncio.run(scenario())

    assert peak == 2

    assert [
        item["request_id"] for item in results
    ] == [
        f"req-{i}" for i in range(8)
    ]


def test_service_propagates_errors():
    """图执行错误不能被 Agent Service 吞掉。"""

    class FailingGraph:
        async def ainvoke(self, inputs, config):
            raise RuntimeError("simulated failure")

    async def scenario():
        service = AgentService(
            agent_graph=FailingGraph(),
            max_concurrent=1,
        )

        with pytest.raises(
            RuntimeError,
            match="simulated failure",
        ):
            await service.run(
                "test",
                thread_id="thread-error",
                request_id="req-error",
            )

        assert get_request_id() == "-"

    asyncio.run(scenario())
