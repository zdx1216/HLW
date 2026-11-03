
"""Tests for HLW session-level concurrency control."""

import asyncio

import pytest

from hlw.agent.service import AgentService
from hlw.core.thread_locks import ThreadExecutionLocks


class BlockingGraph:
    """A fake Graph whose execution can be paused by tests."""

    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release = {
            name: asyncio.Event()
            for name in ("A", "B", "C")
        }
        self.active_threads: set[str] = set()
        self.active_count = 0
        self.max_active = 0

    async def ainvoke(self, inputs, config):
        name = inputs["messages"][0]["content"]
        thread_id = config["configurable"]["thread_id"]

        # The same thread must never execute twice concurrently.
        assert thread_id not in self.active_threads

        self.active_threads.add(thread_id)
        self.active_count += 1
        self.max_active = max(
            self.max_active,
            self.active_count,
        )
        self.started.put_nowait(name)

        try:
            await self.release[name].wait()
            return {"messages": inputs["messages"]}
        finally:
            self.active_threads.remove(thread_id)
            self.active_count -= 1


def test_same_thread_serialization() -> None:
    """A and C serialize; B can execute alongside A."""

    async def scenario() -> None:
        graph = BlockingGraph()
        service = AgentService(graph, max_concurrent=2)
        tasks = []

        try:
            a = asyncio.create_task(
                service.run("A", thread_id="thread-001")
            )
            tasks.append(a)
            assert await asyncio.wait_for(
                graph.started.get(), 2
            ) == "A"

            c = asyncio.create_task(
                service.run("C", thread_id="thread-001")
            )
            tasks.append(c)

            b = asyncio.create_task(
                service.run("B", thread_id="thread-002")
            )
            tasks.append(b)

            assert await asyncio.wait_for(
                graph.started.get(), 2
            ) == "B"

            assert graph.max_active == 2

            # B finishes, but C must still wait for A.
            graph.release["B"].set()
            await asyncio.wait_for(b, 2)
            await asyncio.sleep(0)

            assert graph.started.empty()

            # A releases its thread lock, allowing C to enter.
            graph.release["A"].set()
            await asyncio.wait_for(a, 2)

            assert await asyncio.wait_for(
                graph.started.get(), 2
            ) == "C"

            graph.release["C"].set()
            await asyncio.wait_for(c, 2)

        finally:
            for gate in graph.release.values():
                gate.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

    asyncio.run(scenario())


def test_global_concurrency_limit() -> None:
    """Three distinct threads must respect the limit of two."""

    async def scenario() -> None:
        graph = BlockingGraph()
        service = AgentService(graph, max_concurrent=2)
        tasks = []

        try:
            a = asyncio.create_task(
                service.run("A", thread_id="thread-001")
            )
            tasks.append(a)

            b = asyncio.create_task(
                service.run("B", thread_id="thread-002")
            )
            tasks.append(b)

            assert await asyncio.wait_for(
                graph.started.get(), 2
            ) == "A"
            assert await asyncio.wait_for(
                graph.started.get(), 2
            ) == "B"

            c = asyncio.create_task(
                service.run("C", thread_id="thread-003")
            )
            tasks.append(c)

            await asyncio.sleep(0)
            assert graph.started.empty()

            # One global slot becomes available.
            graph.release["B"].set()
            await asyncio.wait_for(b, 2)

            assert await asyncio.wait_for(
                graph.started.get(), 2
            ) == "C"

            assert graph.max_active == 2

            graph.release["A"].set()
            graph.release["C"].set()

            await asyncio.wait_for(a, 2)
            await asyncio.wait_for(c, 2)

        finally:
            for gate in graph.release.values():
                gate.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

    asyncio.run(scenario())


def test_cancelled_waiter_cleanup() -> None:
    """Cancelling a waiter must not leak a lock entry."""

    async def scenario() -> None:
        locks = ThreadExecutionLocks()
        attempting = asyncio.Event()
        entered = asyncio.Event()

        async def waiter() -> None:
            attempting.set()
            async with locks.hold("thread-001"):
                entered.set()

        async with locks.hold("thread-001"):
            task = asyncio.create_task(waiter())
            await attempting.wait()

            assert not entered.is_set()

            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert locks._entries["thread-001"].users == 1

        assert not locks._entries

        # The same thread can acquire a fresh lock.
        async with locks.hold("thread-001"):
            pass

        assert not locks._entries

    asyncio.run(scenario())


def test_exception_releases_thread_lock() -> None:
    """A failed request must not block the next request."""

    class FlakyGraph:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, inputs, config):
            self.calls += 1

            if self.calls == 1:
                raise RuntimeError("simulated Graph failure")

            return {"messages": inputs["messages"]}

    async def scenario() -> None:
        graph = FlakyGraph()
        service = AgentService(graph, max_concurrent=2)

        with pytest.raises(
            RuntimeError,
            match="simulated Graph failure",
        ):
            await service.run(
                "first",
                thread_id="thread-001",
            )

        result = await asyncio.wait_for(
            service.run(
                "second",
                thread_id="thread-001",
            ),
            timeout=2,
        )

        assert result["messages"][0]["content"] == "second"
        assert graph.calls == 2

    asyncio.run(scenario())
