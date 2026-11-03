
"""HLW 异步任务并发限制器测试。"""

import asyncio

import pytest

from hlw.core.concurrency import AgentConcurrencyLimiter


def test_invalid_limit():
    """并发上限不能为零或负数。"""

    for value in (0, -1):
        with pytest.raises(ValueError):
            AgentConcurrencyLimiter(value)


def test_concurrency_limit():
    """同时运行的任务数不能超过上限。"""

    async def scenario():
        limiter = AgentConcurrencyLimiter(2)

        active = 0
        peak = 0

        async def work():
            nonlocal active, peak

            active += 1
            peak = max(peak, active)

            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1

        await asyncio.gather(
            *(limiter.run(work) for _ in range(8))
        )

        return peak

    assert asyncio.run(scenario()) == 2


def test_release_after_exception():
    """任务失败后，必须释放执行许可。"""

    async def scenario():
        limiter = AgentConcurrencyLimiter(1)

        async def failing_task():
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError):
            await limiter.run(failing_task)

        async def successful_task():
            return "recovered"

        return await limiter.run(successful_task)

    assert asyncio.run(scenario()) == "recovered"


def test_release_after_cancellation():
    """任务取消后，执行许可仍能恢复。"""

    async def scenario():
        limiter = AgentConcurrencyLimiter(1)
        entered = asyncio.Event()

        async def long_task():
            entered.set()
            await asyncio.Future()

        task = asyncio.create_task(
            limiter.run(long_task)
        )

        await asyncio.wait_for(
            entered.wait(),
            timeout=1,
        )

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        async def next_task():
            return "released"

        return await asyncio.wait_for(
            limiter.run(next_task),
            timeout=1,
        )

    assert asyncio.run(scenario()) == "released"
