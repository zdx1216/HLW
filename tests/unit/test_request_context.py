"""HLW 请求上下文测试。"""

import asyncio

from hlw.core.request_context import (
    get_request_id,
    request_context,
)


def test_request_context():
    assert get_request_id() == "-"

    with request_context("req-001"):
        assert get_request_id() == "req-001"

    assert get_request_id() == "-"


def test_nested_context():
    with request_context("req-A"):
        with request_context("req-B"):
            assert get_request_id() == "req-B"

        assert get_request_id() == "req-A"


def test_async_context_isolation():
    async def worker(request_id: str):
        with request_context(request_id):
            await asyncio.sleep(0)
            return get_request_id()

    async def run():
        return await asyncio.gather(
            worker("req-A"),
            worker("req-B"),
            worker("req-C"),
        )

    results = asyncio.run(run())

    assert results == [
        "req-A",
        "req-B",
        "req-C",
    ]
