"""HLW 异步任务请求上下文。"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar(
    "hlw_request_id",
    default="-",
)


def get_request_id() -> str:
    """获取当前请求标识。"""
    return _request_id.get()


@contextmanager
def request_context(
    request_id: str,
) -> Generator[None, None, None]:
    """设置请求标识，并在退出时恢复。"""

    if not request_id.strip():
        raise ValueError("request_id 不能为空")

    token = _request_id.set(request_id)

    try:
        yield
    finally:
        _request_id.reset(token)
