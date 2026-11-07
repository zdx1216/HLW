"""HLW 异步任务请求上下文。"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token

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


# 当前 Agent 执行的根 Run ID。
# ContextVar 保证异步并发请求之间互不干扰。
_run_id: ContextVar[str | None] = ContextVar(
    "hlw_run_id",
    default=None,
)


def get_run_id() -> str | None:
    """读取当前执行的 Run ID。"""
    return _run_id.get()


def set_run_id(run_id: str | None) -> Token[str | None]:
    """设置 Run ID，并返回用于恢复上下文的 Token。"""
    return _run_id.set(run_id)


def reset_run_id(token: Token[str | None]) -> None:
    """执行结束后恢复原来的 Run ID。"""
    _run_id.reset(token)
