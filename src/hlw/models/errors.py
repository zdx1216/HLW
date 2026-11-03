"""HLW 模型调用异常分类。

把各家 SDK 抛出的异常归一成有限的几种「错误意图」，
供上层（模型网关 / 图节点）决定是重试、切换备用模型还是直接失败。
"""

from enum import StrEnum

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)


class ModelErrorKind(StrEnum):
    """模型错误的归一化类别。

    继承 ``StrEnum``，成员本身就是字符串，便于日志输出和序列化。
    """

    CONNECTION = "connection"  # 连不上服务端（网络不通、端口未监听、DNS 失败）
    TIMEOUT = "timeout"  # 连上了但超时（响应太慢或请求被挂起）
    RATE_LIMIT = "rate_limit"  # HTTP 429，请求过快被限流
    SERVER = "server"  # 5xx，服务端自身故障，通常值得重试
    CLIENT = "client"  # 其余 4xx，请求本身有问题，重试也没用
    UNKNOWN = "unknown"  # 无法归类，交由上层按最保守策略处理


def classify_model_error(
    error: Exception,
) -> ModelErrorKind:
    """识别模型错误的类别。

    判定顺序不可随意调整：``APITimeoutError`` 是
    ``APIConnectionError`` 的子类，若先判断 CONNECTION，
    超时会被误判为连接错误。
    """

    # 超时必须先判：SDK 的 APITimeoutError 继承自 APIConnectionError。
    # 同时兜住 httpx 层超时和内置 TimeoutError，覆盖未包装的传输层异常。
    if isinstance(
        error,
        (APITimeoutError, httpx.TimeoutException, TimeoutError),
    ):
        return ModelErrorKind.TIMEOUT

    if isinstance(
        error,
        (APIConnectionError, httpx.NetworkError, ConnectionError),
    ):
        return ModelErrorKind.CONNECTION

    # 只有拿到 HTTP 状态码的错误才能按下表细分。
    if isinstance(error, APIStatusError):
        status = error.status_code

        # 限流单独归类，便于上层做退避等待而不是直接切备用模型。
        if status == 429:
            return ModelErrorKind.RATE_LIMIT

        # 只把常见的可恢复 5xx 归为 SERVER，避免把 501 之类也当成可重试。
        if status in {500, 502, 503, 504}:
            return ModelErrorKind.SERVER

        # 剩余 4xx：参数、鉴权、模型不存在等，重试无意义。
        if 400 <= status < 500:
            return ModelErrorKind.CLIENT

    return ModelErrorKind.UNKNOWN
