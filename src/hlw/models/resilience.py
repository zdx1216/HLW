"""HLW 模型重试与受控回退中间件。

以 LangChain Agent Middleware 的形式拦截每次模型调用：
1. 只对「可恢复」的错误做有限次数的指数退避重试；
2. 主模型重试用尽后，受控地切换到备用模型再试一次；
3. 不可恢复的错误（如 4xx 鉴权/参数错误）直接抛出，绝不吞错。
"""

import asyncio
import time
from time import perf_counter

from langchain.agents.middleware import AgentMiddleware
from langgraph.errors import GraphBubbleUp

from hlw.core.logging_config import get_logger
from hlw.models.errors import (
    ModelErrorKind,
    classify_model_error,
)
from hlw.models.telemetry import ModelCallEvent, emit_model_call

logger = get_logger(__name__)

# 允许「重试或回退」的错误类别；CLIENT / UNKNOWN 不在其中，
# 意味着参数错误、鉴权失败等问题重试也是徒劳，应立即暴露。
RECOVERABLE_ERRORS = {
    ModelErrorKind.CONNECTION,
    ModelErrorKind.TIMEOUT,
    ModelErrorKind.RATE_LIMIT,
    ModelErrorKind.SERVER,
}


def get_model_name(model) -> str:
    """提取模型名称，兼容真实模型和测试模型。

    真实的 ChatOpenAI 对象有 model_name 属性；
    测试里的替身可能是字符串或没有该属性的对象，
    此时退化为类名，保证遥测日志里永远有一个可读的名字。
    """

    if isinstance(model, str):
        return model

    return str(getattr(model, "model_name", type(model).__name__))


class ControlledFallbackMiddleware(AgentMiddleware):
    """对可恢复模型错误进行有限重试和回退。

    工作流程（假设 max_retries=1）::

        主模型 → 失败 → 等待 0.5s → 主模型重试 → 仍失败 → 备用模型兜底

    备用模型的调用只有一次机会，不再重试，避免故障时无限循环。
    """

    def __init__(
        self,
        fallback_model,
        *,
        max_retries: int = 1,
        initial_delay: float = 0.5,
    ):
        super().__init__()

        # 重试次数和延迟都必须非负，否则直接拒绝构造，快速暴露配置错误。
        if max_retries < 0:
            raise ValueError("max_retries 不能小于零")

        if initial_delay < 0:
            raise ValueError("initial_delay 不能小于零")

        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def _record_event(
        self,
        *,
        primary_model: str,
        model_name: str,
        status: str,
        error_kind: str | None,
        retry_count: int,
        fallback_used: bool,
        started_at: float,
    ) -> None:
        """记录本次模型调用的最终结果。

        无论是成功、重试耗尽失败、还是回退成功/失败，
        所有出口都必须经过这里，保证每次调用恰好留下一条遥测记录。
        started_at 由调用方在进入中间件时记录，传进来统一算总耗时。
        """

        # perf_counter 适合测耗时（单调递增，不受系统时间调整影响）
        elapsed_ms = (perf_counter() - started_at) * 1000

        event = ModelCallEvent(
            primary_model=primary_model,
            model_name=model_name,
            status=status,
            error_kind=error_kind,
            retry_count=retry_count,
            fallback_used=fallback_used,
            latency_ms=elapsed_ms,
        )

        emit_model_call(event)

    def wrap_model_call(self, request, handler):
        """同步调用：记录主模型、重试和回退的最终结果。"""

        # 计时起点：从进入中间件就算起，包含重试等待和回退的全部耗时
        started_at = perf_counter()
        primary = get_model_name(request.model)
        # 记录最后一次错误类别；若一次都没失败（首次即成功），保持 None
        last_error = None

        # max_retries 是「重试」次数，总调用次数 = max_retries + 1（首次 + 重试）
        for attempt in range(self.max_retries + 1):
            try:
                response = handler(request)
            except GraphBubbleUp:
                # LangGraph 的控制流异常（如跳转节点）必须原样冒泡，
                # 不能被当成模型错误去重试
                raise
            except Exception as exc:
                kind = classify_model_error(exc)
                last_error = kind.value
                if kind not in RECOVERABLE_ERRORS:
                    # 不可恢复（鉴权/参数错误等）：记一条失败遥测后立即抛出
                    self._record_event(
                        primary_model=primary,
                        model_name=primary,
                        status="failed",
                        error_kind=last_error,
                        retry_count=attempt,
                        fallback_used=False,
                        started_at=started_at,
                    )
                    raise

                logger.warning(
                    "Primary model failed: kind=%s, attempt=%d",
                    kind.value,
                    attempt + 1,
                )
                if attempt < self.max_retries:
                    # 指数退避：0.5s → 1s → 2s …，给故障服务喘息时间
                    delay = self.initial_delay * (2**attempt)
                    time.sleep(delay)
                    continue

                # 重试已用尽：切换到备用模型，最后的机会
                logger.warning("Switching to fallback model")

                fallback_name = get_model_name(self.fallback_model)

                try:
                    # override 返回带新模型的新 request 对象（原对象不变）
                    response = handler(request.override(model=self.fallback_model))

                except GraphBubbleUp:
                    raise

                except Exception as fallback_exc:
                    # 回退也失败：记录（fallback_used=True）后抛出异常。
                    # 回退模型不重试，只有这一次机会
                    self._record_event(
                        primary_model=primary,
                        model_name=fallback_name,
                        status="failed",
                        error_kind=classify_model_error(fallback_exc).value,
                        retry_count=attempt,
                        fallback_used=True,
                        started_at=started_at,
                    )
                    raise

                # 回退成功：主模型虽然失败过（error_kind 留痕），
                # 但整次调用对外是成功的，status=success
                self._record_event(
                    primary_model=primary,
                    model_name=fallback_name,
                    status="success",
                    error_kind=last_error,
                    retry_count=attempt,
                    fallback_used=True,
                    started_at=started_at,
                )

                return response

            else:
                # for...else：循环没被 break/异常走完 = 每次调用都成功
                # （首次即成功，last_error 为 None）
                self._record_event(
                    primary_model=primary,
                    model_name=primary,
                    status="success",
                    error_kind=last_error,
                    retry_count=attempt,
                    fallback_used=False,
                    started_at=started_at,
                )

                return response

    async def awrap_model_call(self, request, handler):
        """异步调用：记录主模型、重试和回退的最终结果。

        与同步版 wrap_model_call 逻辑完全一致，区别只有两点：
        1. handler 需要 await；
        2. 重试等待用 asyncio.sleep（不阻塞事件循环）而不是 time.sleep。
        LangGraph 根据调用场景自动选用其中一个。
        """

        started_at = perf_counter()
        primary = get_model_name(request.model)
        # 记录最后一次错误类别；若一次都没失败（首次即成功），保持 None
        last_error = None

        # max_retries 是「重试」次数，总调用次数 = max_retries + 1（首次 + 重试）
        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)

            except GraphBubbleUp:
                # LangGraph 控制流异常（如跳转节点）必须原样冒泡，不能重试
                raise

            except Exception as exc:
                kind = classify_model_error(exc)
                last_error = kind.value

                if kind not in RECOVERABLE_ERRORS:
                    # 不可恢复（鉴权/参数错误等）：记失败遥测后立即抛出
                    self._record_event(
                        primary_model=primary,
                        model_name=primary,
                        status="failed",
                        error_kind=last_error,
                        retry_count=attempt,
                        fallback_used=False,
                        started_at=started_at,
                    )
                    raise

                logger.warning(
                    "Primary model failed: kind=%s, attempt=%d",
                    kind.value,
                    attempt + 1,
                )

                if attempt < self.max_retries:
                    # 指数退避：0.5s → 1s → 2s …；异步等待不阻塞其他协程
                    delay = self.initial_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue

                # 重试已用尽：切换到备用模型，最后的机会
                logger.warning("Switching to fallback model")

                fallback_name = get_model_name(self.fallback_model)

                try:
                    # override 返回带新模型的新 request 对象（原对象不变）
                    response = await handler(
                        request.override(model=self.fallback_model)
                    )

                except GraphBubbleUp:
                    raise

                except Exception as fallback_exc:
                    # 回退也失败：记录后抛出；回退模型不重试，只有一次机会
                    self._record_event(
                        primary_model=primary,
                        model_name=fallback_name,
                        status="failed",
                        error_kind=classify_model_error(fallback_exc).value,
                        retry_count=attempt,
                        fallback_used=True,
                        started_at=started_at,
                    )
                    raise

                # 回退成功：对外是成功的调用，status=success（error_kind 留痕）
                self._record_event(
                    primary_model=primary,
                    model_name=fallback_name,
                    status="success",
                    error_kind=last_error,
                    retry_count=attempt,
                    fallback_used=True,
                    started_at=started_at,
                )

                return response

            else:
                # for...else：循环没被异常走完 = 调用成功（首次即成功，error_kind=None）
                self._record_event(
                    primary_model=primary,
                    model_name=primary,
                    status="success",
                    error_kind=last_error,
                    retry_count=attempt,
                    fallback_used=False,
                    started_at=started_at,
                )
                return response
