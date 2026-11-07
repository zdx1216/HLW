"""HLW 模型调用可观测性事件。

这里只负责「记录」，不参与重试/回退的决策。
调用链：ControlledFallbackMiddleware 处理完一次请求后，
组装一个 ModelCallEvent 交给 emit_model_call，
由它统一以 JSON 单行格式写进日志，方便后续用 grep 或
日志采集工具按字段检索（例如统计所有 fallback_used=true 的调用）。
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Literal

from hlw.core.logging_config import get_logger
from hlw.core.request_context import get_request_id
from hlw.core.event_logger import emit_event
logger = get_logger(__name__)

# 调用结果只有两种：成功 / 失败（失败时配合 error_kind 细分原因）
ModelCallStatus = Literal["success", "failed"]


@dataclass(frozen=True, slots=True)
class ModelCallEvent:
    """一次模型调用的汇总事件。

    frozen=True：事件一旦创建就不可变——观测数据不应在写出前被篡改；
    slots=True：减少内存占用（事件会被频繁创建）。
    """

    # 用户最初请求的模型（回退后仍记录它，便于统计「哪个模型最常触发回退」）
    primary_model: str
    # 实际真正完成这次调用的模型（回退后会变成 fallback_model 的值）
    model_name: str
    # 最终结果：success 或 failed
    status: ModelCallStatus
    # 失败原因分类（CONNECTION/TIMEOUT/...），成功时为 None
    error_kind: str | None
    # 重试消耗的次数（不含首次请求本身，也不含回退那次调用）
    retry_count: int
    # 是否走到了回退模型（主模型重试耗尽后才为 True）
    fallback_used: bool
    # 从发起请求到拿到最终结果的总耗时（毫秒，含重试与等待）
    latency_ms: float


def emit_model_call(event: ModelCallEvent) -> None:
    """以统一 JSON 格式记录模型调用。

    日志等级随结果变化：成功记 INFO，失败记 ERROR，
    这样默认配置（INFO 级别）下既能看见成功调用，
    又能用 ERROR 级别过滤出所有失败。
    """

    # 防脏数据：负数次数/耗时说明上游统计逻辑有 bug，直接抛错暴露问题
    if event.retry_count < 0:
        raise ValueError("retry_count 不能小于零")

    if event.latency_ms < 0:
        raise ValueError("latency_ms 不能小于零")

    # 先打固定标记 "event": "model_call"，再展开事件字段。
    # asdict() 把 dataclass 转成普通 dict，日志系统（JSON handler 等）好处理
    payload = {
        "event": "model_call",
        "request_id": get_request_id(),
        **asdict(event),
    }

    # 成功 → INFO（正常流水）；失败 → ERROR（告警/排查）
    level = logging.INFO if event.status == "success" else logging.ERROR
    emit_event(
        "model.call.summary",
        payload=asdict(event),
        duration_ms=event.latency_ms,
        status=event.status,
    )
    # ensure_ascii=False 让中文原样输出而不是 \\uXXXX 转义，日志可直接读
    logger.log(
        level,
        "model_call %s",
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
    )
