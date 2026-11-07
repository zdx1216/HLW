"""HLW 正式模型回退配置测试。"""

from hlw.agent.graph import build_fallback_middleware
from hlw.core.config import settings
from hlw.models.resilience import ControlledFallbackMiddleware


def test_fallback_middleware_created():
    """检查回退 Middleware 能够正常创建。"""

    middleware = build_fallback_middleware()

    assert isinstance(
        middleware,
        ControlledFallbackMiddleware,
    )


def test_fallback_model_config():
    """检查备用模型与环境配置一致。"""

    middleware = build_fallback_middleware()

    fallback = middleware.fallback_model

    assert fallback.model_name == settings.fallback_model


def test_retry_config():
    """检查重试次数及退避配置。"""

    middleware = build_fallback_middleware()

    assert middleware.max_retries == 2
    assert middleware.initial_delay >= 0


def test_fallback_middleware_instances_are_independent():
    """检查每次创建的 Middleware 都是独立实例。"""

    first = build_fallback_middleware()
    second = build_fallback_middleware()

    assert first is not second
