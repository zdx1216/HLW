
"""HLW 正式模型回退配置测试。"""

from hlw.agent.graph import fallback_middleware
from hlw.core.config import settings
from hlw.models.resilience import ControlledFallbackMiddleware


def test_fallback_middleware_created():
    """检查正式 Agent 使用受控回退中间件。"""

    assert isinstance(
        fallback_middleware,
        ControlledFallbackMiddleware,
    )


def test_fallback_model_config():
    """检查备用模型与环境配置一致。"""

    fallback = fallback_middleware.fallback_model

    assert fallback.model_name == settings.fallback_model


def test_retry_config():
    """检查重试次数及退避配置。"""

    assert fallback_middleware.max_retries == 2
    assert fallback_middleware.initial_delay >= 0
