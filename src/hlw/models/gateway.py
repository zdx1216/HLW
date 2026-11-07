"""HLW Model Gateway：模型注册和客户端创建。"""

from typing import Literal

from langchain_openai import ChatOpenAI

from hlw.core.config import settings

ModelRole = Literal["primary", "fallback"]


def registered_models() -> dict[str, str]:
    """返回配置的模型注册表，不代表运行健康状态。"""

    models = {
        "primary": settings.chat_model,
        "fallback": settings.fallback_model,
    }

    if not all(models.values()):
        raise ValueError("模型名称不能为空")

    if models["primary"] == models["fallback"]:
        raise ValueError("主模型和备用模型不能相同")

    return models


def build_model(role: ModelRole) -> ChatOpenAI:
    """根据模型角色创建 Ollama 模型客户端。"""

    models = registered_models()

    if role not in models:
        raise ValueError(f"未知模型角色：{role}")

    if settings.model_timeout_seconds <= 0:
        raise ValueError("模型超时必须大于零")

        
    return ChatOpenAI(
        model=models[role],
        base_url=settings.ollama_base_url,
        api_key="ollama",
        temperature=0,
        timeout=settings.model_timeout_seconds,
        max_retries=0,
        stream_usage=True,
    )

