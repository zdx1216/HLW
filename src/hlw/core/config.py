"""HLW 应用配置模块。

集中定义运行时配置的唯一来源：数值来自环境变量，
并通过 ``.env`` 文件提供本地开发时的默认值。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：本文件位于 src/hlw/core/config.py，
# 向上 3 级（core -> hlw -> src -> 项目根）即仓库根路径。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 在导入时加载项目根目录下的 .env；不覆盖已存在的环境变量，
# 因此容器/CI 中注入的环境变量优先级更高。
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AppSettings:
    """HLW 的统一应用配置。

    使用 ``frozen=True`` 使实例不可变，避免运行期被意外修改。
    """

    environment: str  # 运行环境标识，如 development / production
    chat_model: str  # 对话模型名称（默认走 Ollama）
    ollama_base_url: str  # Ollama 的 OpenAI 兼容接口地址
    fallback_model:str 
    model_timeout_seconds:float 

def load_settings() -> AppSettings:
    """从环境变量读取配置。

    所有字段都有默认值，因此在未配置任何环境变量时也可直接运行。
    """

    return AppSettings(
        environment=os.getenv(
            "HLW_ENV", "development"
        ),
        chat_model=os.getenv(
            "HLW_CHAT_MODEL", "qwen3:8b"
        ),
        ollama_base_url=os.getenv(
            "HLW_OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1"
        ),
        fallback_model=os.getenv("HLW_FALLBACK_MODEL", "qwen3:14b"),
        model_timeout_seconds=float(
    os.getenv(
        "HLW_MODEL_TIMEOUT_SECONDS",
        "120",
    )
),
    )


# 模块级单例：其他模块可直接 ``from hlw.core.config import settings``，
# 避免每次调用都重复读取环境变量。
settings = load_settings()