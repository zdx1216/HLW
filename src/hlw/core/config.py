"""HLW 应用配置模块。

集中定义运行时配置的唯一来源。
通过环境变量和 .env 文件管理应用配置。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from hlw.core.identity import normalize_user_id

# ============================================================
# 1. 项目根目录
# ============================================================

# 当前文件位于：
# src/hlw/core/config.py
#
# parents[3] 对应 HLW 项目根目录。

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 2. 加载环境变量
# ============================================================

# 加载项目根目录下的 .env 文件。
# 默认不覆盖系统中已经存在的环境变量。

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 3. 定义统一配置数据结构
# ============================================================

@dataclass(frozen=True)
class AppSettings:
    """HLW 的统一应用配置。

    frozen=True 表示配置对象创建后不可修改。
    """

    # 运行环境
    environment: str

    # 主模型
    chat_model: str

    # Ollama 服务地址
    ollama_base_url: str

    # 备用模型
    fallback_model: str

    # 模型调用超时时间
    model_timeout_seconds: float

    # 当前用户身份
    user_id: str


# ============================================================
# 4. 加载配置
# ============================================================

def load_settings() -> AppSettings:
    """从环境变量读取配置并构建 AppSettings。

    如果没有配置 HLW_USER_ID，
    默认使用 local_user。
    """

    return AppSettings(

        # 运行环境
        environment=os.getenv(
            "HLW_ENV",
            "development",
        ),

        # 主模型
        chat_model=os.getenv(
            "HLW_CHAT_MODEL",
            "qwen3:8b",
        ),

        # Ollama 地址
        ollama_base_url=os.getenv(
            "HLW_OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
        ),

        # 备用模型
        fallback_model=os.getenv(
            "HLW_FALLBACK_MODEL",
            "qwen3:14b",
        ),

        # 模型超时时间
        model_timeout_seconds=float(
            os.getenv(
                "HLW_MODEL_TIMEOUT_SECONDS",
                "120",
            )
        ),

        # 用户身份
        # 读取环境变量后，统一进行合法性校验。
        user_id=normalize_user_id(
            os.getenv(
                "HLW_USER_ID",
                "local_user",
            )
        ),
    )


# ============================================================
# 5. 创建全局配置实例
# ============================================================

# 项目其他模块可以直接导入 settings，
# 不需要重复读取 .env 或创建配置对象。

settings = load_settings()
