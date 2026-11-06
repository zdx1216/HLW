
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 2. 加载环境变量
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 3. 定义统一配置数据结构
# ============================================================

@dataclass(frozen=True)
class AppSettings:
    """HLW 的统一应用配置。"""

    # 运行环境
    environment: str

    # 主模型
    chat_model: str

    # 聊天模型 Ollama 地址
    ollama_base_url: str

    # 备用模型
    fallback_model: str

    # 模型调用超时时间
    model_timeout_seconds: float

    # Embedding 模型名称
    embedding_model: str

    # Embedding Ollama 服务地址
    embedding_base_url: str

    # Embedding 向量维度
    embedding_dimensions: int

    # 当前用户身份
    user_id: str


# ============================================================
# 4. 加载配置
# ============================================================

def load_settings() -> AppSettings:
    """从环境变量读取配置并构建 AppSettings。"""

    # --------------------------------------------------------
    # 4.1 Embedding 配置
    # --------------------------------------------------------

    embedding_model = os.getenv(
        "HLW_EMBEDDING_MODEL",
        "bge-m3:latest",
    ).strip()

    embedding_base_url = os.getenv(
        "HLW_OLLAMA_EMBEDDING_BASE_URL",
        "http://127.0.0.1:11434",
    ).strip().rstrip("/")

    try:
        embedding_dimensions = int(
            os.getenv(
                "HLW_EMBEDDING_DIMENSIONS",
                "1024",
            )
        )
    except ValueError as exc:
        raise ValueError(
            "HLW_EMBEDDING_DIMENSIONS 必须是正整数"
        ) from exc

    if not embedding_model:
        raise ValueError(
            "HLW_EMBEDDING_MODEL 不能为空"
        )

    if not embedding_base_url:
        raise ValueError(
            "HLW_OLLAMA_EMBEDDING_BASE_URL 不能为空"
        )

    if embedding_dimensions < 1:
        raise ValueError(
            "HLW_EMBEDDING_DIMENSIONS 必须是正整数"
        )

    # --------------------------------------------------------
    # 4.2 构建统一配置
    # --------------------------------------------------------

    return AppSettings(

        environment=os.getenv(
            "HLW_ENV",
            "development",
        ),

        chat_model=os.getenv(
            "HLW_CHAT_MODEL",
            "qwen3:8b",
        ),

        ollama_base_url=os.getenv(
            "HLW_OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
        ),

        fallback_model=os.getenv(
            "HLW_FALLBACK_MODEL",
            "qwen3:14b",
        ),

        model_timeout_seconds=float(
            os.getenv(
                "HLW_MODEL_TIMEOUT_SECONDS",
                "120",
            )
        ),

        embedding_model=embedding_model,

        embedding_base_url=embedding_base_url,

        embedding_dimensions=embedding_dimensions,

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

settings = load_settings()
