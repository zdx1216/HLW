"""HLW 统一日志配置。

不同运行模式使用独立日志目录，
只管理 hlw 命名空间，不修改 Root Logger。
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from hlw.core.config import PROJECT_ROOT

LogRuntime = Literal["api", "langgraph_dev"]

VALID_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}

VALID_RUNTIMES = {
    "api",
    "langgraph_dev",
}

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

MANAGED_HANDLER_ATTR = "_hlw_managed"


def setup_logging(
    runtime: LogRuntime = "api",
    level: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """初始化指定运行模式的日志系统。"""

    if runtime not in VALID_RUNTIMES:
        raise ValueError(f"不支持的运行模式：{runtime}")

    log_level = (level or os.getenv("HLW_LOG_LEVEL", "INFO")).upper()

    if log_level not in VALID_LEVELS:
        raise ValueError(f"不支持的日志级别：{log_level}")

    directory = (
        Path(log_dir) if log_dir is not None else PROJECT_ROOT / "logs" / runtime
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (directory / "hlw.log").resolve()

    # 只管理 HLW 自己的 Logger
    logger = logging.getLogger("hlw")
    logger.setLevel(log_level)

    # 保留向 Root Logger 传播的能力，
    # 兼容 pytest caplog 和宿主运行环境。
    logger.propagate = True

    # 清理之前由 HLW 创建的 Handler，
    # 避免重复初始化产生重复日志。
    for handler in list(logger.handlers):
        if getattr(
            handler,
            MANAGED_HANDLER_ATTR,
            False,
        ):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件日志
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    setattr(
        file_handler,
        MANAGED_HANDLER_ATTR,
        True,
    )

    logger.addHandler(file_handler)

    # ========================================================
    # JSONL 结构化事件日志
    # ========================================================

    event_logger = logging.getLogger("hlw.events")

    event_logger.setLevel(logging.INFO)

    # 事件日志不向普通日志传播，确保 JSONL 保持纯 JSON 格式。
    event_logger.propagate = False

    # 清理旧 Handler，避免重复写入。
    for handler in list(event_logger.handlers):
        if getattr(
            handler,
            MANAGED_HANDLER_ATTR,
            False,
        ):
            event_logger.removeHandler(handler)
            handler.close()

    event_file_handler = RotatingFileHandler(
        filename=directory / "events.jsonl",
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
        delay=True,
    )

    event_file_handler.setLevel(logging.INFO)

    # JSONL 每一行必须是独立 JSON，不能添加普通日志前缀。
    event_file_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    setattr(
        event_file_handler,
        MANAGED_HANDLER_ATTR,
        True,
    )

    event_logger.addHandler(event_file_handler)

    # 标记当前运行环境。
    event_logger._hlw_runtime = runtime

    # 如果宿主没有配置 Root Handler，
    # HLW 自己提供控制台输出。
    if not logging.getLogger().handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        setattr(
            console_handler,
            MANAGED_HANDLER_ATTR,
            True,
        )

        logger.addHandler(console_handler)

    return log_file


def get_logger(name: str) -> logging.Logger:
    """获取模块专属 Logger。"""

    return logging.getLogger(name)
