
"""HLW 统一日志配置。"""

import logging
import os
from logging.config import dictConfig
from pathlib import Path

from hlw.core.config import PROJECT_ROOT

VALID_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


def setup_logging(
    level: str | None = None,
    log_dir: Path | None = None,
) -> None:
    """在应用启动时统一配置日志系统。"""

    log_level = (
        level or os.getenv("HLW_LOG_LEVEL", "INFO")
    ).upper()

    if log_level not in VALID_LEVELS:
        raise ValueError(
            f"不支持的日志级别：{log_level}"
        )

    directory = (
        Path(log_dir)
        if log_dir is not None
        else PROJECT_ROOT / "logs"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s | "
                        "%(levelname)-8s | "
                        "%(name)s | "
                        "%(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": (
                        "logging.handlers.RotatingFileHandler"
                    ),
                    "level": log_level,
                    "formatter": "standard",
                    "filename": str(directory / "hlw.log"),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                    "delay": True,
                },
            },
            "loggers": {
                "httpx": {
                    "level": "WARNING",
                    "propagate": True,
                },
                "httpcore": {
                    "level": "WARNING",
                    "propagate": True,
                },
            },
            "root": {
                "level": log_level,
                "handlers": [
                    "console",
                    "file",
                ],
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块专属 Logger。"""

    return logging.getLogger(name)
