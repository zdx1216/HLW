"""HLW 日志配置测试。"""

import logging

import pytest

from hlw.core.logging_config import (
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def cleanup_logging():
    """清理测试创建的日志 Handler。"""

    yield

    logger = logging.getLogger("hlw")

    for handler in list(logger.handlers):
        if getattr(handler, "_hlw_managed", False):
            logger.removeHandler(handler)
            handler.close()


def flush_handlers():
    """确保日志写入文件。"""

    for handler in logging.getLogger("hlw").handlers:
        handler.flush()


def test_setup_logging_writes_to_file(tmp_path):
    log_file = setup_logging(
        runtime="api",
        log_dir=tmp_path / "api",
    )

    logger = get_logger("hlw.tests.logging")
    logger.info("api request recorded")

    flush_handlers()

    assert log_file.exists()
    assert "api request recorded" in log_file.read_text(encoding="utf-8")


def test_runtime_logs_are_separated(tmp_path):
    logger = get_logger("hlw.tests.logging")

    api_file = setup_logging(
        runtime="api",
        log_dir=tmp_path / "api",
    )

    logger.info("API_ONLY")
    flush_handlers()

    dev_file = setup_logging(
        runtime="langgraph_dev",
        log_dir=tmp_path / "langgraph_dev",
    )

    logger.info("DEV_ONLY")
    flush_handlers()

    api_content = api_file.read_text(encoding="utf-8")
    dev_content = dev_file.read_text(encoding="utf-8")

    assert "API_ONLY" in api_content
    assert "DEV_ONLY" not in api_content

    assert "DEV_ONLY" in dev_content
    assert "API_ONLY" not in dev_content


def test_invalid_runtime_rejected(tmp_path):
    with pytest.raises(ValueError):
        setup_logging(
            runtime="invalid",
            log_dir=tmp_path,
        )


def test_invalid_log_level_rejected(tmp_path):
    with pytest.raises(ValueError):
        setup_logging(
            runtime="api",
            level="TRACE",
            log_dir=tmp_path,
        )
