"""HLW Embedding 配置单元测试。"""

import pytest

from hlw.core.config import load_settings


def test_embedding_settings_defaults(monkeypatch):
    """测试 Embedding 配置默认值。"""

    monkeypatch.delenv(
        "HLW_EMBEDDING_MODEL",
        raising=False,
    )

    monkeypatch.delenv(
        "HLW_OLLAMA_EMBEDDING_BASE_URL",
        raising=False,
    )

    monkeypatch.delenv(
        "HLW_EMBEDDING_DIMENSIONS",
        raising=False,
    )

    settings = load_settings()

    assert settings.embedding_model == "bge-m3:latest"

    assert settings.embedding_base_url == "http://127.0.0.1:11434"

    assert settings.embedding_dimensions == 1024


def test_embedding_settings_custom_values(monkeypatch):
    """测试自定义 Embedding 配置。"""

    monkeypatch.setenv(
        "HLW_EMBEDDING_MODEL",
        "bge-m3:latest",
    )

    monkeypatch.setenv(
        "HLW_OLLAMA_EMBEDDING_BASE_URL",
        "http://localhost:11434/",
    )

    monkeypatch.setenv(
        "HLW_EMBEDDING_DIMENSIONS",
        "1024",
    )

    settings = load_settings()

    assert settings.embedding_model == "bge-m3:latest"

    assert settings.embedding_base_url == "http://localhost:11434"

    assert settings.embedding_dimensions == 1024


@pytest.mark.parametrize(
    "invalid_dimensions",
    [
        "0",
        "-1",
        "abc",
        "1.5",
    ],
)
def test_invalid_embedding_dimensions_rejected(
    monkeypatch,
    invalid_dimensions,
):
    """测试非法向量维度必须被拒绝。"""

    monkeypatch.setenv(
        "HLW_EMBEDDING_DIMENSIONS",
        invalid_dimensions,
    )

    with pytest.raises(ValueError):
        load_settings()


def test_empty_embedding_model_rejected(monkeypatch):
    """测试空模型名称必须被拒绝。"""

    monkeypatch.setenv(
        "HLW_EMBEDDING_MODEL",
        "   ",
    )

    with pytest.raises(ValueError):
        load_settings()


def test_empty_embedding_url_rejected(monkeypatch):
    """测试空服务地址必须被拒绝。"""

    monkeypatch.setenv(
        "HLW_OLLAMA_EMBEDDING_BASE_URL",
        "   ",
    )

    with pytest.raises(ValueError):
        load_settings()
