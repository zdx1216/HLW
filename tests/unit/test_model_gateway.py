"""Model Gateway 单元测试。"""

import pytest

from hlw.core.config import settings
from hlw.models.gateway import build_model, registered_models


def test_model_registry():
    models = registered_models()

    assert models["primary"] == settings.chat_model
    assert models["fallback"] == settings.fallback_model
    assert models["primary"] != models["fallback"]


def test_primary_model():
    model = build_model("primary")

    assert model.model_name == settings.chat_model


def test_fallback_model():
    model = build_model("fallback")

    assert model.model_name == settings.fallback_model


def test_invalid_model_role():
    with pytest.raises(ValueError, match="未知模型角色"):
        build_model("unknown")
