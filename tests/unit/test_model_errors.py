
import httpx
import pytest

from hlw.models.errors import (
    ModelErrorKind,
    classify_model_error,
)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ConnectionError("connection refused"),
            ModelErrorKind.CONNECTION,
        ),
        (
            httpx.ConnectError("connection failed"),
            ModelErrorKind.CONNECTION,
        ),
        (
            TimeoutError("request timed out"),
            ModelErrorKind.TIMEOUT,
        ),
        (
            httpx.ReadTimeout("read timed out"),
            ModelErrorKind.TIMEOUT,
        ),
        (
            ValueError("invalid configuration"),
            ModelErrorKind.UNKNOWN,
        ),
    ],
)
def test_error_classification(error, expected):
    assert classify_model_error(error) == expected
