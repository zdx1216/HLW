"""HLW 用户身份标识独立单元测试。"""

import pytest

from hlw.core.identity import normalize_user_id

# ============================================================
# 1. 合法用户 ID 测试
# ============================================================


@pytest.mark.parametrize(
    ("raw_user_id", "expected"),
    [
        ("local_user", "local_user"),
        (" local_user ", "local_user"),
        ("\tuser-001\n", "user-001"),
        ("A", "A"),
        ("a" * 64, "a" * 64),
    ],
)
def test_normalize_valid_user_id(
    raw_user_id: str,
    expected: str,
) -> None:
    """验证合法用户 ID 能否正确规范化。"""

    result = normalize_user_id(raw_user_id)

    assert result == expected


# ============================================================
# 2. 非法用户 ID 测试
# ============================================================


@pytest.mark.parametrize(
    "invalid_user_id",
    [
        "",
        " ",
        "../admin",
        "user/001",
        "user 001",
        "user@001",
        "-user",
        "_user",
        "a" * 65,
        None,
        123,
    ],
)
def test_reject_invalid_user_id(
    invalid_user_id,
) -> None:
    """验证非法用户 ID 是否被拒绝。"""

    with pytest.raises(ValueError):
        normalize_user_id(invalid_user_id)
