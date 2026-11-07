"""HLW 用户身份标识校验模块。

第一版采用单用户模式。
本模块负责校验和规范化 user_id，
为后续 Agent、Memory 和 Store 提供统一的身份标识。
"""

import re

# 用户 ID 的合法格式：
# 1. 长度为 1–64 个字符
# 2. 只能包含英文字母、数字、下划线和连字符
# 3. 第一个字符必须是字母或数字
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def normalize_user_id(user_id: str) -> str:
    """校验并规范化用户 ID。

    Args:
        user_id: 待校验的用户身份标识。

    Returns:
        str: 去除首尾空格后的合法用户 ID。

    Raises:
        ValueError: 用户 ID 类型或格式不合法。
    """

    # 第一步：检查数据类型
    # 这里统一抛 ValueError（而非 TypeError），
    # 与本函数文档约定、以及 AgentService.run 的入参校验保持一致。
    if not isinstance(user_id, str):
        raise ValueError("user_id 必须是字符串")  # noqa: TRY004

    # 第二步：去除首尾空格
    normalized = user_id.strip()

    # 第三步：检查 ID 格式
    if not _USER_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "user_id 必须为 1–64 位，只能包含字母、数字、"
            "下划线和连字符，且必须以字母或数字开头"
        )

    # 第四步：返回规范化后的用户 ID
    return normalized
