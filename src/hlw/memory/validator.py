
"""HLW 长期记忆校验器。

负责检查候选记忆是否满足基本保存规则。

不调用 LLM，不连接数据库。
"""

import re

from hlw.memory.schemas import (
    MemoryExtractionResult,
    MemoryRejection,
    MemoryValidationResult,
)

# 检查常见凭证格式。
# 这里只能识别部分常见秘密格式，不代表完整的敏感信息检测。
SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|"
        r"refresh[_-]?token|password|passwd|secret)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(
        r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"
    ),
    re.compile(
        r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"
    ),
]


def contains_possible_secret(text: str) -> bool:
    """检查文本是否包含疑似凭证。"""

    return any(
        pattern.search(text)
        for pattern in SECRET_PATTERNS
    )


class MemoryValidator:
    """负责候选记忆的最终规则校验。"""

    def validate(
        self,
        extraction: MemoryExtractionResult,
        user_message: str,
    ) -> MemoryValidationResult:
        """校验候选记忆并分别返回通过和拒绝的结果。"""

        if (
            not isinstance(user_message, str)
            or not user_message.strip()
        ):
            raise ValueError("user_message 不能为空")

        accepted = []
        rejected = []

        # 只记录已经通过检查的 key。
        seen_keys: set[str] = set()

        for candidate in extraction.candidates:

            # 检查 1：证据必须来自用户原文。
            if candidate.evidence not in user_message:
                rejected.append(
                    MemoryRejection(
                        key=candidate.key,
                        reason="evidence_not_found",
                    )
                )
                continue

            # 检查 2：候选内容和证据不能包含疑似秘密。
            if (
                contains_possible_secret(candidate.content)
                or contains_possible_secret(candidate.evidence)
            ):
                rejected.append(
                    MemoryRejection(
                        key=candidate.key,
                        reason="possible_secret",
                    )
                )
                continue

            # 检查 3：同一批次不能出现重复 key。
            if candidate.key in seen_keys:
                rejected.append(
                    MemoryRejection(
                        key=candidate.key,
                        reason="duplicate_key",
                    )
                )
                continue

            # 所有检查通过，才允许进入 accepted。
            seen_keys.add(candidate.key)
            accepted.append(candidate)

        return MemoryValidationResult(
            accepted=accepted,
            rejected=rejected,
        )
