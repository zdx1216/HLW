"""HLW 长期记忆提取的数据结构。

本模块只定义数据格式，不连接数据库，
也不负责调用 LLM。

MemoryCandidate 表示一条待审核的候选记忆，
不代表这条记忆已经被允许保存。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryCandidate(BaseModel):
    """从用户对话中提取的一条候选长期记忆。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    # 记忆的稳定标识。
    # 使用小写字母、数字和下划线。
    key: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    # 记忆分类。
    category: Literal[
        "preference",
        "project",
        "workflow",
        "fact",
    ]

    # 提炼后的记忆内容。
    content: str = Field(
        min_length=1,
        max_length=500,
    )

    # 用户原文中的证据。
    # 后续由程序验证证据是否确实存在于原文。
    evidence: str = Field(
        min_length=1,
        max_length=500,
    )


class MemoryExtractionResult(BaseModel):
    """一次提取操作的结构化结果。"""

    model_config = ConfigDict(extra="forbid")

    # 没有值得保存的信息时，返回空列表。
    # 单次最多提取 5 条候选记忆。
    candidates: list[MemoryCandidate] = Field(
        default_factory=list,
        max_length=5,
    )


class MemoryRejection(BaseModel):
    """记录一条候选记忆被拒绝的原因。"""

    key: str

    reason: Literal[
        "evidence_not_found",
        "possible_secret",
        "duplicate_key",
    ]


class MemoryValidationResult(BaseModel):
    """记忆校验的最终结果。"""

    accepted: list[MemoryCandidate] = Field(default_factory=list)

    rejected: list[MemoryRejection] = Field(default_factory=list)


class MemoryDeduplicationResult(BaseModel):
    """记忆去重处理结果。"""

    model_config = ConfigDict(extra="forbid")
    key: str
    status: Literal[
        "created",  # 新记忆
        "unchanged",  # 完全重复
        "conflict",  # 内容不同但没有授权，更新后再次提交相同记忆
        "updated",  # 明确授权更新
    ]
    # 对处理结果的简要说明
    message: str
    # 如果存在旧记忆，记录其原始内容
    existing_content: str | None = None
        # 发生冲突时保存原始记忆快照，供 PendingManager 使用。
    existing_snapshot: dict[str, str] | None = None


class PendingMemoryReference(BaseModel):
    """流水线创建的待确认记忆引用。"""

    model_config = ConfigDict(extra="forbid")

    pending_id: str
    key: str
class MemoryPipelineResult(BaseModel):
    """完整的长期记忆流水线执行结果。"""

    model_config = ConfigDict(extra="forbid")

    # Extractor 的提取结果
    extraction: MemoryExtractionResult

    # Validator 的校验结果
    validation: MemoryValidationResult

    # Deduplicator 对每条合格记忆的处理结果
    deduplication_results: list[MemoryDeduplicationResult] = Field(default_factory=list)
    # 本次流水线创建的待确认记录。
    pending_updates: list[PendingMemoryReference] = Field(
        default_factory=list
    )