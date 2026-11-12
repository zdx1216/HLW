"""HLW 长期记忆流水线。

负责协调：
1. MemoryExtractor：提取候选记忆。
2. MemoryValidator：校验候选记忆。
3. MemoryDeduplicator：去重与冲突处理。
4. 发生记忆冲突时创建待确认记录。
5. 确认与拒绝操作由 MemoryConfirmationService 负责。

本模块不直接创建数据库连接。
数据库资源由外部统一管理。
"""

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import (
    MemoryPipelineResult,
    PendingMemoryReference,
)
from hlw.memory.validator import MemoryValidator


class MemoryPipeline:
    """统一协调长期记忆处理流程。"""

    def __init__(
        self,
        extractor: MemoryExtractor,
        validator: MemoryValidator,
        deduplicator: MemoryDeduplicator,
        *,
        pending_manager: PendingMemoryManager | None = None,
    ) -> None:
        """通过依赖注入获得长期记忆处理组件。

        pending_manager 可选，兼容已有 Pipeline 调用。
        提供时，冲突会创建待确认记录。
        """

        self._extractor = extractor
        self._validator = validator
        self._deduplicator = deduplicator
        self._pending_manager = pending_manager

    async def run(
        self,
        user_id: str,
        user_message: str,
    ) -> MemoryPipelineResult:
        """执行一次完整的长期记忆处理流程。"""

        # STEP 1：检查用户 ID
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")

        # STEP 2：检查用户消息
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message 不能为空")

        # STEP 3：调用 Extractor 提取候选记忆
        extraction = await self._extractor.extract(user_message)

        # STEP 4：调用 Validator 校验候选记忆
        validation = self._validator.validate(
            extraction,
            user_message,
        )

        # STEP 5：逐条处理已经通过验证的候选记忆。
        deduplication_results = []
        pending_updates = []
        for candidate in validation.accepted:
            # 5.1 判断新记忆、重复记忆或冲突记忆。
            result = await self._deduplicator.process(
                user_id=user_id,
                candidate=candidate,
            )

            deduplication_results.append(result)

            # 5.2 没有配置 PendingManager 时，
            # 保持原有 Pipeline 行为。
            if self._pending_manager is None:
                continue

            # 5.3 只有冲突记忆需要用户确认。
            if result.status != "conflict":
                continue

            # 5.4 冲突必须附带原始记忆快照。
            # 快照缺失时拒绝创建待确认记录。
            if result.existing_snapshot is None:
                raise RuntimeError(
                    "记忆冲突缺少原始快照，"
                    "不能安全创建待确认记录"
                )

            # 5.5 保存待确认记录。
            pending_record = (
                await self._pending_manager.create_pending(
                    user_id=user_id,
                    candidate=candidate,
                    existing_snapshot=result.existing_snapshot,
                )
            )

            # 5.6 返回待确认记录的引用信息。
            pending_updates.append(
                PendingMemoryReference(
                    pending_id=pending_record["pending_id"],
                    key=candidate.key,
                )
            )

        # STEP 6：返回流水线处理结果。
        return MemoryPipelineResult(
            extraction=extraction,
            validation=validation,
            deduplication_results=deduplication_results,
            pending_updates=pending_updates,
        )