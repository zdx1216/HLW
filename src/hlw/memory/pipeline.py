
"""HLW 长期记忆流水线。

负责协调：
1. MemoryExtractor：提取候选记忆。
2. MemoryValidator：校验候选记忆。
3. MemoryDeduplicator：去重与冲突处理。
4. 用户授权更新：执行指定记忆的更新。

本模块不直接创建数据库连接。
数据库资源由外部统一管理。
"""

from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.schemas import (
    MemoryCandidate,
    MemoryDeduplicationResult,
    MemoryPipelineResult,
)
from hlw.memory.validator import MemoryValidator


class MemoryPipeline:
    """统一协调长期记忆处理流程。"""

    def __init__(
        self,
        extractor: MemoryExtractor,
        validator: MemoryValidator,
        deduplicator: MemoryDeduplicator,
    ) -> None:
        """通过依赖注入获得所需组件。"""

        self._extractor = extractor
        self._validator = validator
        self._deduplicator = deduplicator

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
        if (
            not isinstance(user_message, str)
            or not user_message.strip()
        ):
            raise ValueError("user_message 不能为空")

        # STEP 3：调用 Extractor 提取候选记忆
        extraction = await self._extractor.extract(
            user_message
        )

        # STEP 4：调用 Validator 校验候选记忆
        validation = self._validator.validate(
            extraction,
            user_message,
        )

        # STEP 5：逐条处理通过校验的候选记忆
        deduplication_results = []

        for candidate in validation.accepted:

            result = await self._deduplicator.process(
                user_id=user_id,
                candidate=candidate,
            )

            deduplication_results.append(result)

        # STEP 6：统一返回整个 Pipeline 的执行结果
        return MemoryPipelineResult(
            extraction=extraction,
            validation=validation,
            deduplication_results=deduplication_results,
        )

    async def confirm_update(
        self,
        user_id: str,
        candidate: MemoryCandidate,
    ) -> MemoryDeduplicationResult:
        """用户明确授权后，更新指定的长期记忆。

        注意：
            candidate 应当来自之前经过 Validator 校验、
            并由服务端保留的候选记忆。

            本方法不重新调用 Extractor，
            也不重新执行整条 Pipeline。
        """

        # STEP 1：检查用户 ID
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id 不能为空")

        # STEP 2：检查候选记忆
        if not isinstance(candidate, MemoryCandidate):
            raise TypeError(
                "candidate 必须是 MemoryCandidate 类型"
            )

        # STEP 3：调用 Deduplicator 执行授权更新
        return await self._deduplicator.confirm_update(
            user_id=user_id,
            candidate=candidate,
        )
