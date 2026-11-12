
"""HLW 自动长期记忆捕获 Middleware。

负责：
1. 识别当前用户的新消息。
2. 调用 MemoryExtractor 提取候选记忆。
3. 调用 MemoryValidator 校验候选记忆。
4. 调用 MemoryDeduplicator 判断新增、重复和冲突。
5. 将新记忆写入 PostgreSQL Store。
6. 将冲突记忆写入 PendingMemoryManager。

安全原则：
- user_id 使用服务端可信配置。
- 不允许模型直接操作数据库。
- 已有记忆发生冲突时不自动覆盖。
- 数据库或模型异常不阻断正常对话。
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)

from hlw.core.config import settings
from hlw.core.logging_config import get_logger
from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import memory_execution_locks
from hlw.memory.extractor import MemoryExtractor
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.pipeline import MemoryPipeline
from hlw.memory.validator import MemoryValidator

logger = get_logger(__name__)


class MemoryCaptureMiddleware(AgentMiddleware):
    """自动识别并保存用户长期记忆。"""

    def __init__(
        self,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        """初始化记忆捕获组件。

        extractor 允许测试时注入模拟提取器。
        """
        self._extractor = extractor
        self._validator = MemoryValidator()

        # 与确认服务共享进程内记忆锁。
        self._execution_locks = memory_execution_locks

    @staticmethod
    def _extract_latest_user_message(
        request: ModelRequest[Any],
    ) -> str | None:
        """只提取模型请求中最后一条用户消息。"""

        messages = getattr(request, "messages", None)

        if not messages:
            return None

        message = messages[-1]

        if getattr(message, "type", None) != "human":
            return None

        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content.strip() or None

        # 兼容多模态消息中的文本内容。
        if isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)

                elif (
                    isinstance(block, dict)
                    and block.get("type") in {"text", "input_text"}
                ):
                    text = block.get("text")

                    if isinstance(text, str):
                        text_parts.append(text)

            combined = "\n".join(text_parts).strip()
            return combined or None

        return None

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelResponse[Any]:
        """在模型调用之前执行自动长期记忆捕获。"""

        # STEP 1：提取当前用户消息。
        user_message = self._extract_latest_user_message(
            request
        )

        if not user_message:
            return await handler(request)

        # STEP 2：获取 Runtime Store。
        runtime = request.runtime
        store = getattr(runtime, "store", None)

        if store is None:
            logger.debug(
                "Memory capture skipped: Runtime Store unavailable"
            )
            return await handler(request)

        try:
            # STEP 3：初始化记忆提取器。
            if self._extractor is None:
                self._extractor = MemoryExtractor()

            # STEP 4：创建正式记忆管理器。
            manager = MemoryManager(store)

            # STEP 5：创建待确认记忆管理器。
            pending_manager = PendingMemoryManager(store)

            # STEP 6：创建记忆去重器。
            deduplicator = MemoryDeduplicator(
                manager=manager,
                execution_locks=self._execution_locks,
            )

            # STEP 7：组装完整记忆流水线。
            pipeline = MemoryPipeline(
                extractor=self._extractor,
                validator=self._validator,
                deduplicator=deduplicator,
                pending_manager=pending_manager,
            )

            # STEP 8：运行提取、校验、去重与 Pending 创建。
            result = await pipeline.run(
                user_id=settings.user_id,
                user_message=user_message,
            )

            # STEP 9：记录处理结果。
            statuses = [
                item.status
                for item in result.deduplication_results
            ]

            logger.info(
                "Automatic memory capture completed "
                "candidates=%d accepted=%d "
                "statuses=%s pending_count=%d",
                len(result.extraction.candidates),
                len(result.validation.accepted),
                statuses,
                len(result.pending_updates),
            )

        except Exception:
            # 记忆子系统故障不能阻断正常对话。
            logger.exception(
                "Automatic memory capture failed"
            )

        # STEP 10：继续正常 Agent 调用。
        return await handler(request)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]],
            ModelResponse[Any],
        ],
    ) -> ModelResponse[Any]:
        """同步模式暂不执行异步记忆捕获。"""

        logger.debug(
            "Automatic memory capture requires async invocation"
        )

        return handler(request)
