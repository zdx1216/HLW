
"""HLW 自动长期记忆加载 PostgreSQL 集成测试。"""

import asyncio
import sys
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from hlw.agent.memory_context_middleware import MemoryContextMiddleware
from hlw.core.config import settings
from hlw.memory.manager import MemoryManager
from hlw.memory.store_postgres import open_postgres_store


class FakeRequest:
    """模拟 LangChain ModelRequest 所需接口。"""

    def __init__(self, messages, system_message, runtime):
        self.messages = messages
        self.system_message = system_message
        self.runtime = runtime

    def override(self, **kwargs):
        return FakeRequest(
            messages=kwargs.get("messages", self.messages),
            system_message=kwargs.get(
                "system_message",
                self.system_message,
            ),
            runtime=kwargs.get("runtime", self.runtime),
        )


async def main():
    run_id = uuid4().hex[:10]
    user_id = settings.user_id

    key = f"auto_load_test_{run_id}"
    marker = f"HLW_AUTO_LOAD_{run_id.upper()}"

    saved = False

    async with open_postgres_store() as store:
        manager = MemoryManager(store)

        try:
            # STEP 1：写入真实 PostgreSQL 记忆
            await manager.save_memory(
                user_id=user_id,
                key=key,
                content=(
                    "HLW 自动记忆加载集成测试，"
                    f"唯一识别标记是 {marker}。"
                ),
                category="fact",
            )

            saved = True
            print("STEP 1: PostgreSQL memory saved PASS")

            # STEP 2：构造模型请求
            request = FakeRequest(
                messages=[
                    HumanMessage(
                        content="我之前保存的自动记忆加载测试标记是什么？"
                    )
                ],
                system_message=SystemMessage(
                    content="你是 HLW 助手，请准确回答用户问题。"
                ),
                runtime=SimpleNamespace(store=store),
            )

            # STEP 3：执行真正的 Memory Middleware
            captured = {}

            async def fake_model_handler(enriched_request):
                captured["system_message"] = (
                    enriched_request.system_message.text
                )
                return "MODEL_HANDLER_REACHED"

            middleware = MemoryContextMiddleware()

            result = await middleware.awrap_model_call(
                request,
                fake_model_handler,
            )

            print("STEP 2: Middleware execution PASS")

            # STEP 4：验证记忆已经注入 SystemMessage
            injected_message = captured["system_message"]

            assert marker in injected_message, (
                "自动加载失败：SystemMessage 中没有找到测试标记"
            )

            assert "HLW 助手" in injected_message

            print("STEP 3: PostgreSQL memory injection PASS")

            # STEP 5：确认原请求没有被直接修改
            assert marker not in request.system_message.text

            print("STEP 4: Original request preserved PASS")

            assert result == "MODEL_HANDLER_REACHED"

            print("=" * 55)
            print("RESULT: Automatic Memory Loading PASS")
            print("=" * 55)

        finally:
            # STEP 6：清理测试数据
            if saved:
                await manager.delete_memory(
                    user_id=user_id,
                    key=key,
                )

                print("CLEANUP: Test memory deleted PASS")


def run():
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
