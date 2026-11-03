
"""P0-02C：受控重试与模型回退的真实集成测试。"""

import asyncio

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from hlw.core.config import settings
from hlw.core.logging_config import setup_logging
from hlw.models.gateway import build_model
from hlw.models.resilience import ControlledFallbackMiddleware

setup_logging()


# 1. 故意创建无法连接的主模型
broken_primary = ChatOpenAI(
    model=settings.chat_model,
    base_url="http://127.0.0.1:1/v1",
    api_key="ollama",
    temperature=0,
    timeout=3,
    max_retries=0,
)


# 2. 备用模型使用真实 Ollama 配置
working_fallback = build_model("fallback")


# 3. 使用我们自己开发的 Middleware
resilience_middleware = ControlledFallbackMiddleware(
    fallback_model=working_fallback,
    max_retries=1,
    initial_delay=0.5,
)


# 4. 创建独立的测试图
probe_graph = create_deep_agent(
    model=broken_primary,
    backend=StateBackend(),
    middleware=[resilience_middleware],
    system_prompt=(
        "你是 HLW 模型容错测试助手。"
        "请使用中文简短回答。"
    ),
)


async def main():
    print("\n===== HLW P0-02C 集成测试 =====")

    print("Primary:", settings.chat_model)
    print("Fallback:", settings.fallback_model)
    print("注入故障: 127.0.0.1:1")
    print("预期主模型请求次数: 2")
    print("预期备用模型请求次数: 1")

    result = await probe_graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请回复：HLW受控回退测试成功。"
                    ),
                }
            ]
        },
        config={
            "recursion_limit": 15,
        },
    )

    final_message = result["messages"][-1]

    assert isinstance(final_message, AIMessage)
    assert final_message.content

    actual_model = final_message.response_metadata.get(
        "model_name"
    )

    print("\n实际响应模型:", actual_model)
    print("最终回复:", final_message.content)

    assert actual_model == settings.fallback_model, (
        f"预期备用模型 {settings.fallback_model}，"
        f"实际响应模型 {actual_model}"
    )

    print("\nP0-02C 真实回退验证通过！")


if __name__ == "__main__":
    asyncio.run(main())
