"""P0-02B：真实 Ollama 模型 Fallback 故障注入。"""

import asyncio

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from hlw.core.config import settings
from hlw.models.gateway import build_model
from hlw.tools.project_tools import get_project_status

# 故意使主模型无法连接
# 不修改正式项目的 .env
broken_primary = ChatOpenAI(
    model=settings.chat_model,
    base_url="http://127.0.0.1:1/v1",
    api_key="ollama",
    temperature=0,
    timeout=3,
    max_retries=0,
)

# 备用模型连接真实 Ollama
working_fallback = build_model("fallback")


# 建立独立测试图
graph = create_deep_agent(
    model=broken_primary,
    backend=StateBackend(),
    tools=[get_project_status],
    middleware=[ModelFallbackMiddleware(working_fallback)],
    system_prompt=("你是 HLW 模型容错测试助手。请使用中文简短回答用户。"),
)


async def main():
    print("===== HLW P0-02B Fallback 实验 =====")

    print("主模型：", settings.chat_model)
    print("主模型地址：127.0.0.1:1（故意不可用）")
    print("备用模型：", settings.fallback_model)

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请简短回复：HLW模型回退测试成功。",
                }
            ]
        },
        config={
            "recursion_limit": 15,
        },
    )

    last = result["messages"][-1]

    assert isinstance(last, AIMessage)
    assert last.content

    print("\n最终回答：", last.content)
    print("响应元数据：", last.response_metadata)
    print("\nFallback 集成测试通过！")


if __name__ == "__main__":
    asyncio.run(main())
