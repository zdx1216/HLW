
"""验证 HLW Agent Service 实际执行。"""

import asyncio

from hlw.agent.graph import graph
from hlw.agent.service import AgentService
from hlw.core.logging_config import setup_logging

# 冒烟专用的服务实例：使用不带 Checkpointer 的图，
# 只验证「请求 → 执行 → 回答」链路能跑通。
# 注意它没有会话持久化能力（thread_id 不会跨调用记忆），
# 会话持久化由 check_service_postgres.py / check_postgres_checkpoint.py 验证。
agent_service = AgentService(
    agent_graph=graph,
    max_concurrent=2,
)


async def main():
    result = await agent_service.run(
        message=(
            "请调用 get_project_status，"
            "查询 HLW 的当前版本和开发阶段。"
        ),
        thread_id="hlw-service-test-001",
    )

    final_message = result["messages"][-1]

    print("Agent 回答：")
    print(final_message.content)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
