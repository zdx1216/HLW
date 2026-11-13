
"""HLW 待确认记忆 HTTP API 单元测试。

测试使用内存 Store 和独立 FastAPI 应用，
不连接真实 PostgreSQL。
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.store.memory import InMemoryStore

from hlw.api.app import app as production_app
from hlw.api.memory_routes import router
from hlw.core.config import settings
from hlw.memory.confirmation_service import MemoryConfirmationService
from hlw.memory.deduplicator import MemoryDeduplicator
from hlw.memory.execution_locks import MemoryExecutionLocks
from hlw.memory.manager import MemoryManager
from hlw.memory.pending_manager import PendingMemoryManager
from hlw.memory.schemas import MemoryCandidate

TEST_API_KEY = "test-only-" + "a" * 40
ENDPOINT = "/api/v1/memories/pending"


def make_test_app() -> tuple[FastAPI, PendingMemoryManager]:
    """创建不依赖真实数据库的 API 测试应用。"""

    test_app = FastAPI()
    test_app.include_router(router)

    manager = PendingMemoryManager(InMemoryStore())
    test_app.state.pending_memory_manager = manager

    return test_app, manager


def auth_headers() -> dict[str, str]:
    return {"X-HLW-API-Key": TEST_API_KEY}


def create_pending_record(
    manager: PendingMemoryManager,
    user_id: str,
    content: str = "喜欢英文代码注释",
) -> dict:
    """为测试准备一条待确认记忆。"""

    candidate = MemoryCandidate(
        key="coding_style",
        category="preference",
        content=content,
        evidence=content,
    )

    snapshot = {
        "content": "喜欢中文代码注释",
        "category": "preference",
        "created_at": "2026-11-01T10:00:00+00:00",
        "updated_at": "2026-11-02T10:00:00+00:00",
    }

    return asyncio.run(
        manager.create_pending(
            user_id=user_id,
            candidate=candidate,
            existing_snapshot=snapshot,
        )
    )


def test_memory_api_requires_configuration(monkeypatch):
    """没有配置 API Key 时必须拒绝访问。"""

    monkeypatch.delenv(
        "HLW_MEMORY_API_KEY",
        raising=False,
    )

    test_app, _ = make_test_app()

    with TestClient(test_app) as client:
        response = client.get(
            ENDPOINT,
            headers=auth_headers(),
        )

    assert response.status_code == 503


def test_memory_api_rejects_missing_key(monkeypatch):
    """没有携带密钥的请求必须被拒绝。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, _ = make_test_app()

    with TestClient(test_app) as client:
        response = client.get(ENDPOINT)

    assert response.status_code == 401


def test_memory_api_rejects_wrong_key(monkeypatch):
    """错误 API Key 不得访问记忆。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, _ = make_test_app()

    with TestClient(test_app) as client:
        response = client.get(
            ENDPOINT,
            headers={
                "X-HLW-API-Key": "incorrect-key",
            },
        )

    assert response.status_code == 401


def test_list_pending_memories(monkeypatch):
    """合法请求应返回当前用户的待确认记录。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager = make_test_app()

    created = create_pending_record(
        manager,
        settings.user_id,
    )

    with TestClient(test_app) as client:
        response = client.get(
            ENDPOINT,
            headers=auth_headers(),
        )

    assert response.status_code == 200

    records = response.json()

    assert len(records) == 1
    assert records[0]["pending_id"] == created["pending_id"]
    assert records[0]["status"] == "pending"
    assert records[0]["candidate"]["key"] == "coding_style"
    assert (
        records[0]["existing_snapshot"]["content"]
        == "喜欢中文代码注释"
    )


def test_pending_api_uses_server_side_user_id(monkeypatch):
    """客户端不能读取其他用户的 Pending。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager = make_test_app()

    # 仅为其他用户创建待确认记录。
    create_pending_record(
        manager,
        "another_user",
    )

    with TestClient(test_app) as client:
        response = client.get(
            ENDPOINT,
            params={"user_id": "another_user"},
            headers=auth_headers(),
        )

    assert response.status_code == 200
    assert response.json() == []


def test_pending_api_validates_pagination(monkeypatch):
    """无效分页参数必须被拒绝。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, _ = make_test_app()

    with TestClient(test_app) as client:
        response = client.get(
            ENDPOINT,
            params={"limit": 0, "offset": -1},
            headers=auth_headers(),
        )

    assert response.status_code == 422


def test_pending_api_unavailable_store(monkeypatch):
    """未注入 PendingManager 时返回服务不可用。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app = FastAPI()
    test_app.include_router(router)

    with TestClient(test_app) as client:
        response = client.get(
            ENDPOINT,
            headers=auth_headers(),
        )

    assert response.status_code == 503


def test_memory_router_registered_in_production_app():
    """生产应用必须注册待确认记忆 GET 接口。"""

    # 使用 FastAPI 官方 OpenAPI 结果检查路由。
    # 不直接遍历 app.routes，因为新版本的
    # _IncludedRouter 对象不一定具有 path 属性。
    schema = production_app.openapi()

    assert ENDPOINT in schema["paths"]
    assert "get" in schema["paths"][ENDPOINT]


def make_confirmation_test_app():
    """构造具有完整确认服务的内存 API 测试环境。"""

    store = InMemoryStore()

    test_app = FastAPI()
    test_app.include_router(router)

    memory_manager = MemoryManager(store)
    pending_manager = PendingMemoryManager(store)

    execution_locks = MemoryExecutionLocks()

    deduplicator = MemoryDeduplicator(
        manager=memory_manager,
        execution_locks=execution_locks,
    )

    confirmation_service = MemoryConfirmationService(
        pending_manager=pending_manager,
        deduplicator=deduplicator,
        execution_locks=execution_locks,
    )

    test_app.state.pending_memory_manager = pending_manager
    test_app.state.memory_confirmation_service = (
        confirmation_service
    )

    return test_app, memory_manager, pending_manager


def create_conflicting_pending(
    memory_manager: MemoryManager,
    pending_manager: PendingMemoryManager,
    user_id: str,
):
    """创建正式旧记忆和一条存在冲突的 Pending。"""

    async def prepare():
        await memory_manager.save_memory(
            user_id=user_id,
            key="coding_style",
            content="喜欢中文代码注释",
            category="preference",
        )

        original = await memory_manager.get_memory(
            user_id,
            "coding_style",
        )

        assert original is not None

        snapshot = {
            field: original[field]
            for field in (
                "content",
                "category",
                "created_at",
                "updated_at",
            )
        }

        candidate = MemoryCandidate(
            key="coding_style",
            category="preference",
            content="喜欢英文代码注释",
            evidence="英文代码注释",
        )

        return await pending_manager.create_pending(
            user_id=user_id,
            candidate=candidate,
            existing_snapshot=snapshot,
        )

    return asyncio.run(prepare())


def read_saved_memory(
    memory_manager: MemoryManager,
    user_id: str,
):
    """查询测试中的正式记忆。"""

    return asyncio.run(
        memory_manager.get_memory(
            user_id,
            "coding_style",
        )
    )


def test_confirm_pending_updates_existing_memory(monkeypatch):
    """HTTP 确认后，应当更新正式长期记忆。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager, pending_manager = (
        make_confirmation_test_app()
    )

    pending = create_conflicting_pending(
        manager,
        pending_manager,
        settings.user_id,
    )

    pending_id = pending["pending_id"]

    # 未确认前仍然保留旧记忆。
    before = read_saved_memory(manager, settings.user_id)
    assert before["content"] == "喜欢中文代码注释"

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/{pending_id}/confirm",
            headers=auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"

    # 确认后正式记忆更新。
    after = read_saved_memory(manager, settings.user_id)
    assert after["content"] == "喜欢英文代码注释"

    # Pending 已被消费。
    assert asyncio.run(
        pending_manager.get_pending(
            settings.user_id,
            pending_id,
        )
    ) is None


def test_reject_pending_preserves_existing_memory(monkeypatch):
    """拒绝更新后，正式记忆不能改变。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager, pending_manager = (
        make_confirmation_test_app()
    )

    pending = create_conflicting_pending(
        manager,
        pending_manager,
        settings.user_id,
    )

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/{pending['pending_id']}/reject",
            headers=auth_headers(),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"

    saved = read_saved_memory(
        manager,
        settings.user_id,
    )

    assert saved["content"] == "喜欢中文代码注释"

    assert asyncio.run(
        pending_manager.get_pending(
            settings.user_id,
            pending["pending_id"],
        )
    ) is None


def test_confirm_pending_twice_returns_not_found(monkeypatch):
    """同一个 Pending 不允许被重复确认执行。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager, pending_manager = (
        make_confirmation_test_app()
    )

    pending = create_conflicting_pending(
        manager,
        pending_manager,
        settings.user_id,
    )

    url = f"{ENDPOINT}/{pending['pending_id']}/confirm"

    with TestClient(test_app) as client:
        first = client.post(
            url,
            headers=auth_headers(),
        )

        second = client.post(
            url,
            headers=auth_headers(),
        )

    assert first.status_code == 200
    assert second.status_code == 404

    saved = read_saved_memory(
        manager,
        settings.user_id,
    )

    assert saved["content"] == "喜欢英文代码注释"


def test_confirm_stale_pending_returns_conflict(monkeypatch):
    """原始记忆变化后，不允许 Pending 覆盖新内容。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager, pending_manager = (
        make_confirmation_test_app()
    )

    pending = create_conflicting_pending(
        manager,
        pending_manager,
        settings.user_id,
    )

    # 模拟用户确认之前，另一操作修改了正式记忆。
    asyncio.run(
        manager.update_memory(
            settings.user_id,
            "coding_style",
            content="喜欢简洁的中文代码注释",
        )
    )

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/{pending['pending_id']}/confirm",
            headers=auth_headers(),
        )

    assert response.status_code == 409
    assert response.json()["detail"]["status"] == "stale"

    # 不能覆盖另一操作的新内容。
    saved = read_saved_memory(
        manager,
        settings.user_id,
    )

    assert saved["content"] == "喜欢简洁的中文代码注释"


def test_post_cannot_access_another_users_pending(monkeypatch):
    """服务端固定身份不能确认其他用户的 Pending。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager, pending_manager = (
        make_confirmation_test_app()
    )

    other_user = (
        "another_user"
        if settings.user_id != "another_user"
        else "other_user"
    )

    pending = create_conflicting_pending(
        manager,
        pending_manager,
        other_user,
    )

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/{pending['pending_id']}/confirm",
            headers=auth_headers(),
        )

    assert response.status_code == 404

    # 其他用户的正式记忆不得被更改。
    saved = read_saved_memory(
        manager,
        other_user,
    )

    assert saved["content"] == "喜欢中文代码注释"

    # 其他用户的 Pending 仍然存在。
    record = asyncio.run(
        pending_manager.get_pending(
            other_user,
            pending["pending_id"],
        )
    )

    assert record is not None


@pytest.mark.parametrize(
    "action",
    ["confirm", "reject"],
)
def test_memory_post_requires_api_key(monkeypatch, action):
    """确认和拒绝接口都必须经过 API Key 校验。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, manager, pending_manager = (
        make_confirmation_test_app()
    )

    pending = create_conflicting_pending(
        manager,
        pending_manager,
        settings.user_id,
    )

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/{pending['pending_id']}/{action}",
        )

    assert response.status_code == 401

    # 未认证请求不能消费 Pending。
    record = asyncio.run(
        pending_manager.get_pending(
            settings.user_id,
            pending["pending_id"],
        )
    )

    assert record is not None


def test_post_rejects_invalid_pending_uuid(monkeypatch):
    """非法 Pending ID 应返回 422。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app, _, _ = make_confirmation_test_app()

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/invalid-uuid/confirm",
            headers=auth_headers(),
        )

    assert response.status_code == 422


def test_post_returns_503_without_confirmation_service(
    monkeypatch,
):
    """未注入确认服务时，不能执行更新。"""

    monkeypatch.setenv(
        "HLW_MEMORY_API_KEY",
        TEST_API_KEY,
    )

    test_app = FastAPI()
    test_app.include_router(router)

    with TestClient(test_app) as client:
        response = client.post(
            f"{ENDPOINT}/"
            "00000000-0000-4000-8000-000000000001/confirm",
            headers=auth_headers(),
        )

    assert response.status_code == 503


def test_post_memory_routes_registered():
    """生产应用应注册确认和拒绝路由。"""

    schema = production_app.openapi()
    paths = schema["paths"]

    confirm_path = (
        "/api/v1/memories/pending/{pending_id}/confirm"
    )
    reject_path = (
        "/api/v1/memories/pending/{pending_id}/reject"
    )

    assert confirm_path in paths
    assert reject_path in paths

    assert "post" in paths[confirm_path]
    assert "post" in paths[reject_path]
