# Hermes Local Workbench（HLW）Docker 部署操作手册

**适用环境：Windows 11 + PowerShell + Conda/uv + Docker Desktop + Ollama + PostgreSQL**

**项目路径：**

```text
E:\myself\airarcle2\deepagents_learning\HLW
```

---

# 第一章：先理解 Docker 部署的整体架构

## 1.1 为什么要使用 Docker？

在没有 Docker 之前，你的 HLW 主要依赖 Windows 本机环境运行：

- Python 3.11
- uv 虚拟环境
- FastAPI
- LangGraph
- DeepAgents
- PostgreSQL

这种方式存在一个问题：程序依赖于本机的 Python 环境、依赖版本、系统配置。

如果以后换电脑、换服务器，可能需要重新配置整个环境。

Docker 解决的就是这个问题。

它可以把应用程序及其运行依赖封装成一个独立的运行环境，使应用在不同设备上具有更一致的运行方式。

注意：Docker 并不是把整个 Windows 系统打包，而是利用容器技术隔离应用的运行环境。

## 1.2 HLW 当前的部署架构

你的系统采用三个主要运行组件：

```text
                 Windows 主机
┌─────────────────────────────────────────────┐
│                                             │
│                  Postman                    │
│                     │                       │
│                     ▼                       │
│          http://127.0.0.1:8000              │
│                     │                       │
│    ┌──────────────────────────────────┐     │
│    │         Docker Desktop           │     │
│    │                                  │     │
│    │  ┌────────────────────────────┐  │     │
│    │  │       HLW API 容器         │  │     │
│    │  │                            │  │     │
│    │  │  FastAPI                   │  │     │
│    │  │       ↓                    │  │     │
│    │  │  AgentService              │  │     │
│    │  │       ↓                    │  │     │
│    │  │  Deep Agent Graph          │  │     │
│    │  │       ↓                    │  │     │
│    │  │  Model Gateway             │  │     │
│    │  └────────────┬───────────────┘  │     │
│    │               │                  │     │
│    │               │                  │     │
│    │  ┌────────────▼───────────────┐  │     │
│    │  │     PostgreSQL 容器        │  │     │
│    │  │                            │  │     │
│    │  │     Checkpoint             │  │     │
│    │  │     对话状态持久化         │  │     │
│    │  └────────────┬───────────────┘  │     │
│    │               │                  │     │
│    └───────────────┼──────────────────┘     │
│                    │                        │
│              data/postgres                  │
│              Windows 项目目录               │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │          Windows 本机 Ollama         │   │
│  │                                      │   │
│  │          qwen3:8b                    │   │
│  │          qwen3:14b                   │   │
│  │                                      │   │
│  │          端口：11434                 │   │
│  └──────────────────────────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

这里有一个重要设计：

**我们没有把 Ollama 和大模型放进 Docker，而是让 Docker 中的 HLW API 调用 Windows 本机的 Ollama。**

这样做有几个好处：

1. 不需要在 Docker 中重复安装大模型。
2. 不需要把模型文件复制到容器。
3. 可以继续使用本机已有的 Ollama 模型。
4. Docker 负责应用运行，Ollama 负责模型推理。

## 1.3 先认识几个 Docker 概念

| 名称 | 含义 | HLW 中的对应对象 |
|---|---|---|
| Docker Engine | Docker 的核心运行引擎 | 负责运行容器 |
| Docker Desktop | Windows 下管理 Docker 的软件 | 你的 Docker Desktop |
| Dockerfile | 描述如何构建镜像的文件 | 项目根目录 Dockerfile |
| Image 镜像 | 应用运行环境的模板 | hlw-api |
| Container 容器 | 镜像启动后的运行实例 | hlw-api-1 |
| Docker Compose | 管理多个容器的工具 | compose.yaml |
| Volume / Bind Mount | 数据持久化存储方式 | data/postgres |

可以这样理解：

- Dockerfile 是制作说明书。
- Image 是制作完成的产品。
- Container 是产品启动后的运行实例。
- Docker Compose 是负责管理这些实例的调度工具。

---

# 第二章：准备 Docker 运行环境

## 2.1 启动 Docker Desktop

首先打开 Windows 中的 Docker Desktop。

等待 Docker Engine 完全启动。

然后进入项目目录：

```powershell
cd E:\myself\airarcle2\deepagents_learning\HLW
```

为什么要先进入项目目录？

因为后续的 `docker compose` 默认会在当前目录寻找 `compose.yaml` 或 `docker-compose.yaml`。

如果不在项目目录执行，可能找不到配置文件。

## 2.2 检查 Docker 是否安装成功

执行：

```powershell
docker --version
```

作用：查看 Docker 客户端版本。

你的结果：

```text
Docker version 29.8.0
```

再执行：

```powershell
docker compose version
```

作用：检查 Docker Compose 是否可用。

你的结果：

```text
Docker Compose version v5.5.1
```

最后执行：

```powershell
docker info
```

作用：检查 Docker 客户端和服务端是否都能正常通信。

这里需要特别理解：

Docker 有客户端和服务端两个部分。

- Client：你在 PowerShell 输入的 docker 命令。
- Server：真正负责创建和运行容器的 Docker Engine。

如果看到：

```text
Client:
Version: 29.8.0

Server:
Version: 29.8.0
```

说明客户端已经成功连接 Docker Engine。

你之前遇到过：

```text
failed to connect to the docker API
```

这是因为 Docker Desktop 显示运行，不代表 Linux Engine 已经准备好。

遇到这种情况，应等待 Docker Desktop 完全启动，然后重新执行 `docker info`。

**成功标准：docker info 能正常显示 Server 信息。**

---

# 第三章：准备 PostgreSQL 数据库

## 3.1 为什么需要 PostgreSQL？

HLW 使用 LangGraph 和 DeepAgents。

Agent 在运行过程中会产生很多状态：

- 用户消息
- AI 消息
- 工具调用
- 工具返回结果
- Graph 执行节点状态
- Checkpoint 状态快照

如果只把这些信息放在内存中，那么程序关闭后就可能丢失。

因此需要 Checkpointer。

你目前使用的是：

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
```

它负责把 LangGraph 的状态保存到 PostgreSQL。

这样，即使 API 容器重启，也可以通过相同的 `thread_id` 恢复历史状态。

## 3.2 为什么不能把数据库数据只存在容器内部？

假设 PostgreSQL 数据直接保存在容器内部：

```text
PostgreSQL Container
    └── 数据库文件
```

如果容器被删除，容器内部的数据也可能随之消失。

所以我们要把数据库数据保存到 Windows 项目文件夹。

你的最终目录是：

```text
HLW/
│
├── data/
│   └── postgres/
│       ├── PG_VERSION
│       ├── base/
│       ├── global/
│       ├── pg_wal/
│       └── ...
│
├── src/
├── scripts/
├── tests/
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── uv.lock
└── .env
```

其中：

```text
data/postgres/
```

就是 PostgreSQL 实际存放数据库文件的位置。

## 3.3 创建数据库目录

执行：

```powershell
New-Item -ItemType Directory -Force .\data\postgres | Out-Null
```

逐项解释：

- `New-Item`：创建项目。
- `-ItemType Directory`：创建文件夹。
- `-Force`：如果文件夹已经存在，不会因为重复创建而报错。
- `.\data\postgres`：相对于当前项目目录的路径。
- `Out-Null`：不显示创建结果。

为什么这样做？

因为我们需要先创建一个本机目录，供 Docker 映射为 PostgreSQL 的数据存储位置。

## 3.4 配置 compose.yaml

你的 PostgreSQL 配置如下：

```yaml
name: hlw

services:
  postgres:
    image: postgres:16
    restart: unless-stopped

    environment:
      POSTGRES_USER: ${HLW_DB_USER:-hlw}
      POSTGRES_PASSWORD: ${HLW_DB_PASSWORD:?Set HLW_DB_PASSWORD}
      POSTGRES_DB: ${HLW_DB_NAME:-hlw}

    ports:
      - "127.0.0.1:5432:5432"

    volumes:
      - ./data/postgres:/var/lib/postgresql/data

    healthcheck:
      test:
        - CMD-SHELL
        - pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB
      interval: 5s
      timeout: 5s
      retries: 10
```

逐项理解：

### name

```yaml
name: hlw
```

Compose 项目名称。

Docker 会根据项目名称组织容器、网络等资源。

### services

```yaml
services:
  postgres:
```

表示定义一个名为 postgres 的服务。

这里的 `postgres` 不只是服务名称，它还可以作为同一个 Compose 网络中其他容器访问 PostgreSQL 的主机名。

### image

```yaml
image: postgres:16
```

表示使用官方 PostgreSQL 16 镜像。

其中：

- postgres：镜像名称。
- 16：PostgreSQL 主版本。

### restart

```yaml
restart: unless-stopped
```

表示除非人为主动停止，否则容器异常退出或 Docker 重启后，会尝试重新启动。

### environment

```yaml
environment:
  POSTGRES_USER: ${HLW_DB_USER:-hlw}
  POSTGRES_PASSWORD: ${HLW_DB_PASSWORD:?Set HLW_DB_PASSWORD}
  POSTGRES_DB: ${HLW_DB_NAME:-hlw}
```

这是 PostgreSQL 初始化时需要的环境变量。

| 环境变量 | 含义 | 默认值 |
|---|---|---|
| POSTGRES_USER | 数据库用户 | hlw |
| POSTGRES_PASSWORD | 数据库密码 | 必须提供 |
| POSTGRES_DB | 默认数据库 | hlw |

这里的语法：

```yaml
${HLW_DB_USER:-hlw}
```

意思是：如果没有设置 HLW_DB_USER，就使用 hlw。

而：

```yaml
${HLW_DB_PASSWORD:?Set HLW_DB_PASSWORD}
```

意思是：如果没有设置密码，就报错，不允许继续启动。

为什么要这样设计？

避免数据库使用空密码运行。

### ports

```yaml
ports:
  - "127.0.0.1:5432:5432"
```

格式：

```text
主机IP:主机端口:容器端口
```

这里：

- Windows 主机端口：5432。
- PostgreSQL 容器端口：5432。

因此本机数据库客户端可以访问：

```text
127.0.0.1:5432
```

绑定 `127.0.0.1` 的作用是限制数据库端口只在本机开放，不直接暴露给局域网。

### volumes

```yaml
volumes:
  - ./data/postgres:/var/lib/postgresql/data
```

这是整个 PostgreSQL 数据持久化的关键。

左边：

```text
./data/postgres
```

是 Windows 项目目录。

右边：

```text
/var/lib/postgresql/data
```

是 PostgreSQL 容器内部的数据目录。

Docker 将两者映射起来。

所以：

**数据库写入容器内部的数据目录，实际上就是写入 Windows 项目的 data/postgres 文件夹。**

这叫 Bind Mount（绑定挂载）。

### healthcheck

```yaml
healthcheck:
  test:
    - CMD-SHELL
    - pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB
  interval: 5s
  timeout: 5s
  retries: 10
```

作用：定期检查 PostgreSQL 是否真的可以接受连接。

- interval：每隔 5 秒检查。
- timeout：每次检查最多等待 5 秒。
- retries：连续失败 10 次后标记为不健康。

注意：容器显示 Running，不一定意味着数据库已经可以正常接受连接。

因此我们使用 Healthy 判断数据库是否真正准备好。

## 3.5 配置 .env

在 HLW 项目根目录创建或编辑：

```text
.env
```

配置示例：

```env
HLW_DB_USER=hlw
HLW_DB_PASSWORD=填写你自己的强密码
HLW_DB_NAME=hlw
HLW_DB_PORT=5432

HLW_ENV=development

HLW_CHAT_MODEL=qwen3:8b
HLW_FALLBACK_MODEL=qwen3:14b

HLW_OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
HLW_MODEL_TIMEOUT_SECONDS=120

HLW_LOG_LEVEL=INFO
```

注意：

这里的 `127.0.0.1` 是本机运行程序时访问 Ollama 的地址。

Docker API 容器内访问 Ollama 时，需要使用：

```text
http://host.docker.internal:11434/v1
```

因此 Docker API 服务需要通过 Compose 环境变量覆盖 Ollama 地址。

同样，API 容器访问 PostgreSQL 时应该使用：

```text
postgres
```

而不是：

```text
127.0.0.1
```

因为在 Docker 容器中，127.0.0.1 指向容器自己。

这两个网络地址的区别非常重要。

另外，`.env` 中的密码不要上传到 GitHub，应该加入 `.gitignore`。

## 3.6 检查 Compose 配置

执行：

```powershell
docker compose config --quiet
```

作用：检查 Compose 文件及其环境变量是否能够正确解析。

成功时通常没有输出，直接返回命令提示符。

你之前遇到：

```text
required variable HLW_DB_PASSWORD is missing a value
```

就是因为没有提供数据库密码。

解决后再次执行，成功通过。

---

# 第四章：启动 PostgreSQL

## 4.1 启动数据库

执行：

```powershell
docker compose up -d postgres
```

拆解：

- `docker compose`：使用 Compose 管理项目。
- `up`：创建并启动服务。
- `-d`：后台运行。
- `postgres`：只启动 PostgreSQL 服务。

为什么先启动 PostgreSQL？

因为 HLW API 初始化时需要打开 PostgreSQL Checkpointer。

如果数据库没有准备好，AgentService 就可能初始化失败。

所以先启动数据库，再启动 API。

## 4.2 检查容器

执行：

```powershell
docker compose ps
```

预期结果：

```text
NAME             IMAGE         SERVICE
hlw-postgres-1   postgres:16   postgres
```

状态应该是：

```text
Up (healthy)
```

## 4.3 验证数据库是否真的能连接

执行：

```powershell
docker compose exec postgres psql -U hlw -d hlw -c "SELECT current_database(), current_user;"
```

解释：

- `exec`：进入正在运行的容器执行命令。
- `postgres`：指定容器服务。
- `psql`：PostgreSQL 命令行客户端。
- `-U hlw`：使用 hlw 用户。
- `-d hlw`：连接 hlw 数据库。
- `-c`：执行 SQL 命令。

你的成功结果：

```text
 current_database | current_user
------------------+--------------
 hlw              | hlw
```

说明数据库可以正常连接。

## 4.4 检查数据目录

执行：

```powershell
Test-Path .\data\postgres\PG_VERSION
```

返回：

```text
True
```

再执行：

```powershell
Get-Content .\data\postgres\PG_VERSION
```

返回：

```text
16
```

说明 PostgreSQL 16 已经在项目数据目录中初始化。

这一步证明数据不是只存在容器内部。

---

# 第五章：准备 FastAPI 和项目依赖

## 5.1 为什么要添加 FastAPI？

之前你的 HLW 主要通过 LangGraph API Server 或 Python 脚本调用 Agent。

现在我们希望通过 Postman 访问 HLW。

因此需要一个 HTTP API 入口。

FastAPI 就负责这项工作。

例如：

```text
POST /chat
```

接收用户消息，然后调用 AgentService。

## 5.2 安装依赖

在项目根目录执行：

```powershell
uv add fastapi "uvicorn[standard]"
```

其中：

- `uv add`：向项目添加依赖。
- `fastapi`：HTTP API 框架。
- `uvicorn[standard]`：ASGI 服务器，用来运行 FastAPI。

为什么需要 Uvicorn？

FastAPI 只是定义 API 的框架，不是独立运行的 Web 服务器。

Uvicorn 负责监听端口、接收 HTTP 请求，并将请求交给 FastAPI。

安装完成后，检查：

```powershell
uv run python -c "import fastapi, uvicorn; print(fastapi.__version__); print(uvicorn.__version__)"
```

你实际得到：

```text
FastAPI: 0.162.0
Uvicorn: 0.54.0
```

## 5.3 理解 pyproject.toml 和 uv.lock

你的项目有两个重要文件：

### pyproject.toml

记录项目需要什么依赖。

例如：

```toml
dependencies = [
    "deepagents>=0.7.32",
    "langchain-openai>=1.7.2",
    "langgraph-checkpoint-postgres>=3.2.0",
    "psycopg[binary]>=3.3.6",
    "fastapi",
    "uvicorn[standard]"
]
```

### uv.lock

记录实际解析出来的依赖版本。

两者的区别：

- pyproject.toml：声明需要哪些软件。
- uv.lock：记录具体使用哪些版本。

Docker 构建时使用：

```dockerfile
uv sync --frozen --no-dev
```

其中：

- `uv sync`：安装项目依赖。
- `--frozen`：严格按照 uv.lock 安装，不重新修改锁文件。
- `--no-dev`：不安装开发依赖。

为什么使用 `--no-dev`？

因为正式运行 API 不需要 pytest、ruff 等开发工具。

这样可以减少镜像体积，避免把开发工具带进运行环境。

注意：FastAPI 和 Uvicorn 必须属于生产依赖，不能只放在 dev 依赖组中。

## 5.4 测试代码

执行：

```powershell
uv run pytest -q
```

你之前的结果：

```text
42 passed
```

再执行：

```powershell
uv run ruff check src tests
```

结果：

```text
All checks passed!
```

说明当前测试和代码规范检查通过。

---

# 第六章：准备 Dockerfile 和 API 服务

## 6.1 Dockerfile 是什么？

Dockerfile 是告诉 Docker：

如何制作 HLW 应用镜像的文件。

你的构建日志显示，主要进行了以下操作：

```dockerfile
FROM python:3.11-slim
```

使用 Python 3.11 的轻量基础镜像。

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
```

将 uv 工具复制到镜像。

```dockerfile
WORKDIR /app
```

设置容器内部工作目录。

```dockerfile
COPY pyproject.toml uv.lock README.md ./
```

先复制项目依赖配置。

```dockerfile
RUN uv sync --frozen --no-dev --no-install-project
```

先安装依赖，但暂时不安装 HLW 本身。

```dockerfile
COPY src ./src
```

将 HLW 源代码复制到镜像。

```dockerfile
RUN uv sync --frozen --no-dev
```

最后安装 HLW 项目本身。

为什么先复制依赖文件，再复制源代码？

因为 Docker 有构建缓存机制。

如果只修改 Python 源代码，而依赖配置没有变化，前面的依赖安装步骤就可能直接复用缓存。

这样可以缩短后续构建时间。

## 6.2 API 服务如何运行？

你的 FastAPI 入口文件：

```text
src/hlw/api/app.py
```

里面定义了：

```python
app = FastAPI(
    title="Hermes Local Workbench API",
    version="0.1.0",
    lifespan=lifespan,
)
```

其中 lifespan 负责应用生命周期管理。

启动时：

1. 初始化日志系统。
2. 打开 PostgreSQL Checkpointer。
3. 创建持久化 Agent Graph。
4. 创建 AgentService。
5. 将 service 保存到 app.state。

关闭时：

1. 清理 app.state。
2. 退出 Checkpointer 上下文。
3. 释放 PostgreSQL 连接。

你的 API 入口还有两个重要路由：

```text
GET  /health
POST /chat
```

`/health` 用于健康检查。

`/chat` 用于真正调用 Agent。

## 6.3 为什么使用 host.docker.internal？

这是 Docker 部署中非常重要的知识。

在 Windows 本机：

```text
127.0.0.1:11434
```

指向 Windows 自己。

但在 Docker API 容器内部：

```text
127.0.0.1:11434
```

指向 API 容器自己，而不是 Windows。

因此需要：

```text
host.docker.internal:11434
```

它是 Docker Desktop 提供的特殊主机名，可以让容器访问宿主机。

你最终的日志已经证明通信成功：

```text
POST http://host.docker.internal:11434/v1/chat/completions
HTTP/1.1 200 OK
```

说明容器成功访问 Windows Ollama。

---

# 第七章：构建 Docker 镜像

## 7.1 为什么要构建镜像？

之前我们已经准备好了：

- Python 环境
- 项目依赖
- FastAPI
- Agent 代码
- PostgreSQL 配置
- Dockerfile

但这些还没有变成可以运行的 HLW Docker 镜像。

因此需要执行：

```powershell
docker compose build api
```

解释：

- `build`：构建镜像。
- `api`：只构建名为 api 的服务。

构建过程中 Docker 会读取 Dockerfile，逐步执行里面的指令。

## 7.2 你遇到的网络错误

第一次构建时：

```text
failed to fetch oauth token
```

原因是 Docker Hub 网络连接超时。

失败的位置：

```dockerfile
FROM python:3.11-slim
```

解决方式是先手动拉取基础镜像：

```powershell
docker pull python:3.11-slim
```

这条命令的作用：

从 Docker Hub 下载 Python 3.11 基础镜像到本地。

成功后再执行：

```powershell
docker compose build api
```

你的最终结果：

```text
[+] build 1/1
✔ Image hlw-api Built
```

这说明 HLW API 镜像构建成功。

生成的镜像名称：

```text
hlw-api:latest
```

注意：镜像构建成功不代表容器已经启动。构建和运行是两个不同阶段。

---

# 第八章：启动 HLW Docker 服务

## 8.1 启动所有服务

执行：

```powershell
docker compose up -d
```

这条命令会：

1. 检查 Compose 文件。
2. 创建所需网络。
3. 创建或复用 PostgreSQL 容器。
4. 创建 HLW API 容器。
5. 启动所有服务。
6. 让服务在后台运行。

为什么不再单独启动 postgres？

因为这里没有指定具体服务名称，所以会启动 Compose 文件中定义的所有服务。

## 8.2 检查容器状态

执行：

```powershell
docker compose ps
```

你的实际结果：

```text
NAME             IMAGE         SERVICE
hlw-api-1        hlw-api       api
hlw-postgres-1   postgres:16   postgres
```

状态：

```text
hlw-api-1        Up (healthy)
hlw-postgres-1   Up (healthy)
```

这说明两个服务都已经启动并通过健康检查。

## 8.3 查看 API 日志

执行：

```powershell
docker compose logs --tail 50 api
```

解释：

- `logs`：查看容器日志。
- `--tail 50`：只显示最后 50 行。
- `api`：只查看 API 服务。

你看到：

```text
HLW API startup
HLW AgentService initialized
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

分别代表：

| 日志 | 含义 |
|---|---|
| HLW API startup | FastAPI 开始启动 |
| AgentService initialized | AgentService 创建完成 |
| Application startup complete | 应用启动完成 |
| Uvicorn running | HTTP 服务开始监听端口 |

其中：

```text
0.0.0.0:8000
```

表示容器内部监听所有网络接口上的 8000 端口。

而你的端口映射将其暴露到 Windows 本机的 8000 端口。

---

# 第九章：Postman 测试 HLW API

## 9.1 健康检查

在 Postman 创建请求：

```text
GET http://127.0.0.1:8000/health
```

预期结果：

```json
{
  "status": "ok"
}
```

说明 HTTP API 能够正常接收请求。

## 9.2 调用 Agent

新建请求：

```text
POST http://127.0.0.1:8000/chat
```

Headers：

```text
Content-Type: application/json
```

Body 选择：

```text
raw → JSON
```

输入：

```json
{
  "message": "请调用 get_project_status，告诉我当前项目的开发里程碑。",
  "thread_id": "hlw-docker-001"
}
```

点击 Send。

预期返回：

```json
{
  "thread_id": "hlw-docker-001",
  "request_id": "自动生成的请求ID",
  "answer": "当前项目里程碑为 P0-02……"
}
```

理解三个字段：

| 字段 | 含义 |
|---|---|
| message | 用户输入的问题 |
| thread_id | 会话 ID，用于区分不同会话 |
| request_id | 单次请求 ID，用于日志追踪 |

特别注意：

同一个 thread_id 可以有多次请求。

每次请求可以使用不同的 request_id。

例如：

```text
thread_id = hlw-docker-001

第一次请求：
request_id = A

第二次请求：
request_id = B

第三次请求：
request_id = C
```

这三次请求属于同一个会话，但属于三个不同的请求。

---

# 第十章：验证 PostgreSQL 持久化

## 10.1 查询 Checkpoint

Postman 请求成功后，回到 PowerShell：

```powershell
docker compose exec postgres psql -U hlw -d hlw -c "SELECT thread_id, COUNT(*) AS checkpoint_count FROM checkpoints WHERE thread_id = 'hlw-docker-001' GROUP BY thread_id;"
```

你的实际结果：

```text
thread_id      | checkpoint_count
---------------+-----------------
hlw-docker-001 | 7
```

说明：

Agent 已经成功将执行状态写入 PostgreSQL。

注意：7 表示 Checkpoint 数量，不是 7 条用户消息。

## 10.2 为什么需要验证同一 thread_id？

因为我们需要证明 PostgreSQL 不只是接收数据，而且能够恢复对话状态。

再次通过 Postman 发送：

```json
{
  "message": "我上一轮让你查询的项目里程碑是什么？请只回答编号。",
  "thread_id": "hlw-docker-001"
}
```

如果 Agent 能回答：

```text
P0-02
```

说明它能够通过相同 thread_id 恢复之前的会话上下文。

## 10.3 验证 API 重启后的记忆恢复

执行：

```powershell
docker compose restart api
```

注意：这里只重启 API，不重启 PostgreSQL。

检查：

```powershell
docker compose ps
```

确保 API 再次显示 Healthy。

然后再次使用原来的 thread_id：

```json
{
  "message": "请告诉我之前对话中，我最初询问的项目里程碑是什么？",
  "thread_id": "hlw-docker-001"
}
```

如果仍然可以正确回答：

```text
P0-02
```

就证明 API 容器重启后能够重新加载 PostgreSQL 中保存的历史状态。

**这是 Docker + PostgreSQL + LangGraph 持久化机制最重要的验收实验。**

---

# 第十一章：日常 Docker 管理命令

以后不需要每次重新安装依赖，也不需要每次重新构建镜像。

掌握下面这些命令就够了。

## 11.1 启动服务

```powershell
docker compose up -d
```

作用：启动所有服务。

## 11.2 查看运行状态

```powershell
docker compose ps
```

作用：查看容器是否运行、是否健康。

## 11.3 查看 API 日志

```powershell
docker compose logs --tail 50 api
```

作用：查看最近 50 行 API 日志。

## 11.4 实时观察日志

```powershell
docker compose logs -f api
```

作用：持续显示新产生的日志。

按 Ctrl+C 只是退出日志观察，不会停止容器。

## 11.5 重启 API

```powershell
docker compose restart api
```

作用：重启 API 容器。

适合代码配置不变、只需要重启服务的情况。

## 11.6 修改代码后重新构建

```powershell
docker compose build api
```

作用：重新构建 API 镜像。

然后：

```powershell
docker compose up -d
```

作用：使用新镜像启动 API。

也可以合并为：

```powershell
docker compose up -d --build
```

表示先构建，再启动。

## 11.7 停止服务

```powershell
docker compose stop
```

作用：停止容器，但保留容器本身。

以后可以再次执行：

```powershell
docker compose start
```

重新启动。

## 11.8 删除容器和网络

```powershell
docker compose down
```

作用：停止并删除当前 Compose 项目的容器和网络。

注意：

这不会删除你通过 Bind Mount 保存的：

```text
data/postgres/
```

数据库文件。

但是，不要随意执行：

```powershell
docker compose down -v
```

这个命令会额外删除 Compose 管理的命名卷和匿名卷。以后如果增加其他命名卷，可能导致数据丢失。

更重要的是：不要手动删除 `data/postgres`。

## 11.9 查看镜像

```powershell
docker images
```

作用：查看本机已经下载或构建的镜像。

## 11.10 查看数据库

```powershell
docker compose exec postgres psql -U hlw -d hlw
```

进入 PostgreSQL 交互界面。

输入：

```sql
SELECT COUNT(*) FROM checkpoints;
```

查询 Checkpoint 总数。

退出：

```sql
\q
```

---

# 第十二章：常见错误及解决办法

| 错误 | 原因 | 解决方式 |
|---|---|---|
| Docker API 连接失败 | Docker Engine 未启动 | 启动 Docker Desktop，检查 docker info |
| HLW_DB_PASSWORD missing | 没有配置数据库密码 | 检查 .env |
| Docker Hub Token 超时 | 网络连接问题 | 先 docker pull 基础镜像 |
| PyPI 超时 | Python 包源网络问题 | 检查网络，重新执行 uv add 或构建 |
| PostgreSQL 不健康 | 数据库没有正常启动 | 查看 postgres 日志 |
| API 无法连接数据库 | 数据库地址或密码不正确 | 容器内使用 postgres 服务名 |
| API 无法调用 Ollama | 容器访问地址错误 | 使用 host.docker.internal |
| Postman 连接失败 | API 未启动或端口不正确 | 检查 docker compose ps 和 logs |
| 找不到历史对话 | thread_id 不一致或持久化异常 | 检查 thread_id 和 checkpoints |

常用故障排查命令：

```powershell
docker compose ps
```

```powershell
docker compose logs --tail 100 api
```

```powershell
docker compose logs --tail 100 postgres
```

```powershell
docker compose config --quiet
```

---

# 第十三章：本次 HLW 部署成果与验收

根据实际运行结果，目前已经完成以下工作：

| 验收项目 | 结果 |
|---|---|
| Docker Engine 安装 | 成功 |
| Docker Compose 安装 | 成功 |
| PostgreSQL 16 镜像下载 | 成功 |
| PostgreSQL 本地数据目录 | 成功 |
| PostgreSQL 连接验证 | 成功 |
| PostgreSQL Checkpointer | 成功 |
| FastAPI 安装 | 成功 |
| Uvicorn 安装 | 成功 |
| Dockerfile 构建 | 成功 |
| HLW API 镜像 | 成功 |
| PostgreSQL 容器 | Healthy |
| HLW API 容器 | Healthy |
| Postman 调用 | HTTP 200 OK |
| Docker 调用 Windows Ollama | 成功 |
| Qwen3:8b 推理 | 成功 |
| PostgreSQL 写入 Checkpoint | 成功，首次测试为 7 条 |
| 单元测试 | 42 passed |
| Ruff 检查 | All checks passed |

还有一项需要单独完成：

**API 容器重启之后，使用原 thread_id 恢复对话的端到端测试。**

此前已经验证过 PostgreSQL 持久化及独立服务恢复，但 Docker 容器重启后的这项测试还需要实际执行。

---

# 第十四章：以后应该怎样理解整个部署过程？

将整个流程浓缩为八个阶段：

| 阶段 | 做什么 | 为什么 |
|---|---|---|
| 1 | 安装并启动 Docker Desktop | 提供容器运行环境 |
| 2 | 创建 PostgreSQL 数据目录 | 保证数据库数据持久化 |
| 3 | 配置 .env 和 compose.yaml | 统一管理数据库及服务配置 |
| 4 | 安装 FastAPI、Uvicorn | 提供 HTTP 服务入口 |
| 5 | 准备 Dockerfile | 定义应用镜像的制作过程 |
| 6 | docker compose build api | 将应用打包成镜像 |
| 7 | docker compose up -d | 创建并运行容器 |
| 8 | Postman + SQL 验收 | 验证 Agent、模型和数据库完整链路 |

最后记住这四个最重要的命令：

```powershell
# 1. 构建镜像
docker compose build api

# 2. 启动服务
docker compose up -d

# 3. 查看运行状态
docker compose ps

# 4. 查看 API 日志
docker compose logs -f api
```

**Docker 部署的核心逻辑：**

先准备环境，再制作镜像，然后启动容器，最后验证应用运行和数据持久化。

这就是 HLW 从 Windows 本地 Python 程序走向容器化应用部署的完整过程。