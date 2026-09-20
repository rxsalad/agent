# MCP (Model Context Protocol)

MCP（Model Context Protocol）是一种标准化协议，用于让 AI Agent 与外部工具、数据源和服务进行交互。它将工具的**发现**、**描述**和**调用**统一为标准接口，使 Agent 不需要为每个外部系统编写专用集成代码。

一句话概括：**MCP 是连接 AI Agent 和外部工具/数据/服务的标准协议，而不是模型本身的 Tool Calling API。**

---

## 核心关系：MCP vs Function Calling

理清 MCP 和 Function Calling 的关系，是理解整个体系的关键：

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| **本质** | 模型调用"工具"的机制 | 工具标准化、发现化、连接化的协议 |
| **谁来实现** | 模型（经过训练理解 Schema 并生成 Tool Call） | Client ↔ Server 之间的通信层 |
| **模型需要懂吗** | 需要 —— 理解 Schema，决定何时调用 | 不需要 —— 由 MCP Client 转换 |

**分工：**

- **模型**：经过训练后能够理解 Tool Schema，根据用户 Prompt 判断是否需要调用工具，生成工具名称和参数
- **Agent / MCP Client**：把 MCP Server 提供的工具信息转换成模型 API 可用的 `tools[]`，执行模型返回的 Tool Call，将结果送回模型
- **MCP Server**：通过标准接口暴露工具（name / description / input schema），处理实际的业务逻辑

---

## 底层：JSON-RPC 2.0

MCP 使用 JSON-RPC 2.0 作为通信格式。理解 JSON-RPC，就理解了 MCP 的消息结构。

### 基本调用

假设有函数 `add(a, b)`，客户端想调用 `add(10, 20)`：

**请求：**
```json
{
  "jsonrpc": "2.0",
  "method": "add",
  "params": { "a": 10, "b": 20 },
  "id": 1
}
```

**响应：**
```json
{
  "jsonrpc": "2.0",
  "result": 30,
  "id": 1
}
```

### 错误响应

当调用出错时，`result` 替换为 `error`，包含标准错误码：

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found"
  },
  "id": 1
}
```

### 通知（Notification）

不带 `id` 字段 = 不需要响应，用于状态告知：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

---

## MCP 协议：三消息模型

在 JSON-RPC 之上，MCP 定义了三个核心消息，覆盖了完整的工具交互流程：

```
  ① initialize      — 握手，协商版本与能力
  ② tools/list      — 发现，查询可用工具及 Schema
  ③ tools/call      — 调用，传入参数，获得结果
```

完整交互时序：

```
  Client                              Server
    |                                    |
    |—— initialize ━━━━━━━━━━━━━━━━━━━>|  握手：发送协议版本、客户端信息
    |<━━ result (serverInfo) ━━━━━━━━━|  返回：服务器名称、版本、能力
    |                                    |
    |—— notifications/initialized ━━━>|  告知握手完成（单向通知）
    |                                    |
    |—— tools/list ━━━━━━━━━━━━━━━━━━━>|  查询：有哪些可用工具？
    |<━━ [tool defs + schemas] ━━━━━━━|  返回：工具名、描述、参数 Schema
    |                                    |
    |—— tools/call (name, args) ━━━━━>|  调用：执行某个工具
    |<━━ result (tool output) ━━━━━━━━|  返回：工具执行结果
```

---

## MCP 的三大使能

MCP Server 可以向 Client 暴露三类能力：

| 能力 | 作用 | 类比 |
|------|------|------|
| **Tools** | 可执行的操作（如查天气、搜索、数据库查询） | 函数的 API |
| **Resources** | 可读的数据（如文件内容、数据库记录、日志） | 数据的文件路径 |
| **Prompts** | 可复用的提示词模板（如代码审查模板、摘要模板） | 预写好的 Prompt 模板 |

Tools 是 Agent 最核心的能力载体 —— 模型通过 Tool Calling 机制调用 MCP 暴露的 Tools。Resources 和 Prompts 则提供了数据读取和交互模板的能力。

---

## 架构与传输层

### 典型架构

```
  ┌─────────────────────────────────────────────────┐
  │                 Agent / Application              │
  │                                                  │
  │   ┌─────────────┐    ┌────────────────────────┐  │
  │   │   LLM Model  │    │       MCP Client        │  │
  │   │             │    │  · 连接 MCP Server       │  │
  │   │  决定调用    │◄──►│  · 转换 Schema          │  │
  │   │  生成参数    │    │  · 执行 Tool Call       │  │
  │   │  解读结果    │    │  · 回送 Tool Result     │  │
  │   └─────────────┘    └──────────┬─────────────┘  │
  └─────────────────────────────────┼────────────────┘
                                    │  JSON-RPC 2.0
                                    │  (stdio / HTTP)
                                    ▼
                          ┌──────────────────────┐
                          │     MCP Server        │
                          │  · Tools 注册         │
                          │  · Resources 暴露     │
                          │  · Prompts 模板       │
                          └──────────────────────┘
```

### 传输层协议

| 传输方式 | 典型场景 | 状态 |
|----------|----------|------|
| **stdio** | Client 本机启动 MCP Server 子进程 | ✅ 推荐（本地方式） |
| **Streamable HTTP** | 远程 MCP Server | ✅ **推荐（远程方式）** |
| **HTTP + SSE** | 老版远程 MCP Server | ⚠️ Legacy / Deprecated |

stdio 传输通过 `stdin`/`stdout` 传递 JSON-RPC 消息（每行一条 JSON），Streamable HTTP 则通过 HTTP POST 发送请求，支持 session 管理。

---

## MCP Server 的主要职责

使用 MCP SDK（如 Python 的 `mcp.server.MCPServer`）时，SDK 帮我们处理了协议层的全部细节：

- **初始化握手** — `initialize` 协商版本、`notifications/initialized` 确认完成
- **工具发现** — `tools/list` 扫描 `@mcp.tool()` 装饰器注册的工具
- **工具调用** — `tools/call` 路由到对应函数，处理参数校验
- **JSON-RPC 封包/解包** — 请求 ID 管理、错误格式、通知机制

开发者只需要：

```python
from mcp.server import MCPServer

mcp = MCPServer("weather")

@mcp.tool()
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"temperature": 65, "unit": "F", "condition": "Partly cloudy"}

if __name__ == "__main__":
    mcp.run(transport="stdio")         # 本地子进程
    # mcp.run(transport="streamable-http")  # 远程 HTTP
```

`@mcp.tool()` 装饰器完成了：从函数签名推导参数 Schema，从 docstring 提取描述，注册到工具表。

---

## 实验

### 环境准备

```bash
pip3 install "mcp[cli]" --break-system-packages
pip3 install -U python-dotenv --break-system-packages
pip3 install -U openai --break-system-packages
pip3 install httpx --break-system-packages
```

### 实验 1：stdio 传输（纯手写 JSON，不依赖 MCP SDK）

客户端 `mcp/mcp-client-simple.py` 通过 `subprocess` 启动 Server，用原生 JSON-RPC 完成 MCP 交互：

```bash
python3 mcp/mcp-client-simple.py
# 自动启动子进程 mcp/mcp-weather-server.py
```

输出：

```
启动 MCP server ...
握手 initialize ...
  server: {'name': 'weather', 'version': ''}
获取工具列表 ...
  工具: get_weather — Get the current weather for a city.

调用工具: get_weather({'city': 'San Francisco'})
工具返回: {
  "temperature": 65,
  "unit": "F",
  "condition": "Partly cloudy"
}

最终答案: The current weather in San Francisco is partly cloudy with a temperature of 65°F.
```

### 实验 2：Streamable HTTP 传输（纯手写 JSON）

服务端 `mcp/mcp-weather-server-http.py` 先启动，客户端 `mcp/mcp-client-simple-http.py` 后连接：

```bash
# 终端 1：启动 Server
python3 mcp/mcp-weather-server-http.py

# 终端 2：运行 Client
python3 mcp/mcp-client-simple-http.py
```

输出：

```
连接 MCP server http://127.0.0.1:8000/mcp ...
握手 initialize ...
  server: {'name': 'weather', 'version': ''}
  session-id: 356ef86cad574bc5ad6b3b341692ee04
获取工具列表 ...
  工具: get_weather — Get the current weather for a city.

调用工具: get_weather({'city': 'San Francisco'})
工具返回: {
  "temperature": 65,
  "unit": "F",
  "condition": "Partly cloudy"
}

最终答案: The weather in San Francisco is currently partly cloudy with a temperature of 65°F.

连接已关闭。
```

### 实验 3：使用 MCP SDK（async + stdio）

`mcp/mcp-client-async.py` 使用 `mcp` SDK 的 `ClientSession` 和 `stdio_client`，展示了生产级用法：

```bash
python3 mcp/mcp-client-async.py
```

---

## 关键理解

1. **模型不需要"懂 MCP"** —— MCP 是 Client 和 Server 之间的协议。模型只需要理解 Tool Schema（name / description / parameters），这些由 MCP Client 从 Server 获取后转换为模型 API 的 `tools[]` 格式。

2. **MCP 解决的是"连接"问题** —— 没有 MCP 之前，每个工具需要写专用集成代码；有了 MCP，工具实现者只需实现一个 MCP Server，任何 MCP Client 都能发现和调用。

3. **MCP 和 Tool Calling 是互补的** —— Tool Calling 让模型知道"可以调用什么工具"，MCP 让工具能够被"标准化地发现和连接"。二者结合，构成了完整的 Agent 工具调用链路。

---

## 参考资料

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP Specification - Transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP 中文解析](https://zhuanlan.zhihu.com/p/27327515233)
