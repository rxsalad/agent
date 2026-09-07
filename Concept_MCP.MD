# MCP

MCP（Model Context Protocol）是一种标准化的协议，用于让 AI Agent 与外部工具、数据源和服务进行交互。 MCP 将工具的发现（Tool Discovery）、Schema 描述以及工具调用统一为标准接口，使 Agent 不需要针对每个外部系统开发一套专用集成。典型架构中，MCP Client 运行在 Agent/应用侧，与一个或多个 MCP Server 建立连接；MCP Server 对外提供 Tools、Resources 和 Prompts，Client 可以发现这些能力，并在模型需要时调用相应的 Tool。MCP 本身负责 Agent 与外部能力之间的标准化通信和能力暴露，而模型最终看到的 Tool Schema、Prompt 以及 Tool Result，则由 Agent/MCP Client 根据具体模型 API 和 Chat Template 转换成模型能够理解的上下文。

MCP 是连接 AI Agent 和外部工具/数据/服务 的标准协议，而不是模型本身的 Tool Calling API；MCP Client 负责把 MCP Server 提供的能力转换成模型 API 可以使用的 Tools/Context，并将模型产生的 Tool Call 转换回 MCP 请求。



## 心得: Function Calling + MCP

Function Calling 是 LLM 调用“工具”的机制；MCP 是把这些工具标准化、发现化、连接化的一套协议。

模型需要具备 Tool/Function Calling 能力：知道什么时候应该调用工具、选择哪个工具、生成正确的参数。 

不需要专门训练成“懂 MCP”：MCP 是 Client ↔ Server 之间的通信协议，主要由 Claude Code、MCP Client、MCP Server 来处理。 

MCP Server 会把工具的 name / description / input schema 提供给 Client，Client 再把这些工具信息提供给模型。

The model is trained to understand tool/function schemas and generate tool calls with appropriate arguments. The agent provides the tool schemas to the model, receives the model's tool call, executes the corresponding tool (which may be exposed through MCP), and sends the tool result back to the model. The model then uses the tool result to generate the final response, which the agent returns to the user.


Model 本身并不执行 Function，而是经过训练后能够理解 Function/Tool Schema，并根据用户的 Prompt 判断是否需要调用某个 Function，同时生成对应的 Function 名称和参数。

调用 Model 时，Agent/Application 会把 Function Schema 一起传给 Model，Schema 通常包含 Function 的 Name、Description 和 Parameters。

Model 根据用户的 Prompt 和这些 Schema，选择合适的 Function，并返回一个结构化的 Tool Call，例如 Function Name 和对应的 Arguments。

然后，Agent/Application 根据 Model 返回的 Tool Call 实际执行 Function。这个 Function 可以是本地代码、API、数据库操作，也可以通过 MCP 暴露出来。Function 执行完成后，Agent 将 Function 的结果作为 Tool Result 再发送给 Model。

最后，Model 根据原始 Prompt、之前的 Tool Call 以及 Tool Result，生成最终的自然语言回答，Agent 再将这个最终结果返回给用户。

因此，Function Calling 的核心流程就是：Model 决定调用什么以及传什么参数，Agent 负责实际执行，Model 再根据执行结果生成最终回答。

## Documentation and Resources 

- [JSON-RPC](https://www.jsonrpc.org/)

- [MCP Transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)

- [MCP - CHN](https://zhuanlan.zhihu.com/p/27327515233)

## 实验

环境：

``` bash
# pip install "mcp[cli]" --break-system-packages

# pip install -U python-dotenv --break-system-packages

# pip install -U openai --break-system-packages
```

实验1: stdio

``` bash
# python3 mcp-client-simple.py # 自动运行子进程 python3 mcp-weather-server.py

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

实验2: http

``` bash
# python3 mcp-weather-server-http.py # 先运行

INFO:     Started server process [1524087]
INFO:     Waiting for application startup.
[09/07/26 16:51:39] INFO     StreamableHTTP session manager        streamable_http_manager.py:148
                             started                                                             
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

[09/07/26 16:52:33] INFO     Created new transport with session ID: 356ef86cad574bc5ad6b3b341692ee04                    streamable_http_manager.py:300
INFO:     127.0.0.1:41662 - "POST /mcp HTTP/1.1" 200 OK
INFO:     127.0.0.1:41662 - "POST /mcp HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:41662 - "POST /mcp HTTP/1.1" 200 OK
INFO:     127.0.0.1:41662 - "POST /mcp HTTP/1.1" 200 OK
[09/07/26 16:52:35] INFO     Terminating session: 356ef86cad574bc5ad6b3b341692ee04                                              streamable_http.py:814
INFO:     127.0.0.1:41662 - "DELETE /mcp HTTP/1.1" 200 OK


# python3 mcp-client-simple-http.py # 后运行

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

## MCPServer的主要功能

Tool 注册（@mcp.tool()），用装饰器把普通 Python 函数注册为 MCP 工具。支持标准的传输层，包括stdio、SSE等。

处理MCP协议的全部细节：
  - 初始化握手 — initialize / notifications/initialized
  - 工具发现 — tools/list（客户端查询有哪些工具）
  - 工具调用 — tools/call（客户端调用，SDK 路由到对应函数）
  - JSON-RPC 2.0 封包/解包 — 请求 ID 管理、错误格式

```
  Client                          Server (MCPServer)
    |                                 |
    |—— initialize ——————————————>|  SDK 处理握手
    |<—— result ————————————————|  返回 serverInfo
    |                                 |
    |—— tools/list —————————————————>|  SDK 扫描 @mcp.tool() 注册表
    |<—— [tool defs] ————————|  返回工具名+描述+schema
    |                                 |
    |—— tools/call —————————————————>|  SDK 解析 → 调用 get_weather(city)
    |<—— result ————————————————|  返回函数结果
```

## 支持的传输层协议

| 通信方式                | 典型场景                       | 当前状态                    |
| ------------------- | -------------------------- | ----------------------- |
| **stdio**           | Client 本机启动 MCP Server 子进程 | 推荐/主流的本地方式              |
| **Streamable HTTP** | Remote MCP Server          | **推荐的远程方式**             |
| **HTTP + SSE**      | 老的 Remote MCP Server       | **Legacy / Deprecated** |

## JSON-RPC 2.0

假设服务器提供一个函数：
```
add(a, b)
```

客户端想远程调用：
```
add(10, 20)
```

JSON-RPC 2.0 的请求：
``` json
{
  "jsonrpc": "2.0",
  "method": "add",
  "params": {
    "a": 10,
    "b": 20
  },
  "id": 1
}
```

服务器返回：
``` json
{
  "jsonrpc": "2.0",
  "result": 30,
  "id": 1
}
```