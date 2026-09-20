# MCP (Model Context Protocol)

MCP (Model Context Protocol) is a standardized protocol that enables AI Agents to interact with external tools, data sources, and services. It unifies tool **discovery**, **description**, and **invocation** into a standard interface, so Agents don't need custom integration code for every external system.

In one sentence: **MCP is the standard protocol connecting AI Agents to external tools/data/services — not the model's own Tool Calling API.**

---

## Core Relationship: MCP vs Function Calling

Understanding the relationship between MCP and Function Calling is key to grasping the whole system:

| Dimension | Function Calling | MCP |
|-----------|-----------------|-----|
| **Essence** | Mechanism by which the model invokes "tools" | Protocol for standardizing, discovering, and connecting tools |
| **Implemented by** | The model (trained to understand schemas and generate tool calls) | The communication layer between Client ↔ Server |
| **Does the model need to know it?** | Yes — understands schemas, decides when to call | No — translated by the MCP Client |

**Division of labor:**

- **Model**: Trained to understand tool schemas, decides whether to invoke a tool based on the user prompt, generates tool name and arguments
- **Agent / MCP Client**: Converts tool info from the MCP Server into the `tools[]` format the model API expects, executes the model's tool calls, sends results back to the model
- **MCP Server**: Exposes tools via standard interfaces (name / description / input schema), handles the actual business logic

---

## Foundation: JSON-RPC 2.0

MCP uses JSON-RPC 2.0 as its wire format. Understanding JSON-RPC means understanding MCP's message structure.

### Basic Call

Suppose there is a function `add(a, b)`, and the client wants to call `add(10, 20)`:

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "add",
  "params": { "a": 10, "b": 20 },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": 30,
  "id": 1
}
```

### Error Response

When a call fails, `result` is replaced by `error` with a standard error code:

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

### Notification

No `id` field = no response expected; used for one-way state signals:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

---

## The MCP Protocol: Three-Message Model

On top of JSON-RPC, MCP defines three core messages that cover the complete tool interaction flow:

```
  ① initialize      — Handshake: negotiate version and capabilities
  ② tools/list      — Discover: query available tools and their schemas
  ③ tools/call      — Invoke: pass arguments, get results
```

Full interaction sequence:

```
  Client                              Server
    |                                    |
    |—— initialize ━━━━━━━━━━━━━━━━━━━>|  Handshake: send protocol version, client info
    |<━━ result (serverInfo) ━━━━━━━━━|  Reply: server name, version, capabilities
    |                                    |
    |—— notifications/initialized ━━━>|  Handshake done (one-way notification)
    |                                    |
    |—— tools/list ━━━━━━━━━━━━━━━━━━━>|  Query: what tools are available?
    |<━━ [tool defs + schemas] ━━━━━━━|  Reply: tool names, descriptions, param schemas
    |                                    |
    |—— tools/call (name, args) ━━━━━>|  Invoke: execute a tool
    |<━━ result (tool output) ━━━━━━━━|  Reply: tool execution result
```

---

## MCP's Three Capabilities

An MCP Server can expose three kinds of capabilities to the Client:

| Capability | Purpose | Analogy |
|------------|---------|---------|
| **Tools** | Executable operations (e.g., weather lookup, search, database queries) | A function's API |
| **Resources** | Readable data (e.g., file contents, database records, logs) | A file path to data |
| **Prompts** | Reusable prompt templates (e.g., code review template, summary template) | Pre-written prompt templates |

Tools are the core capability carrier for Agents — the model invokes MCP-exposed Tools through the Tool Calling mechanism. Resources and Prompts provide data access and interaction template capabilities.

---

## Architecture & Transport Layer

### Typical Architecture

```
  ┌─────────────────────────────────────────────────┐
  │                 Agent / Application              │
  │                                                  │
  │   ┌─────────────┐    ┌────────────────────────┐  │
  │   │   LLM Model  │    │       MCP Client        │  │
  │   │             │    │  · Connect to MCP Server │  │
  │   │  Decide call│◄──►│  · Convert schemas      │  │
  │   │  Generate   │    │  · Execute tool calls   │  │
  │   │  Interpret  │    │  · Return tool results  │  │
  │   │  results    │    └──────────┬─────────────┘  │
  │   └─────────────┘               │                │
  └─────────────────────────────────┼────────────────┘
                                    │  JSON-RPC 2.0
                                    │  (stdio / HTTP)
                                    ▼
                          ┌──────────────────────┐
                          │     MCP Server        │
                          │  · Tools registry     │
                          │  · Resources exposed  │
                          │  · Prompt templates   │
                          └──────────────────────┘
```

### Transport Protocols

| Transport | Typical Scenario | Status |
|-----------|-----------------|--------|
| **stdio** | Client spawns MCP Server as a subprocess locally | ✅ Recommended (local) |
| **Streamable HTTP** | Remote MCP Server | ✅ **Recommended (remote)** |
| **HTTP + SSE** | Legacy remote MCP Server | ⚠️ Legacy / Deprecated |

stdio passes JSON-RPC messages over `stdin`/`stdout` (one JSON per line), while Streamable HTTP sends requests via HTTP POST with session management support.

---

## MCP Server Responsibilities

When using an MCP SDK (e.g., Python's `mcp.server.MCPServer`), the SDK handles all protocol-layer details:

- **Initialization handshake** — version negotiation via `initialize`, completion via `notifications/initialized`
- **Tool discovery** — `tools/list` scans tools registered with `@mcp.tool()`
- **Tool invocation** — `tools/call` routes to the matching function, handles parameter validation
- **JSON-RPC framing/unframing** — request ID management, error formats, notifications

Developers only need:

```python
from mcp.server import MCPServer

mcp = MCPServer("weather")

@mcp.tool()
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    return {"temperature": 65, "unit": "F", "condition": "Partly cloudy"}

if __name__ == "__main__":
    mcp.run(transport="stdio")            # Local subprocess
    # mcp.run(transport="streamable-http") # Remote HTTP
```

The `@mcp.tool()` decorator does the heavy lifting: derives the parameter schema from the function signature, extracts the description from the docstring, and registers the tool in the server's tool registry.

---

## Experiments

### Environment Setup

```bash
pip3 install "mcp[cli]" --break-system-packages
pip3 install -U python-dotenv --break-system-packages
pip3 install -U openai --break-system-packages
pip3 install httpx --break-system-packages
```

### Experiment 1: stdio Transport (handwritten JSON, no MCP SDK)

The client `mcp/mcp-client-simple.py` launches the server via `subprocess` and completes the MCP interaction using raw JSON-RPC:

```bash
python3 mcp/mcp-client-simple.py
# Auto-spawns subprocess: mcp/mcp-weather-server.py
```

Output:

```
Starting MCP server ...
Handshake (initialize) ...
  server: {'name': 'weather', 'version': ''}
Listing tools ...
  Tool: get_weather — Get the current weather for a city.

Calling tool: get_weather({'city': 'San Francisco'})
Tool result: {
  "temperature": 65,
  "unit": "F",
  "condition": "Partly cloudy"
}

Final answer: The current weather in San Francisco is partly cloudy with a temperature of 65°F.
```

### Experiment 2: Streamable HTTP Transport (handwritten JSON)

Start the server `mcp/mcp-weather-server-http.py` first, then connect with the client `mcp/mcp-client-simple-http.py`:

```bash
# Terminal 1: Start server
python3 mcp/mcp-weather-server-http.py

# Terminal 2: Run client
python3 mcp/mcp-client-simple-http.py
```

Output:

```
Connecting to MCP server http://127.0.0.1:8000/mcp ...
Handshake (initialize) ...
  server: {'name': 'weather', 'version': ''}
  session-id: 356ef86cad574bc5ad6b3b341692ee04
Listing tools ...
  Tool: get_weather — Get the current weather for a city.

Calling tool: get_weather({'city': 'San Francisco'})
Tool result: {
  "temperature": 65,
  "unit": "F",
  "condition": "Partly cloudy"
}

Final answer: The weather in San Francisco is currently partly cloudy with a temperature of 65°F.

Connection closed.
```

### Experiment 3: Using the MCP SDK (async + stdio)

`mcp/mcp-client-async.py` uses the `mcp` SDK's `ClientSession` and `stdio_client`, demonstrating production-grade usage:

```bash
python3 mcp/mcp-client-async.py
```

---

## Key Takeaways

1. **The model doesn't need to "understand MCP"** — MCP is a protocol between Client and Server. The model only needs to understand tool schemas (name / description / parameters), which the MCP Client fetches from the Server and converts into the model API's `tools[]` format.

2. **MCP solves the "connection" problem** — Before MCP, each tool required custom integration code. With MCP, tool implementors only need to build an MCP Server, and any MCP Client can discover and invoke it.

3. **MCP and Tool Calling are complementary** — Tool Calling tells the model "what tools you can call"; MCP lets tools be "discovered and connected in a standardized way". Together, they form the complete Agent tool-invocation pipeline.

---

## References

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Specification - Transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Chinese Analysis](https://zhuanlan.zhihu.com/p/27327515233)
