# Tool Calling

Tool Calling 是大语言模型与外部工具进行交互的一种机制。
- 收到用户请求后，应用/Agent 首先通过 API 向模型推理服务器发送用户请求，同时提供可用工具及其参数 Schema.
- 推理服务器中的模型根据用户请求判断是否需要调用某个工具，并生成结构化的 Tool Call（包括工具名称和参数）.
- 应用/Agent 收到 Tool Call 后负责实际执行工具，并将执行结果作为 Tool Result 再次通过 API 提交给模型推理服务器.
- 模型基于 Tool Result 继续推理并生成最终响应，推理服务器将响应返回给应用/Agent，最终由应用/Agent 呈现给用户。

因此，Tool Calling 本身主要解决的是 **"模型如何声明需要调用什么工具以及传递什么参数"**，而工具的实际执行、权限控制和与外部系统的通信由应用/Agent 负责。

## Tool Schema 在 API 与推理链路中的位置

**API 层**：Tool Schema 以 `tools` 数组作为 API 请求中的独立字段发送给模型推理服务器。

**Token 层**：推理服务器通过 Chat Template 将 `tools[]` 序列化成模型实际看到的 token sequence，通常位于 System Prompt 附近。因此从模型的视角来看，Tool Schema 就是 Prompt/Context 的一部分。

**缓存层**：Coding Agent 的每次 Model API Call 通常都会携带相同的 Tool Schema，但 Tool Schema 在多轮请求中保持不变，是一个理想的 Static Prefix —— 推理服务器开启 Prefix Caching / KV Cache 即可避免重复计算。

Coding Agent 通常有很多 Tools：
```
~15–40 tools      常见
~30–80 tools      复杂 Agent
100+ tools        多 MCP / 企业级 Agent
```

Tool 数量本身不是最重要的，Tool Schema 的 token 数量才是关键：
```
50 tools × 平均 300 tokens ≈ 15K tokens
```

## Documentation and Resources

- [Introduction](https://zhuanlan.zhihu.com/p/1993366249963538339)

## 实验

环境：

``` bash
# pip install "mcp[cli]" --break-system-packages

# pip install -U python-dotenv --break-system-packages

# pip install -U openai --break-system-packages

# pip3 install httpx --break-system-packages
```

```
python3 ~/work/agent/client/client-tool-calling.py
```

## Tool Calling 完整交互流程

### 应用/Agent第一次API请求

Agent 向 vLLM Server 发送用户请求，同时提供可用的 Tool Schema。

```
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in San Francisco?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

其中：

messages：用户请求和对话历史。
tools：可供模型调用的 Tool Schema，包括 Tool 名称、描述以及参数定义。
tool_choice="auto"：允许模型自行决定是否调用 Tool。

### vLLM Server 处理第一次请求

vLLM Server 收到 API 请求后，根据模型对应的 Chat Template 构造 Prompt。

Tool Schema 会被 Chat Template 序列化到模型能够理解的 Prompt 中，通常属于 system/tool context 的一部分。

模型根据用户请求和 Tool Schema 生成结果。如果模型判断需要调用 Tool，则生成相应的 Tool Call；如果不需要调用 Tool，则可以直接生成普通回答。

例如模型生成一个 Tool Call：

```
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\":\"San Francisco\"}"
      }
    }
  ]
}
```

注意：`qwen3_xml` 不是模型生成 Tool Call 的组件，而是 vLLM 用来 **解析** 模型输出的 Tool Call 的 Parser。模型负责生成符合 XML 格式的 Tool Call 文本，vLLM 的 Parser 负责将其解析并转换成 OpenAI-compatible 的 `tool_calls` 结构，再返回给 Agent。

### 应用/Agent 执行 Tool

Agent 收到 `tool_calls` 后，根据工具名称匹配到本地的工具实现，由 Agent/Runtime 实际执行对应的 Python 函数、HTTP API、MCP Tool 或其他外部服务。

### 应用/Agent 第二次 API 请求

Tool 执行完成后，Agent 将 Tool Call 和 Tool Result 都加入 messages，然后再次向 vLLM Server 发起 API 请求。

Tool Schema 通常仍然需要随请求提供，因为这是 API 请求的一部分；而 messages 则携带了之前的 Tool Call 和 Tool Result。具体是否需要客户端在每次请求都重复 tools，取决于所使用的 API/Agent 框架，但 OpenAI-compatible 的典型调用方式是每次请求都提供。

```
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in San Francisco?"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"San Francisco\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "content": "{\"temperature\":65,\"condition\":\"Partly cloudy\"}"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### vLLM Server 处理第二次请求

vLLM Server 再次根据更新后的 messages 和 Chat Template 构造 Prompt，模型基于 Tool Result 生成最终的自然语言回答，vLLM 将该响应返回给 Agent。

```
The current weather in San Francisco is partly cloudy with a temperature of 65°F.
```

### 应用/Agent 将最终结果返回给用户

Agent 收到模型生成的最终回答后，将其呈现给用户，完成整个 Tool Calling 流程。
