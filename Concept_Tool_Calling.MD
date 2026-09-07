
# Tool Calling

Tool Calling 是大语言模型与外部工具进行交互的一种机制。 应用首先通过 API 向模型提供可用工具及其参数 Schema，模型根据用户请求判断是否需要调用某个工具，并生成结构化的 Tool Call，包括工具名称和参数；应用程序收到 Tool Call 后负责实际执行工具，并将执行结果作为 Tool Result 返回给模型，模型再基于结果继续推理并生成最终响应。因此，Tool Calling 本身主要解决的是 “模型如何声明需要调用什么工具以及传递什么参数”，而工具的实际执行、权限控制和与外部系统的通信通常由 Agent/应用负责。

Tool 在 API 层是独立字段。

Tool Schema 通常不是直接放在 system 字段里。

Inference server 会通过 Chat Template 把 tools[] 序列化成模型实际看到的 token sequence。
所以从模型/inference 的角度，Tool Schema 可以视为 Prompt/Context 的一部分。

Coding Agent 通常有很多 Tools:
```
~15–40 tools      常见
~30–80 tools      复杂 Agent
100+ tools        多 MCP / 企业级 Agent
```

Tool 数量本身不是最重要的，Tool Schema 的 token 数量更重要。
```
50 tools × 平均 300 tokens ≈ 15K tokens
```

每个 Model API Call 通常都会带当前 Tools, 所以从 API payload 看，Tool Schema 很可能每个 request 都会出现。

Tool Schema在多个 request 中保持不变，那么它实际上是一个非常好的 Static Prefix。因此 Prefix Caching 可以避免重复计算 Tool Schema。

Tool 在 API 层是独立的 tools[]；经过 Chat Template 后，Tool Schema 通常进入模型的 Prompt Token Sequence，往往位于 System Prompt 附近；Coding Agent 的每次 Model Call 通常都会携带这些 Tool Schema，但通过 Prefix/KV Cache 可以避免每次重新计算。

## Documentation and Resources 

- [Introudction](https://zhuanlan.zhihu.com/p/1993366249963538339)

## 实验

环境：

``` bash
# pip install "mcp[cli]" --break-system-packages

# pip install -U python-dotenv --break-system-packages

# pip install -U openai --break-system-packages
```

```
python client/client-tool-calling.py
```

## 交互过程

### Agent第一次API请求

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

此时，qwen3_xml 不是模型生成 Tool Call 的组件，而是 vLLM 用来解析模型生成的 Tool Call 的 Parser。

vLLM 将模型生成的 Tool Call 解析后，转换成 OpenAI-compatible tool_calls，然后返回给 Agent。

### Agent 执行 Tool


Agent 收到 tool_calls 后，根据其中的信息找到对应的实际 Tool.

然后由 Agent/runtime 实际执行 Python 函数、HTTP API、MCP Tool 或其他外部 Tool。

### Agent 第二次 API 请求

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

vLLM Server 再次根据更新后的 messages 和 Chat Template 构造 Prompt, vLLM 将模型生成的结果返回给 Agent。