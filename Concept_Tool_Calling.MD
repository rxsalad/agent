# Tool Calling

Tool Calling is a mechanism by which large language models interact with external tools.
- Upon receiving a user request, the application/Agent first sends the user request to the model inference server via an API, along with the available tools and their parameter Schemas. 
- The model in the inference server determines whether a tool needs to be invoked based on the user request, and generates a structured Tool Call (including the tool name and parameters). 
- After the application/Agent receives the Tool Call, it is responsible for actually executing the tool, and submits the execution result as a Tool Result back to the model inference server via the API. 
- The model continues reasoning based on the Tool Result and generates the final response. The inference server returns the response to the application/Agent, which ultimately presents it to the user. 

Therefore, Tool Calling itself primarily addresses **"how the model declares which tool it needs to invoke and what parameters to pass"**, while actual tool execution, access control, and communication with external systems are handled by the application/Agent.

## The Role of Tool Schema in the API and Inference Pipeline

**API Layer**: The Tool Schema is sent to the model inference server as an independent `tools` array field in the API request.

**Token Layer**: The inference server uses the Chat Template to serialize `tools[]` into the token sequence that the model actually sees, typically located near the System Prompt. Thus, from the model's perspective, the Tool Schema is part of the Prompt/Context.

**Caching Layer**: Each Model API Call by a Coding Agent usually carries the same Tool Schema. Because the Tool Schema remains unchanged across multiple rounds of requests, it is an ideal Static Prefix — enabling Prefix Caching / KV Cache on the inference server avoids redundant computation.

Coding Agents typically have many Tools:
```
~15–40 tools      Common
~30–80 tools      Complex Agent
100+ tools        Multi-MCP / Enterprise Agent
```

The number of tools itself is not the most important factor — the token count of the Tool Schema is what matters:
```
50 tools × average 300 tokens ≈ 15K tokens
```

## Documentation and Resources

- [Introduction](https://zhuanlan.zhihu.com/p/1993366249963538339)

## Experiment

Environment:

``` bash
# pip install "mcp[cli]" --break-system-packages

# pip install -U python-dotenv --break-system-packages

# pip install -U openai --break-system-packages

# pip3 install httpx --break-system-packages
```

```
python3 ~/work/agent/client/client-tool-calling.py
```

## Complete Tool Calling Interaction Flow

### First API Request from the Application/Agent

The Agent sends the user request to the vLLM Server, along with the available Tool Schema.

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

Where:

messages: The user request and conversation history.
tools: The Tool Schema available for the model to invoke, including tool names, descriptions, and parameter definitions.
tool_choice="auto": Allows the model to decide on its own whether to invoke a Tool.

### vLLM Server Processes the First Request

Upon receiving the API request, the vLLM Server constructs the Prompt based on the Chat Template corresponding to the model.

The Tool Schema is serialized into a model-understandable Prompt by the Chat Template, typically as part of the system/tool context.

Based on the user request and the Tool Schema, the model generates a result. If the model determines that a Tool should be invoked, it generates the corresponding Tool Call; if no tool invocation is needed, it can generate a regular response directly.

For example, the model generates a Tool Call:

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

Note: `qwen3_xml` is not the component by which the model generates Tool Calls, but rather the Parser that vLLM uses to **parse** the Tool Call from the model's output. The model is responsible for generating Tool Call text in a format that conforms to XML; vLLM's Parser is responsible for parsing and converting it into an OpenAI-compatible `tool_calls` structure, which is then returned to the Agent.

### Application/Agent Executes the Tool

After receiving `tool_calls`, the Agent matches the tool name to a local tool implementation, and the Agent/Runtime actually executes the corresponding Python function, HTTP API, MCP Tool, or other external service.

### Second API Request from the Application/Agent

After the Tool execution completes, the Agent adds both the Tool Call and the Tool Result to the messages, and then initiates another API request to the vLLM Server.

The Tool Schema is typically still provided with the request, because it is part of the API request; the messages carry the previous Tool Call and Tool Result. Whether the client needs to repeat tools on every request depends on the API/Agent framework in use, but the typical calling pattern for OpenAI-compatible APIs is to provide them on every request.

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

### vLLM Server Processes the Second Request

The vLLM Server constructs the Prompt again based on the updated messages and the Chat Template. The model generates the final natural language response based on the Tool Result, and vLLM returns this response to the Agent.

```
The current weather in San Francisco is partly cloudy with a temperature of 65°F.
```

### Application/Agent Returns the Final Result to the User

After receiving the model-generated final response, the Agent presents it to the user, completing the entire Tool Calling flow.
