# Agent Exploration

Hands-on exploration of the complete Agent stack, covering LLM inference, reasoning, orchestration, and standardized tool invocation, using Qwen3.6-27B and the Model Context Protocol (MCP).

```
                              User Prompt
                                   │
                                   ▼
              ┌──────────────────────────────────┐
              │               Agent              │
              │                                  │
              │  Orchestration / Reasoning       │
              │  • Planning                      │
              │  • Tool Selection                │
              │  • Context Management            │
              │  • MCP Client                    │
              └───────────────┬──────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
             LLM API                 MCP / JSON-RPC
                 │                         │
                 ▼                         ▼
        ┌─────────────────┐      ┌──────────────────┐
        │   vLLM + Qwen   │      │    MCP Server    │
        │                 │      │                  │
        │   Thinking      │      │  • get_weather   │
        │   Generation    │      │  • search        │
        └────────┬────────┘      │  • database      │
                 │               └────────┬─────────┘
                 │                        │
                 │ Tool                   │ Tool Result
                 │                        │
                 └────────────┐   ┌───────┘
                              ▼   ▼
                         ┌────────────┐
                         │   Agent    │
                         │  Next Step │
                         └────────────┘
```

## What's Here

Conceptual guides (bilingual) and runnable experiments covering:

- **Thinking Mode** — how a Chat Template toggle switches reasoning on/off with 2-3 tokens
- **Tool Calling** — the two-turn flow: model declares → agent executes → model answers
- **MCP Protocol** — standardized tool discovery and invocation via JSON-RPC 2.0

---

## Documentation

| Document | Description |
|----------|-------------|
| [Qwen Exploration](Qwen_Exploration.MD) · [中文](Qwen_Exploration_CHN.MD) | Model deep dive: architecture, Thinking Mode mechanics, CoT vs Reasoning vs Thinking, environment setup, vLLM pipeline |
| [Concept: Tool Calling](Concept_Tool_Calling.MD) · [中文](Concept_Tool_Calling_CHN.MD) | Full two-turn flow with request/response walkthroughs; schema caching and KV cache implications |
| [Concept: MCP](Concept_MCP.MD) · [中文](Concept_MCP_CHN.MD) | MCP protocol fundamentals: three-message model, JSON-RPC 2.0, tools/resources/prompts, transport options |
| [Qwen3.6-27B Quick Reference](Qwen_3.6_27B.MD) | vLLM container deployment, model weight layout, curl test commands |
| [Claude Code + Self-Hosted Models](Claude_Code_with_Self_Hosted_Models.MD) | Point Claude Code at a self-hosted vLLM endpoint via environment variables |

---

## Code

| Directory | Files | What it shows |
|-----------|-------|---------------|
| [`qwen/`](qwen/) | 4 scripts | Transformers-level calls: Pipeline API, AutoModel with thinking toggle, Chat Template diff verification, model architecture inspection |
| [`client/`](client/) | 3 scripts | OpenAI-compatible API usage: tool calling loop, multimodal image understanding, plain text requests |
| [`mcp/`](mcp/) | 5 scripts | MCP protocol: handwritten JSON-RPC over stdio and HTTP, SDK-based async client, mock weather server |

---

## Key Findings

### 1. Thinking Mode is a Chat Template toggle

Not a weight swap, not a code branch — the ON vs OFF difference is **2-3 tokens** in the input sequence. Same weights, same `forward()`, same computation graph.

```
ON  → template outputs: ...<thinking>\n          (open tag, model fills reasoning)
OFF → template outputs: ...<thinking>\n</thinking>\n\n  (closed tag, model skips)
```

See `qwen/low-automodel-thinking-prompt-diff.py` for the experiment.

### 2. MCP is transparent to the model

The model never sees MCP — it only understands `tools[]` schemas. The MCP Client sits between the model API and the MCP Server, translating in both directions:

- Server schemas → model `tools[]` format (for the model to choose from)
- Model tool calls → MCP `tools/call` requests (to invoke the server)

### 3. Tool Calling is a two-turn conversation

```
Turn 1:  User msg + tools[]  →  model returns tool_calls
         Agent executes the tool
Turn 2:  User msg + tool_calls + tool result  →  model generates final answer
```

The Tool Schema is sent in both turns, making it an ideal static prefix for KV cache reuse.

---

## Quick Start

```bash
# Run the Thinking Mode ON vs OFF diff experiment
python3 qwen/low-automodel-thinking-prompt-diff.py

# Run a tool calling loop (requires vLLM server + .env)
python3 client/client-tool-calling.py

# Run an MCP client (stdio transport)
python3 mcp/mcp-client-simple.py
```

---

## Hardware & Software

| Component | Value |
|-----------|-------|
| GPU | NVIDIA H200 141 GB (or H100 80 GB for dev container) |
| Model | Qwen/Qwen3.6-27B (~52 GB weights, ~132 GB VRAM at BF16) |
| Framework | vLLM (serving) / Transformers 5.16 (low-level) / PyTorch 2.14 |
| Python | 3.12 |

---

## License

[MIT](LICENSE)
