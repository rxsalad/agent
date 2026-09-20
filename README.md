# Qwen3.6-27B Exploration

Hands-on exploration of the **Qwen3.6-27B** model — a 27B dense multimodal model with built-in reasoning capabilities — covering deployment, tool calling, MCP (Model Context Protocol), and low-level inference mechanics.

## Overview

This repository documents and experiments with the full stack of running a self-hosted LLM as an Agent:

- **Model layer** — Qwen3.6-27B architecture, thinking mode mechanics, Transformers API usage
- **Serving layer** — vLLM deployment, dev container setup, hardware configuration
- **Tool Calling** — How the model declares tool invocations, the two-turn API flow, and schema caching
- **MCP Protocol** — Standardized tool discovery and invocation via JSON-RPC 2.0 (stdio & HTTP transports)
- **Agent integration** — Claude Code with self-hosted models, OpenAI-compatible client code

## Documentation

| File | Description |
|------|-------------|
| [Concept_MCP](Concept_MCP.MD) / [CHN](Concept_MCP_CHN.MD) | MCP protocol fundamentals: three-message model, JSON-RPC 2.0, tools/resources/prompts, architecture, and transport options |
| [Concept_Tool_Calling](Concept_Tool_Calling.MD) / [CHN](Concept_Tool_Calling_CHN.MD) | Tool Calling mechanics: API requests, vLLM processing, agent execution, and the complete two-turn flow |
| [Qwen_Exploration](Qwen_Exploration.MD) / [CHN](Qwen_Exploration_CHN.MD) | Deep dive into Qwen3.6-27B: thinking mode verification (it's a Chat Template toggle), CoT vs Reasoning vs Thinking, environment setup, vLLM pipeline |
| [Qwen_3.6_27B](Qwen_3.6_27B.MD) | Quick reference: vLLM container deployment, model weight layout, and curl test commands |
| [Claude_Code_with_Self_Hosted_Models](Claude_Code_with_Self_Hosted_Models.MD) | How to point Claude Code at a self-hosted vLLM + Qwen endpoint via environment variables |

## Code Examples

| Directory | Contents |
|-----------|----------|
| [`mcp/`](mcp/) | MCP protocol experiments — handwritten JSON-RPC (stdio & HTTP) and SDK-based async client, plus a weather mock server |
| [`client/`](client/) | OpenAI-compatible API clients — tool calling loop, multimodal image understanding, and plain text-only requests |
| [`qwen/`](qwen/) | Transformers low-level scripts — AutoModel with thinking toggle, Chat Template diff verification, model architecture inspection, and Pipeline API |

