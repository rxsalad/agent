# mcp-client-simple-http.py
#
# No MCP SDK dependency — handwritten MCP protocol using httpx + JSON.
# Fully synchronous, zero async, for clarity on how MCP works under the hood.
#
# MCP protocol (HTTP transport) = JSON-RPC 2.0 over HTTP POST
# Only three messages are needed:
#   1. initialize    — handshake
#   2. tools/list    — discover available tools
#   3. tools/call    — invoke a tool


import json

from openai import OpenAI


# ============================================================
# Step 0: MCP protocol helper functions
#
# MCP (HTTP) is simply: POST one JSON line to the server, read one JSON line back
# ============================================================

rpc_id = 0  # Increment id for each request


def mcp_send(client, path, method, params=None, headers=None, *, include_resp=False):
    """Send a JSON-RPC request to the MCP server (HTTP transport).

    Returns the parsed JSON body by default.
    When include_resp=True, returns (body, response) for reading response headers.
    """
    global rpc_id
    rpc_id += 1
    request = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": method,
        "params": params or {},
    }
    resp = client.post(
        path, json=request, headers=headers or {}
    )
    resp.raise_for_status()
    if include_resp:
        return resp.json(), resp
    return resp.json()


def mcp_notify(client, path, method, params=None, headers=None):
    """Send a JSON-RPC notification (no id, no response expected)."""
    notification = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
    }
    resp = client.post(
        path, json=notification, headers=headers or {}
    )
    resp.raise_for_status()


# ============================================================
# Step 1: Connect to the MCP server (HTTP)
# ============================================================

import httpx

MCP_URL = "http://127.0.0.1:8000"
MCP_PATH = "/mcp"

print(f"Connecting to MCP server {MCP_URL}{MCP_PATH} ...")
client = httpx.Client(base_url=MCP_URL)

# Headers: client accepts JSON and SSE; server returns JSON
default_headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

# ============================================================
# Step 2: MCP handshake (initialize)
# ============================================================

print("Handshake (initialize) ...")
body, raw_resp = mcp_send(client, MCP_PATH, "initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "simple-client-http", "version": "0.1.0"},
}, headers=default_headers, include_resp=True)
print(f"  server: {body.get('result', {}).get('serverInfo', {})}")

# Extract session id from response headers (used in subsequent requests)
session_id = raw_resp.headers.get("mcp-session-id")
if session_id:
    default_headers["mcp-session-id"] = session_id
    print(f"  session-id: {session_id}")

# Notify server that handshake is complete
mcp_notify(client, MCP_PATH, "notifications/initialized",
           headers=default_headers)

# ============================================================
# Step 3: List available tools
# ============================================================

print("Listing tools ...")
resp = mcp_send(client, MCP_PATH, "tools/list",
                headers=default_headers)
mcp_tools = resp["result"]["tools"]
for t in mcp_tools:
    print(f"  Tool: {t['name']} — {t['description']}")

# Convert to OpenAI-compatible format
openai_tools = [{
    "type": "function",
    "function": {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["inputSchema"],
    },
} for t in mcp_tools]


# ============================================================
# Step 4: Initialize the LLM client
# ============================================================

from dotenv import load_dotenv
load_dotenv()

llm_client = OpenAI()
MODEL = "Qwen/Qwen3.6-27B"


# ============================================================
# Step 5: Agent loop
#
#   Send message to Qwen
#     → Qwen decides whether to call a tool
#       → Yes: call MCP server via HTTP, send result back to Qwen, loop again
#       → No: output final answer, done
# ============================================================

messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

while True:
    # --- Request Qwen ---
    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = response.choices[0].message

    # --- No tool needed — Qwen answers directly ---
    if not msg.tool_calls:
        print(f"\nFinal answer: {msg.content}")
        break

    # --- Qwen requests a tool call ---
    messages.append(msg)

    for tool_call in msg.tool_calls:
        name  = tool_call.function.name
        args  = json.loads(tool_call.function.arguments)

        print(f"\nCalling tool: {name}({args})")

        # Invoke MCP tool via HTTP (POST JSON, read JSON back)
        resp = mcp_send(client, MCP_PATH, "tools/call", {
            "name": name,
            "arguments": args,
        }, headers=default_headers)

        # Extract result text
        result_text = resp["result"]["content"][0]["text"]
        print(f"Tool result: {result_text}")

        # Send result back to Qwen
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

# Cleanup: if session id exists, notify server to close the session
if session_id:
    client.delete(MCP_PATH, headers=default_headers)
client.close()
print("\nConnection closed.")
