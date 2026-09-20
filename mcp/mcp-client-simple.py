# mcp-client-simple.py
#
# No MCP SDK dependency — handwritten MCP protocol using subprocess + JSON.
# Fully synchronous, zero async, for clarity on how MCP works under the hood.
#
# MCP protocol = JSON-RPC 2.0 over stdin/stdout
# Only three messages are needed:
#   1. initialize    — handshake
#   2. tools/list    — discover available tools
#   3. tools/call    — invoke a tool


import json
import subprocess

from openai import OpenAI


# ============================================================
# Step 0: MCP protocol helper functions
#
# MCP is simply: write one line of JSON to stdin, read one line from stdout
# ============================================================

rpc_id = 0  # Increment id for each request


def mcp_send(conn, method, params=None):
    """Send a JSON-RPC request to the MCP server."""
    global rpc_id
    rpc_id += 1
    request = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": method,
        "params": params or {},
    }
    conn.stdin.write(json.dumps(request) + "\n")
    conn.stdin.flush()
    return json.loads(conn.stdout.readline())


# ============================================================
# Step 1: Launch the MCP server subprocess
# ============================================================

print("Starting MCP server ...")
conn = subprocess.Popen(
    ["python3", "/home/ubuntu/work/agent/mcp/mcp-weather-server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

# ============================================================
# Step 2: MCP handshake (initialize)
# ============================================================

print("Handshake (initialize) ...")
resp = mcp_send(conn, "initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "simple-client", "version": "0.1.0"},
})
print(f"  server: {resp.get('result', {}).get('serverInfo', {})}")

# Notify server that handshake is complete
conn.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
conn.stdin.flush()

# ============================================================
# Step 3: List available tools
# ============================================================

print("Listing tools ...")
resp = mcp_send(conn, "tools/list")
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

client = OpenAI()
MODEL = "Qwen/Qwen3.6-27B"


# ============================================================
# Step 5: Agent loop
#
#   Send message to Qwen
#     → Qwen decides whether to call a tool
#       → Yes: call via MCP, send result back to Qwen, loop again
#       → No: output final answer, done
# ============================================================

messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

while True:
    # --- Request Qwen ---
    response = client.chat.completions.create(
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

        # Invoke via MCP (one JSON line out, one JSON line back)
        resp = mcp_send(conn, "tools/call", {
            "name": name,
            "arguments": args,
        })

        # Extract result text
        result_text = resp["result"]["content"][0]["text"]
        print(f"Tool result: {result_text}")

        # Send result back to Qwen
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

# Cleanup
conn.terminate()
