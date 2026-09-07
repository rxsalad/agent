# mcp-client-simple-http.py
#
# 不依赖 MCP SDK，用纯 httpx + JSON 手写 MCP 协议
# 全同步，零 async，方便理解 MCP 的本质
#
# MCP 协议（HTTP 传输）= JSON-RPC 2.0 over HTTP POST
# 只需要三个消息：
#   1. initialize    — 握手
#   2. tools/list    — 询问对方有哪些工具
#   3. tools/call    — 调用工具


import json

from openai import OpenAI


# ============================================================
# 第 0 步：MCP 协议工具函数
#
# MCP（HTTP）就是：POST 一行 JSON 到服务器，读一行 JSON 回来
# ============================================================

rpc_id = 0  # 每发一条请求，id +1


def mcp_send(client, path, method, params=None, headers=None, *, include_resp=False):
    """发一条 JSON-RPC 请求给 MCP server（HTTP 传输）

    默认返回解析后的 JSON body。
    当 include_resp=True 时，返回 (body, response)，方便读取 response headers。
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
    """发一条 JSON-RPC 通知（不带 id，不需要响应）"""
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
# 第 1 步：连接 MCP server（HTTP）
# ============================================================

import httpx

MCP_URL = "http://127.0.0.1:8000"
MCP_PATH = "/mcp"

print(f"连接 MCP server {MCP_URL}{MCP_PATH} ...")
client = httpx.Client(base_url=MCP_URL)

# 请求头：client 需要接受 JSON 和 SSE；server 返回 JSON
default_headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

# ============================================================
# 第 2 步：MCP 握手（initialize）
# ============================================================

print("握手 initialize ...")
body, raw_resp = mcp_send(client, MCP_PATH, "initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "simple-client-http", "version": "0.1.0"},
}, headers=default_headers, include_resp=True)
print(f"  server: {body.get('result', {}).get('serverInfo', {})}")

# 从响应 header 获取 session id（后续请求携带）
session_id = raw_resp.headers.get("mcp-session-id")
if session_id:
    default_headers["mcp-session-id"] = session_id
    print(f"  session-id: {session_id}")

# 告知 server 握手完成
mcp_notify(client, MCP_PATH, "notifications/initialized",
           headers=default_headers)

# ============================================================
# 第 3 步：获取工具列表
# ============================================================

print("获取工具列表 ...")
resp = mcp_send(client, MCP_PATH, "tools/list",
                headers=default_headers)
mcp_tools = resp["result"]["tools"]
for t in mcp_tools:
    print(f"  工具: {t['name']} — {t['description']}")

# 转为 OpenAI 格式
openai_tools = [{
    "type": "function",
    "function": {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["inputSchema"],
    },
} for t in mcp_tools]


# ============================================================
# 第 4 步：启动 LLM 客户端
# ============================================================

from dotenv import load_dotenv
load_dotenv()

llm_client = OpenAI()
MODEL = "Qwen/Qwen3.6-27B"


# ============================================================
# 第 5 步：Agent 循环
#
#   发消息给 Qwen
#     → Qwen 决定要不要调用工具
#       → 要：通过 HTTP 调用 MCP server，把结果送回 Qwen，继续循环
#       → 不要：输出最终答案，结束
# ============================================================

messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

while True:
    # --- 请求 Qwen ---
    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = response.choices[0].message

    # --- Qwen 不需要工具，直接回答 ---
    if not msg.tool_calls:
        print(f"\n最终答案: {msg.content}")
        break

    # --- Qwen 请求调用工具 ---
    messages.append(msg)

    for tool_call in msg.tool_calls:
        name  = tool_call.function.name
        args  = json.loads(tool_call.function.arguments)

        print(f"\n调用工具: {name}({args})")

        # 通过 HTTP 调用 MCP 工具（POST JSON，读 JSON 回来）
        resp = mcp_send(client, MCP_PATH, "tools/call", {
            "name": name,
            "arguments": args,
        }, headers=default_headers)

        # 提取结果文本
        result_text = resp["result"]["content"][0]["text"]
        print(f"工具返回: {result_text}")

        # 把结果送回 Qwen
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

# 清理：如果有 session id，通知 server 关闭 session
if session_id:
    client.delete(MCP_PATH, headers=default_headers)
client.close()
print("\n连接已关闭。")
