# mcp-client-simple.py
#
# 不依赖 MCP SDK，用纯 subprocess + JSON 手写 MCP 协议
# 全同步，零 async，方便理解 MCP 的本质
#
# MCP 协议 = JSON-RPC 2.0 over stdin/stdout
# 只需要三个消息：
#   1. initialize    — 握手
#   2. tools/list    — 询问对方有哪些工具
#   3. tools/call    — 调用工具


import json
import subprocess

from openai import OpenAI


# ============================================================
# 第 0 步：MCP 协议工具函数
#
# MCP 就是：往子进程 stdin 写一行 JSON，从 stdout 读一行 JSON
# ============================================================

rpc_id = 0  # 每发一条请求，id +1


def mcp_send(conn, method, params=None):
    """发一条 JSON-RPC 请求给 MCP server"""
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
# 第 1 步：启动 MCP server 子进程
# ============================================================

print("启动 MCP server ...")
conn = subprocess.Popen(
    ["python3", "/root/work/data/agent/mcp/mcp-weather-server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

# ============================================================
# 第 2 步：MCP 握手（initialize）
# ============================================================

print("握手 initialize ...")
resp = mcp_send(conn, "initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "simple-client", "version": "0.1.0"},
})
print(f"  server: {resp.get('result', {}).get('serverInfo', {})}")

# 告知 server 握手完成
conn.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
conn.stdin.flush()

# ============================================================
# 第 3 步：获取工具列表
# ============================================================

print("获取工具列表 ...")
resp = mcp_send(conn, "tools/list")
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

client = OpenAI()
MODEL = "Qwen/Qwen3.6-27B"


# ============================================================
# 第 5 步：Agent 循环
#
#   发消息给 Qwen
#     → Qwen 决定要不要调用工具
#       → 要：通过 MCP 调用，把结果送回 Qwen，继续循环
#       → 不要：输出最终答案，结束
# ============================================================

messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

while True:
    # --- 请求 Qwen ---
    response = client.chat.completions.create(
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

        # 通过 MCP 调用工具（就是一行 JSON 出去，一行 JSON 回来）
        resp = mcp_send(conn, "tools/call", {
            "name": name,
            "arguments": args,
        })

        # 提取结果文本
        result_text = resp["result"]["content"][0]["text"]
        print(f"工具返回: {result_text}")

        # 把结果送回 Qwen
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

# 清理
conn.terminate()
