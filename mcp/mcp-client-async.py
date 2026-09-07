# mcp-client.py

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# OpenAI-compatible client -> vLLM -> Qwen
# ============================================================

client = OpenAI()

MODEL = "Qwen/Qwen3.6-27B"



# ============================================================
# MCP server configuration
#
# The MCP client launches mcp-weather-server.py as a subprocess.

# MCP communication:
#
#     MCP Client
#          |
#       stdin/stdout
#          |
#          v
#     MCP Server
# ============================================================

# 定义要启动什么
server_params = StdioServerParameters(
    command="python3",
    args=["mcp-weather-server.py"],
)


# ============================================================
# Qwen / vLLM parameters
# ============================================================

qwen_extra_body = {
    "top_k": 20,

    "chat_template_kwargs": {
        # Disable Qwen thinking/reasoning for this example.
        "enable_thinking": False,
    },
}


# ============================================================
# Convert MCP tools -> OpenAI-compatible tools
# ============================================================

def convert_mcp_tools(mcp_tools):

    tools = []

    for tool in mcp_tools:

        tools.append(
            {
                "type": "function",

                "function": {
                    "name": tool.name,

                    "description": (
                        tool.description
                        if tool.description
                        else ""
                    ),

                    # IMPORTANT:
                    #
                    # Your installed MCP SDK uses
                    # input_schema, NOT inputSchema.
                    #
                    "parameters": tool.input_schema,
                },
            }
        )

    return tools


# ============================================================
# Convert MCP result -> string/object for OpenAI tool message
# ============================================================

def extract_mcp_result(result):

    # --------------------------------------------------------
    # Prefer structured content if available
    # --------------------------------------------------------

    if getattr(result, "structuredContent", None) is not None:

        return result.structuredContent


    # --------------------------------------------------------
    # Fall back to textual content
    # --------------------------------------------------------

    text_parts = []

    for content in result.content:

        if hasattr(content, "text"):

            text_parts.append(
                content.text
            )


    return "\n".join(text_parts)


# ============================================================
# Main
# ============================================================

async def main():

    # ========================================================
    # Start MCP server
    # ========================================================
  
    # 启动子进程
    async with stdio_client(server_params) as (
        read,
        write,
    ):

        # ====================================================
        # Create MCP client session
        # ====================================================

        async with ClientSession(
            read,
            write,
        ) as mcp_client:

            # ------------------------------------------------
            # Initialize MCP connection
            # ------------------------------------------------

            await mcp_client.initialize()


            print(
                "\n========== CONNECTED TO MCP SERVER =========="
            )


            # =================================================
            # Discover MCP tools
            # =================================================

            mcp_tools_result = (
                await mcp_client.list_tools()
            )


            print(
                "\n========== MCP TOOLS =========="
            )


            for tool in mcp_tools_result.tools:

                print(
                    "Name:",
                    tool.name,
                )

                print(
                    "Description:",
                    tool.description,
                )

                print(
                    "Input schema:",
                    tool.input_schema,
                )


            # =================================================
            # Convert MCP tools -> OpenAI tools
            # =================================================

            tools = convert_mcp_tools(
                mcp_tools_result.tools
            )


            print(
                "\n========== OPENAI TOOLS =========="
            )

            print(
                json.dumps(
                    tools,
                    indent=2,
                    default=str,
                )
            )


            # =================================================
            # Initial conversation
            # =================================================

            messages = [
                {
                    "role": "user",

                    "content": (
                        "What's the weather like "
                        "in San Francisco?"
                    ),
                }
            ]


            # =================================================
            # Agent loop
            #
            # Model
            #   |
            #   +--> tool call --> MCP
            #   |                  |
            #   |                  +--> result
            #   |                       |
            #   <-----------------------+
            #   |
            #   +--> final answer
            #
            # The loop allows the model to make multiple
            # tool calls if necessary.
            # =================================================

            while True:

                print(
                    "\n========== MODEL REQUEST =========="
                )


                # ------------------------------------------------
                # Send conversation to Qwen through vLLM
                # ------------------------------------------------

                response = (
                    client.chat.completions.create(
                        model=MODEL,

                        messages=messages,

                        tools=tools,

                        tool_choice="auto",

                        max_tokens=1024,

                        temperature=1.0,

                        top_p=0.95,

                        extra_body=qwen_extra_body,
                    )
                )


                message = (
                    response.choices[0].message
                )


                print(
                    "\n========== MODEL RESPONSE =========="
                )

                print(message)


                # =================================================
                # No tool call
                #
                # Qwen decided it can answer directly.
                # =================================================

                if not message.tool_calls:

                    print(
                        "\n========== FINAL ANSWER =========="
                    )

                    print(
                        message.content
                    )

                    break


                # =================================================
                # Qwen requested one or more tools
                #
                # IMPORTANT:
                # Preserve the assistant message containing
                # the tool_calls.
                # =================================================

                messages.append(
                    message
                )


                # =================================================
                # Execute every requested tool
                # =================================================

                for tool_call in message.tool_calls:

                    function_name = (
                        tool_call.function.name
                    )


                    # ------------------------------------------------
                    # Parse arguments generated by Qwen
                    # ------------------------------------------------

                    try:

                        function_arguments = (
                            json.loads(
                                tool_call.function.arguments
                            )
                        )

                    except json.JSONDecodeError as e:

                        print(
                            "\nERROR: Invalid tool arguments"
                        )

                        print(
                            tool_call.function.arguments
                        )

                        raise e


                    print(
                        "\n========== MCP TOOL CALL =========="
                    )

                    print(
                        "Tool:",
                        function_name,
                    )

                    print(
                        "Arguments:",
                        function_arguments,
                    )


                    # =================================================
                    # Call MCP server
                    # =================================================

                    result = (
                        await mcp_client.call_tool(
                            function_name,

                            arguments=function_arguments,
                        )
                    )


                    print(
                        "\n========== MCP RESULT =========="
                    )

                    print(result)


                    # =================================================
                    # Extract MCP result
                    # =================================================

                    tool_result = (
                        extract_mcp_result(
                            result
                        )
                    )


                    print(
                        "\n========== TOOL RESULT =========="
                    )

                    print(
                        json.dumps(
                            tool_result,
                            indent=2,
                            default=str,
                        )
                    )


                    # =================================================
                    # Send MCP result back to Qwen
                    # =================================================

                    messages.append(
                        {
                            "role": "tool",

                            "tool_call_id": (
                                tool_call.id
                            ),

                            "content": json.dumps(
                                tool_result,
                                default=str,
                            ),
                        }
                    )


                # =================================================
                # Loop back to Qwen
                #
                # Qwen now sees:
                #
                #   user
                #   assistant tool_call
                #   tool result
                #
                # and can generate the final answer or request
                # another tool.
                # =================================================

                print(
                    "\n========== TOOL RESULTS SENT TO MODEL =========="
                )

                print(
                    json.dumps(
                        messages,
                        indent=2,
                        default=str,
                    )
                )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )