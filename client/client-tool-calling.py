import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

MODEL = "Qwen/Qwen3.6-27B"

# ============================================================
# 2. Define the actual Python function
#
# The model DOES NOT execute this function.
# Your Agent/application executes it.
# ============================================================

def get_weather(city: str):
    """
    Get weather information for a city.

    In a real application, this function would call
    an actual weather API.
    """

    # Mock result for demonstration
    weather_data = {
        "San Francisco": {
            "temperature": 65,
            "unit": "F",
            "condition": "Partly cloudy",
        },
        "New York": {
            "temperature": 72,
            "unit": "F",
            "condition": "Sunny",
        },
        "Seattle": {
            "temperature": 58,
            "unit": "F",
            "condition": "Rainy",
        },
    }

    return weather_data.get(
        city,
        {
            "temperature": None,
            "unit": "F",
            "condition": "Unknown",
        },
    )


# ============================================================
# 3. Describe the function to the model
#
# This is the Function Schema.
# The model sees this schema and learns what tool it can call.
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get weather for.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


# ============================================================
# 4. Initial conversation
# ============================================================

messages = [
    {
        "role": "user",
        "content": "What's the weather like in San Francisco?",
    }
]


def print_api_prompt(messages):
    print("\n========== PROMPT SENT TO API ==========")
    print(json.dumps(messages, indent=2, default=str))


# ============================================================
# 5. Ask Qwen whether it wants to call a tool
# ============================================================

print_api_prompt(messages)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",

    # Normal generation parameters
    max_tokens=1024,
    temperature=1.0,
    top_p=0.95,

    # Qwen-specific parameters
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    },
)


message = response.choices[0].message


print("\n========== MODEL RESPONSE ==========")
print(message)


# ============================================================
# 6. Check whether the model requested a tool call
# ============================================================

if message.tool_calls:

    print("\n========== TOOL CALLS ==========")

    # IMPORTANT:
    # Add the model's tool-call message to the conversation.
    messages.append(message)

    for tool_call in message.tool_calls:

        function_name = tool_call.function.name
        function_arguments = json.loads(
            tool_call.function.arguments
        )

        print("Function:", function_name)
        print("Arguments:", function_arguments)

        # ====================================================
        # 7. Execute the actual Python function
        # ====================================================

        if function_name == "get_weather":

            result = get_weather(
                **function_arguments
            )

        else:

            result = {
                "error": f"Unknown function: {function_name}"
            }


        print("Result:", result)


        # ====================================================
        # 8. Send the function result back to the model
        # ====================================================

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )


    # ========================================================
    # 9. Ask Qwen to generate the final answer
    # ========================================================

    print_api_prompt(messages)

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",

        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,

        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    )


    print("\n========== FINAL ANSWER ==========")

    print(
        final_response.choices[0].message.content
    )


else:

    # Model decided that it didn't need a tool.
    print("\n========== FINAL ANSWER ==========")

    print(message.content)


