from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

messages1 = [
    {
        "role": "user",
        "content": 'Type "I love Qwen3.6" backwards',
    }
]

messages = [
    {
        "role": "user",
        "content": 'How to learn AI/ML? 100 words',
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B",
    messages=messages,
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={"top_k": 20},
)

print(80 * "-" + "> Output")
print(response.choices[0].message.content)