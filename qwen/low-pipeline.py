import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3.6-27B",
    device_map="auto",
)

print(80 * "-" + "> Output")
print(pipe("The capital of France is", max_new_tokens=20)[0]["generated_text"])