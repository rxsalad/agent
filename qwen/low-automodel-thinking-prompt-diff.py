import torch
from transformers import AutoProcessor

model_name = "Qwen/Qwen3.6-27B"
processor = AutoProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is 2+2?"}
        ],
    }
]

# ---- Non-Thinking Mode ----
inputs_off = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)
text_off = processor.decode(inputs_off["input_ids"][0])

# ---- Thinking Mode ----
inputs_on = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=True,
)
text_on = processor.decode(inputs_on["input_ids"][0])

# ---- Compare ----
print("=== Non-Thinking Mode (token count: {}) ===".format(len(inputs_off["input_ids"][0])))
print(repr(text_off))
print()
print("=== Thinking Mode (token count: {}) ===".format(len(inputs_on["input_ids"][0])))
print(repr(text_on))
print()

# ---- Show the diff ----
# Split into lines and find differing region
off_lines = text_off.split("\n")
on_lines = text_on.split("\n")

print("=== Differences ===")
for i, (a, b) in enumerate(zip(off_lines, on_lines)):
    if a != b:
        print(f"  Line {i}:")
        print(f"    OFF: {repr(a)}")
        print(f"    ON:  {repr(b)}")

# Show extra lines beyond the common length
max_len = max(len(off_lines), len(on_lines))
for i in range(min(len(off_lines), len(on_lines)), max_len):
    if i < len(on_lines):
        print(f"  Line {i} (extra in Thinking ON):")
        print(f"    ON:  {repr(on_lines[i])}")
    if i < len(off_lines):
        print(f"  Line {i} (extra in Thinking OFF):")
        print(f"    OFF: {repr(off_lines[i])}")



"""
=== Non-Thinking Mode (token count: 19) ===
'<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

=== Thinking Mode (token count: 17) ===
'<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n'

=== Differences ===
  Line 5 (extra in Thinking OFF):
    OFF: '</think>'
  Line 6 (extra in Thinking OFF):
    OFF: ''
  Line 7 (extra in Thinking OFF):
    OFF: ''

"""
