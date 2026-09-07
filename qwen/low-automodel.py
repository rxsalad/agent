import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_name = "Qwen/Qwen3.6-27B"

# AutoProcessor loads the input-processing components - tokenizer, chat template and multimodal proprocessing
processor = AutoProcessor.from_pretrained(model_name)

# Load the Qwen3.6-27B model weights and architecture.
model = AutoModelForMultimodalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "How to learn AI/ML? 100 words."}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    #enable_thinking=False,  # Thinking disabled
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
)

answer = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print(80 * "-" + "> Output")
print(answer)



""" Non Thinking
--------------------------------------------------------------------------------> Output
Start with Python and essential math: linear algebra, calculus, and statistics. Build fundamentals through free courses like Andrew Ng’s ML specialization. Master core libraries: NumPy, Pandas, and Scikit-learn. Practice relentlessly by building projects and competing on Kaggle. Progress to deep learning using TensorFlow or PyTorch, focusing on neural networks, NLP, and vision. Prioritize hands-on coding over passive watching. Read papers, follow communities, and document progress. Stay curious, embrace debugging, and iterate continuously. Consistency beats intensity—dedicate two hours daily, tackle real problems, and gradually contribute to open-source. AI/ML is a marathon; patience wins. Always track metrics, optimize models, and deploy endpoints.
"""

""" Thinking
--------------------------------------------------------------------------------> Output
Thinking Process:

1.  **Deconstruct the user's request:**
    *   Topic: How to learn AI/ML?
    *   Constraint: 100 words.

2.  **Brainstorm key components of learning AI/ML:**
    *   Math basics (Linear algebra, calculus, stats).
    *   Programming (Python).
    *   Libraries (Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch).
    *   Projects/Practice.
    *   Courses/Resources.
    *   Continuous learning.

3.  **Drafting content (Mental or rough text):**
    *   *Draft 1:* Start with Python and math basics like statistics and linear algebra. Take a structured course on Coursera or edX to grasp machine learning concepts. Master libraries like NumPy, Pandas, Scikit-learn, and TensorFlow. Build practical projects to apply knowledge; Kaggle competitions are excellent for practice. Read research papers and follow blogs to stay updated. Join communities for support. Remember, consistency is key. Experiment with different algorithms, debug code, and iterate. Finally, document your journey and contribute to open source. AI/ML is a vast field, so focus on fundamentals before diving into deep learning or specialized domains.

4.  **Counting words and refining (Iterative reduction):**
    *   *Draft 1 Word Count:* ~85 words.
    *   *Analysis:* It's close, but I need to ensure it's impactful and exactly or very close to 100 words without fluff. Let's check the count carefully.
    *   *Text:* "Start with Python and math basics like statistics and linear algebra. Take a structured course on Coursera or edX to grasp machine learning concepts. Master libraries like NumPy, Pandas, Scikit-learn, and TensorFlow. Build practical projects to apply knowledge; Kaggle competitions are excellent for practice. Read research papers and follow blogs to stay updated. Join communities for support. Remember, consistency is key. Experiment with different algorithms, debug code, and iterate. Finally, document your journey and contribute to open source. AI/ML is a vast field, so focus on fundamentals before diving into deep learning or specialized domains."
    *   *Word Count:*
        *   Start(1) with(2) Python(3) and(4) math(5) basics(6)
"""