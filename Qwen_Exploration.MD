# Qwen Exploration

Qwen3.6-27B is a 27B-parameter **dense** multimodal model open-sourced by Alibaba Cloud's Qwen series, with native vision-language understanding capabilities (joint image, video, and text input).

It is a model with reasoning capabilities built into a single unified checkpoint: you switch between **Thinking Mode** and **Non-Thinking Mode** by explicitly setting parameters (such as `enable_thinking` or a similar `thinking` boolean field in the chat template) in the inference request (not referring to a specific inference server's API request), rather than relying on soft prompt guidance.

## Quick Reference

| Property | Value |
|------|------|
| Parameters | 27B (dense, all activated at inference) |
| Architecture | Transformer (not MoE, no expert routing) |
| Multimodal | Joint image / video / text input |
| Inference Frameworks | vLLM / Transformers / SGLang |
| Deployment VRAM | ~132 GB (BF16, single H200) |
| Model Weights | ~52 GB (15 safetensors shards) |
| Thinking Toggle | Chat Template level (`enable_thinking`) |
| MoE in Same Series | Qwen3.6-35B-A3B (MoE architecture, 35B total / 3B active) |

### Mode Comparison

| Mode | Behavior | Use Cases | Latency |
|------|------|---------|------|
| **Thinking Mode** | Outputs `<thinking>` reasoning process + final answer | Math, programming, complex reasoning | High (~11K reasoning tokens) |
| **Non-Thinking Mode** | Skips reasoning, generates answer directly | General conversation, fast response | Low |

Both modes **share the same model weights and inference code** -- no need to switch between different checkpoints.

### Architecture Characteristics

As a dense model (not MoE), Qwen3.6-27B activates all parameters during inference (active parameters = total parameters), without the complexity of expert routing. This makes deployment and serving relatively straightforward, a practical choice for developers who want flagship-level capability at a mid-range scale while avoiding MoE deployment complexity.

According to the official introduction, it outperforms the previous generation's much larger open-source flagship Qwen3.5-397B-A17B (a MoE model with 397B total / 17B active parameters) on major coding benchmarks -- but this comparison should be read with caution: one is a dense model fully activating 27B, while the other MoE model only activates 17B per inference step. Directly comparing "parameter count" across architectures can be misleading.

### Practical Considerations

- Thinking Mode and Non-Thinking Mode typically require different sampling parameters (temperature, top_p, top_k, etc.); mixing them degrades output quality
- Some inference frameworks (such as certain versions of vLLM plugins) have compatibility issues when switching between the two modes; verify before deploying
- Model weights are open-sourced on Hugging Face / ModelScope, and online access is also available via Alibaba Cloud Model Studio API and Qwen Studio

---

## Core Concepts

Three progressive concepts -- understanding their relationship is key to understanding Thinking Mode:

| | **CoT** | **Reasoning Model** | **Thinking Mode** |
|---|---------|---------------------|-------------------|
| **Essence** | A prompting technique at inference time | Model capability (trained) | A functional toggle presented by the model or through an inference server API |
| **Requires training?** | No, "emerges" in large models | Requires SFT + RL | No, relies on the underlying Reasoning Model |
| **Who controls** | User (via prompt) | Model (autonomous reasoning) | Developer / User (via API parameters) |
| **Typical usage** | `"let's think step by step"` | o1 / o3 / DeepSeek-R1 | `enable_thinking: true` |

```
CoT          = A prompting technique, no specialized training needed
Reasoning    = A model capability, trained through SFT + RL
Thinking     = Packaging the Reasoning capability as an invokable product feature
```

### CoT (Chain of Thought)

Essentially a technique at inference time, first systematically proposed in a 2022 paper. The core idea is simple: do not let the model spit out the answer directly, but guide it to first write out intermediate reasoning steps before giving the final answer. The earliest implementation was as simple as adding "let's think step by step" to the prompt, or providing a few examples with reasoning processes (few-shot CoT).

This does not require specialized training -- as long as the base model is large enough and capable enough, this ability "emerges." It relies on the knowledge and pattern-matching capabilities learned during pretraining; CoT simply "guides" these capabilities out.

### Reasoning Model

This is about model-level capability, referring to models like OpenAI's o1/o3 and DeepSeek-R1 that are specifically trained for "deep reasoning." The difference from "using CoT to prompt an ordinary model" is that this long-chain reasoning capability is trained, not induced by prompts. The model autonomously learns behaviors such as exploration, backtracking, self-verification, and error correction -- things an ordinary model struggles to do consistently with prompts alone.

**Training Pipeline:**

1. Start with a pretrained base model
2. Use human-annotated or high-quality synthetic "question -> reasoning process -> answer" data for SFT cold start
3. Core training uses reinforcement learning (PPO, GRPO, etc.) on tasks with objectively verifiable answers -- math problems, coding tasks, logic puzzles where "right or wrong is immediately clear"
4. The reward signal typically only checks whether the final answer is correct, without supervising the specific content of intermediate reasoning, letting the model figure out "how to think" to more easily arrive at the correct answer

The training data is primarily auto-scored problems: math competition problems, programming problems (where test cases determine correctness), logic puzzles, etc., because they provide clean reward signals suitable for large-scale RL training.

### Thinking Mode / Extended Thinking

This is the concrete presentation of "Reasoning Model capability": before giving the final answer, the model first generates a segment of "thinking" content (visible or optionally hidden), and often with an adjustable token budget for the thinking. This relies on both the underlying reasoning capability (trained) and is also an interface-level feature that developers / users can actively invoke and configure. See the "Thinking Mode Mechanism Verification" section below for details.

---

## Thinking Mode Mechanism Verification

### Core Conclusion

Thinking Mode is a toggle at the **Chat Template level** -- it is not a switch of model weights or inference code. The same 27B weights, the same `model.generate()` call; the behavioral difference comes entirely from **differences in the input token sequence**.

### Experimental Method

`qwen/low-automodel-thinking-prompt-diff.py` takes the same message `"What is 2+2?"` and runs it through `apply_chat_template()` with `enable_thinking=False` and `enable_thinking=True`, then decodes and compares the token sequences:

```python
# OFF: template directly closes the thinking tag
inputs_off = processor.apply_chat_template(messages, ..., enable_thinking=False)
text_off = processor.decode(inputs_off["input_ids"][0])

# ON: template only opens the thinking tag, waits for the model to fill in reasoning
inputs_on = processor.apply_chat_template(messages, ..., enable_thinking=True)
text_on = processor.decode(inputs_on["input_ids"][0])
```

### Experimental Results

```
=== Non-Thinking Mode (token count: 19) ===
'<|im_start|>user\nWhat is 2+2?\n<|im_end|>\n<|im_start|>assistant\n<thinking>\n\n</thinking>\n\n<|im_end|>\n'

=== Thinking Mode (token count: 17) ===
'<|im_start|>user\nWhat is 2+2?\n<|im_end|>\n<|im_start|>assistant\n<thinking>\n'
```

**The only difference**: the Non-Thinking template adds a pair of `\n</thinking>\n` after `<thinking>\n` (the template closes the thinking tag on the model's behalf), while the Thinking template only outputs `\n` (leaving the tag open, waiting for the model to fill in the reasoning content and close it).

### Mechanism Diagram

```
enable_thinking=True  (Thinking Mode)
  Template outputs:  ...<|im_start|>assistant\n<thinking>\n
  Model sees open tag -> writes reasoning process -> writes </thinking> -> writes final answer

enable_thinking=False (Non-Thinking Mode)
  Template outputs:  ...<|im_start|>assistant\n<thinking>\n</thinking>\n\n
  Model sees closed tag -> skips reasoning -> writes final answer directly
```

### Key Comparison

| Dimension | Changes? |
|------|---------|
| Model weights | No -- same 27B set |
| Inference code | No -- `forward()` is identical, no `if thinking_mode` |
| `model.generate()` call | No -- parameters are identical |
| Tokens injected by Chat Template | Yes -- ON has one fewer pair of closing tags |
| input_ids the model sees | Yes -- 17 vs 19 tokens |
| Sampling parameters (temperature, etc.) | Warning -- official docs recommend different values per mode |

Both modes follow the **exact same computation graph**:

```
Thinking ON  input_ids -+
                        +---> token_embed() -> forward() layer 0~63 -> lm_head() -> sample
Thinking OFF input_ids -+
                        (weights, code, path all identical; only variable is those 2-3 tokens)
```

### Why This Design

- **Simplicity**: No need to maintain two sets of weights or two checkpoints
- **Flexibility**: Same deployment, switch on demand at the API level
- **Efficiency**: Non-Thinking mode skips the reasoning process, lower latency, fewer tokens consumed
- **Unified**: The model's reasoning capability is trained (SFT + RL); the template just guides whether it "should think"

### Full Call Chain

From client API request to model output, the `enable_thinking` parameter passes through three layers:

```
+--------------------------------------------------------------------+
|                        Client (API Request)                         |
|                                                                      |
|  POST /v1/chat/completions                                           |
|  {                                                                   |
|    "messages": [...],                                                |
|    "extra_body": {                                                   |
|      "chat_template_kwargs": {                                       |
|        "enable_thinking": true      <-- client specifies thinking    |
|      }                                                               |
|    }                                                                 |
|  }                                                                   |
+-----------------------------------------------------------------------+
|
v
+--------------------------------------------------------------------+
|                      vLLM Inference Server                          |
|                                                                      |
|  1. Parse API request, extract chat_template_kwargs                 |
|  2. Call Chat Template to serialize messages                        |
|     apply_chat_template(messages, enable_thinking=true)             |
|  3. Template injects <thinking> tag into the prompt                 |
|     -> generates input_ids (containing thinking guidance tokens)    |
|  4. Feed token sequence into model inference                        |
|     model.generate(input_ids, ...) <-- model is unaware of "API"    |
|                                                                      |
+-----------------------------------------------------------------------+
|
v
+--------------------------------------------------------------------+
|                        Model (27B Weights)                          |
|                                                                      |
|  Sees open <thinking> tag -> begins reasoning                        |
|  Generates: <thinking>reasoning</thinking> + final answer            |
|                                                                      |
+--------------------------------------------------------------------+
```

**Three-layer meaning:**

| Layer | Parameter | Controlled By | Role |
|------|------|--------|------|
| **API Layer** | `extra_body.chat_template_kwargs.enable_thinking` | Client specifies in request | Tells server "this request needs thinking" |
| **Template Layer** | Chat Template injects `<thinking>` tag | Server internal, template serialization phase | Translates thinking instruction into tokens the model understands |
| **Model Layer** | Input token sequence (a few more/fewer tokens) | Entirely determined by template | Sees open tag -> reasons; sees closed tag -> skips |

API parameter -> controls Chat Template -> changes the token sequence the model sees. The vLLM server is the middle layer (request -> template -> serialization -> inference -> return); weights and inference code never change, only those 2-3 tokens change. Regardless of which inference server or API format is used, it all converges to an `input_ids` tensor fed into the model.

### API Support vs Model Support

These are two **independent dimensions**:

| | **Model Supports** (has reasoning training) | **Model Does Not Support** (no reasoning training) |
|---|---|---|
| **API supports thinking** | Normal operation (e.g. Qwen3.6-27B + vLLM) | Warning -- template injects tokens, but model was not trained for reasoning; poor output quality |
| **API does not support thinking** | Warning -- model has capability but cannot activate via API; can manually construct tokens as a workaround | Neither works |

- **Model Support** = reasoning capability is **trained** (SFT + RL)
- **API Support** = the inference server provides a **parameter entry point**, translating client requests into correct token sequences

Qwen3.6-27B's situation: **Model has capability + vLLM has parameter entry + Chat Template knows how to inject** -- all three are required. Without API support, you can still manually construct tokens; without model training, it simply does not work.

### Usage Across Different Clients

| Scenario | How to Pass |
|------|---------|
| **OpenAI-compatible API** (curl / SDK) | `extra_body: {"chat_template_kwargs": {"enable_thinking": true}}` |
| **Transformers** (Python direct call) | `processor.apply_chat_template(..., enable_thinking=True)` |
| **vLLM Native** | `--enable-auto-tool-choice --reasoning-parser qwen3` |
| **Claude Code** (self-hosted) | `ANTHROPIC_MODEL` pointing to vLLM, with vLLM handling the template |

When using the Transformers library directly, there is no "API support or not" issue -- you are the API layer, the `input_ids` returned by `apply_chat_template()` goes directly into `model.generate()`, fully transparent. This is the approach used by `qwen/low-automodel.py` and `qwen/low-automodel-thinking-prompt-diff.py`.

---

## Environment Setup

### Hardware and Software Quick Reference

| Component | Host (Bare Metal) | Dev Container |
|------|---------------|---------------|
| GPU | NVIDIA H100 80GB HBM3 | NVIDIA H100 80GB HBM3 |
| Driver | NVIDIA 580.173.02 | NVIDIA 580.173.02 |
| CUDA | 13.1 (host) / 13.0 (container) | 13.0 |
| NCCL | Available | (2, 30, 7) |
| Docker | 29.1.3 | -- |
| containerd | 2.2.1 | -- |
| nvidia-container-toolkit | 1.19.1 | -- |
| Python | -- | 3.12.3 |
| PyTorch | -- | 2.14.0+cu130 |
| Transformers | -- | 5.16.1 |

### Host

```bash
# GPU status
nvidia-smi
# GPU: NVIDIA H100 80GB HBM3, 0 MiB / 81559 MiB

# CUDA compiler
nvcc --version
# CUDA 13.1, V13.1.115

# NVIDIA Container Toolkit
nvidia-ctk --version
# 1.19.1

# Docker
docker version
# Client/Server: 29.1.3, API 1.52
# containerd: 2.2.1, runc: 1.3.4

# Pull PyTorch image
docker pull pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel
# Image size: 36.3 GB (content 13.4 GB)
```

### Deploy Dev Container

```bash
docker run --rm -d \
  --name dev-server \
  --gpus all \
  --privileged \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel \
  -c 'set -ex; sleep infinity'

# View logs
docker logs -f dev-server

# Enter container
docker exec -it dev-server /bin/bash
```

Install VS Code CLI inside the container and start a tunnel:

```bash
# Inside container
apt update && apt install curl

# Install VS Code CLI
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
  -o vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz && \
  mv code /usr/local/bin/code && rm vscode_cli.tar.gz

# GitHub authentication
code tunnel user login --provider github

# Start tunnel
nohup code tunnel --accept-server-license-terms --name 001 &> output.log &
```

### Container Environment Verification

```bash
# GPU
nvidia-smi
# NVIDIA H100 80GB HBM3, 0 MiB / 81559 MiB

# CUDA
nvcc --version
# CUDA 13.0, V13.0.88

# PyTorch
python3 -c "import torch; print(torch.__version__)"
# 2.14.0+cu130

python3 -c "import torch; print(torch.cuda.is_available())"
# True

python3 -c "import torch; print(torch.cuda.nccl.version())"
# (2, 30, 7)

# Python
python3 --version
# Python 3.12.3
```

### Hugging Face Transformers

```bash
# Install dependencies
pip install -U transformers accelerate safetensors huggingface_hub --break-system-packages

# Verify
python -c "import transformers; print(transformers.__version__)"
# 5.16.1

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
# 2.14.0+cu130, True, 1
```

---

## vLLM

### Inference Pipeline

vLLM's inference core can be summarized as:

```
Scheduler -> KV Cache Manager -> Model Runner -> forward() -> Attention -> KV Cache -> Sampling
```

- **Scheduler**: determines which requests and which tokens need computation
- **KV Cache Manager**: manages and allocates KV Cache in blocks
- **Model Runner**: calls the model's `forward()`
- **Attention**: reads historical K/V from KV Cache, writes newly generated K/V back to Cache
- **Sampling**: generates the next token

In a TP=8 scenario, the KV Cache is distributed across multiple GPUs. Components like MooncakeConnector can intervene at the KV Cache layer, transferring Prefill-produced KV Cache between different workers/GPUs via RDMA and other methods.

### Model Parameter Changes vs Architecture Changes

If a new model only changes parameter scale, number of layers, hidden size, vocab size, etc., but the model architecture and KV Cache / Attention mechanism remain the same, vLLM's overall inference framework does not need to be redesigned. Existing model implementations are typically reusable -- just load different parameters.

Only when a new model introduces new Attention mechanisms, KV/State Cache, MoE, Vision Encoder, MTP, or other architectural or execution mechanisms is it necessary to modify the corresponding model `forward()`, Attention/KV Cache, or other runtime components.

### Framework Ecosystem Adaptation

If the Qwen series introduces an entirely new model architecture, to make it work correctly in the mainstream ecosystem, you typically need to adapt:

- **Hugging Face Transformers**: provide standard model, configuration, Processor, and `forward()` implementation
- **vLLM / SGLang**: adapt model execution, Attention, KV/State Cache, Tensor Parallel, etc. for their respective inference runtimes to achieve high-performance inference

The specific code does not necessarily have to be maintained by Alibaba; it can be jointly completed by the Qwen team, framework teams, and the community.

---

## Testing and Verification

Run in the Dev Container:

```bash
# Lowest level: Pipeline API
python qwen/low-pipeline.py

# AutoModel + AutoProcessor (with Thinking toggle)
python qwen/low-automodel.py

# Model architecture exploration (config, layers, parameters, device distribution)
python qwen/low-model-arch.py

# Thinking Mode ON vs OFF -- Chat Template difference verification
python qwen/low-automodel-thinking-prompt-diff.py
```

### Thinking Mode Verification

`qwen/low-automodel-thinking-prompt-diff.py` confirms: the Thinking toggle is a **Chat Template behavior**, not a change in model weights or inference code. See the "Thinking Mode Mechanism Verification" section above for details.

---

## References

- [CoT (Chain of Thought)](https://www.nvidia.cn/glossary/cot-prompting/)
- [Qwen Official Blog](https://qwen.ai/blog?id=qwen3.6-27b)
- [Hugging Face - Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Transformers Model Doc](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen Code](https://github.com/QwenLM/qwen-code)
- [vLLM Recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [Deep Learning Technical Analysis (CN)](http://eadst.com/blog/325)
