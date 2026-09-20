# Qwen Exploration

Qwen3.6-27B 是阿里云 Qwen 系列开源的一个 27B 参数**稠密（dense）**多模态模型，具备原生的视觉语言理解能力（图像、视频与文本联合输入）。

它是一个统一 checkpoint 中同时内置 reasoning 能力的模型：通过在推理请求中（不是指具体推理服务器的API请求）显式设置参数（如聊天模板中的 `enable_thinking` 或类似的 `thinking` 布尔字段）来切换 **Thinking Mode** 与 **Non-Thinking Mode**，而非依赖 prompt 软性引导。

## 参数速查

| 属性 | 值 |
|------|------|
| 参数量 | 27B（稠密，推理时全部激活） |
| 架构 | Transformer（非 MoE，无专家路由） |
| 多模态 | 图像 / 视频 / 文本联合输入 |
| 推理框架 | vLLM / Transformers / SGLang |
| 部署显存 | ~132 GB（BF16，单张 H200） |
| 模型权重 | ~52 GB（15 个 safetensors 分片） |
| Thinking 开关 | Chat Template 层面（`enable_thinking`） |
| 同系列 MoE | Qwen3.6-35B-A3B（MoE 架构，35B 总参 / 3B 激活） |

### 模式对比

| 模式 | 行为 | 适用场景 | 延迟 |
|------|------|---------|------|
| **Thinking Mode** | 输出 `<thinking>` 推理过程 + 最终答案 | 数学、编程、复杂推理 | 高（~11K 推理 token） |
| **Non-Thinking Mode** | 跳过推理，直接生成答案 | 通用对话、快速响应 | 低 |

两种模式**共享同一套模型权重及推理代码**，无需切换到不同 checkpoint。

### 架构特点

作为稠密模型（而非 MoE），Qwen3.6-27B 在推理时激活全部参数（active parameters = total parameters），不涉及专家路由（expert routing）的复杂性，因此部署和 serving 相对简单，是希望在中等规模下获得旗舰级能力、同时避免 MoE 部署复杂度的开发者的实用选择。

据官方介绍，其在主要 coding 基准上的表现超过了上一代规模远大于它的开源旗舰 Qwen3.5-397B-A17B（397B 总参数 / 17B 激活参数的 MoE 模型）——但这个对比需要谨慎解读：一个是稠密模型全量激活 27B，另一个 MoE 模型每次推理只激活 17B，直接拿"参数量"做横向比较容易产生误导。

### 实践注意事项

- Thinking Mode 与 Non-Thinking Mode 通常需要配套不同的采样参数（temperature、top_p、top_k 等），混用会影响输出质量
- 部分推理框架（如某些版本的 vLLM 插件）在两种模式间切换时存在兼容性问题，部署前建议先验证
- 模型已在 Hugging Face / ModelScope 开源权重，同时通过 Alibaba Cloud Model Studio API 及 Qwen Studio 提供在线访问

---

## 核心概念

三个递进的概念，理解它们的关系是理解 Thinking Mode 的关键：

| | **CoT** | **Reasoning Model** | **Thinking Mode** |
|---|---------|---------------------|-------------------|
| **本质** | 推理时的提示技巧 | 模型能力（训练出来的） | 模型推理 或 通过推理服务器API 呈现的功能开关 |
| **需要训练？** | 不需要，大模型"涌现" | 需要 SFT + RL | 不需要，依赖底层 Reasoning Model |
| **谁控制** | 用户（通过 prompt） | 模型（自主推理） | 开发者 / 用户（通过 API 参数） |
| **典型用法** | `"let's think step by step"` | o1 / o3 / DeepSeek-R1 | `enable_thinking: true` |

```
CoT          = 提示技巧，不需要专门训练
Reasoning    = 模型能力，靠 SFT + RL 训练出来
Thinking     = 把 Reasoning 能力包装成可调用的产品功能
```

### CoT（Chain of Thought，思维链）

本质上是一种推理时的技巧，2022 年的论文首次系统提出。核心做法很简单：不让模型直接吐出答案，而是引导它先写出中间推理步骤，再给最终答案。最早的实现方式甚至只是在提示词里加一句 "let's think step by step"，或者给几个带推理过程的示例（few-shot CoT）。

这不需要专门训练——只要基础模型足够大、能力足够强，这种能力就是"涌现"出来的，靠的是模型预训练时学到的知识和模式匹配能力，CoT 只是把这些能力"引导"出来。

### Reasoning Model（推理模型）

这是模型能力层面的东西，指像 OpenAI 的 o1/o3、DeepSeek-R1 这类专门针对"深度推理"训练过的模型。它们和"用 CoT 提示普通模型"的区别在于：这种长链推理能力是训练出来的，不是靠 prompt 诱导的。模型会自主学会探索、回溯、自我验证、纠错这些行为，而普通模型光靠 prompt 很难稳定做到这些。

**训练流程：**

1. 先有一个预训练好的基础模型
2. 用人工或高质量合成的"问题 → 推理过程 → 答案"数据做 SFT 冷启动
3. 核心是用强化学习（PPO、GRPO 等），在有客观可验证答案的任务上训练——数学题、代码题、逻辑题这类"答案对错一目了然"的任务
4. 奖励信号通常只看最终答案是否正确，不监督中间推理的具体内容，让模型自己摸索出"怎么想"更容易得到正确答案

训练数据主要是可自动打分的题目：数学竞赛题、编程题（能跑测试用例判断对错）、逻辑推理题等，因为它们能提供干净的奖励信号，适合大规模 RL 训练。

### Thinking Mode / Extended Thinking（思考模式）

这是"Reasoning Model 能力"的具体呈现形式：模型在给出最终回答前，先生成一段可见（或可选择隐藏）的"思考"内容，通常还能调节思考的 token 预算。这既依赖底层推理能力（训练出来的），也是一个开发者 / 用户可以主动调用、配置的接口层功能。详见下方「Thinking Mode 工作机制验证」一节。

---

## Thinking Mode 工作机制验证

### 核心结论

Thinking Mode 是 **Chat Template 层面**的开关，不是模型权重或推理代码的切换。同一套 27B 权重、同一个 `model.generate()` 调用，行为差异完全来自 **input token 序列的不同**。

### 实验方法

`qwen/low-automodel-thinking-prompt-diff.py` 对同一条消息 `"What is 2+2?"` 分别用 `enable_thinking=False` 和 `enable_thinking=True` 走一遍 `apply_chat_template()`，然后 decode 出来对比 token 序列：

```python
# OFF: 模板直接关闭 thinking 标签
inputs_off = processor.apply_chat_template(messages, ..., enable_thinking=False)
text_off = processor.decode(inputs_off["input_ids"][0])

# ON: 模板只打开 thinking 标签，等模型填入推理
inputs_on = processor.apply_chat_template(messages, ..., enable_thinking=True)
text_on = processor.decode(inputs_on["input_ids"][0])
```

### 实验结果

```
=== Non-Thinking Mode (token count: 19) ===
'<|im_start|>user\nWhat is 2+2?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

=== Thinking Mode (token count: 17) ===
'<|im_start|>user\nWhat is 2+2?\n<|im_end|>\n<|im_start|>assistant\n<think>\n'
```

**唯一差异**：Non-Thinking 模板在 `<think>\n` 后面多了一对 `\n</think>\n`（模板替模型把 thinking 标签关闭了），而 Thinking 模板只输出 `\n`（留下开放标签，等模型自己填入推理内容并关闭）。

### 机制图解

```
enable_thinking=True  (Thinking Mode)
  Template 输出:  ...<|im_start|>assistant\n<think>\n
  模型看到开放标签 → 先写推理过程 → 写 </thinking> → 写最终答案

enable_thinking=False (Non-Thinking Mode)
  Template 输出:  ...<|im_start|>assistant\n<think>\n</thinking>\n\n
  模型看到已关闭的标签 → 跳过推理 → 直接写最终答案
```

### 关键对比

| 维度 | 是否变化 |
|------|---------|
| 模型权重 | ❌ 同一套 27B |
| 推理代码 | ❌ `forward()` 完全一致，没有 `if thinking_mode` |
| `model.generate()` 调用 | ❌ 参数完全一样 |
| Chat Template 注入的 token | ✅ ON 少一对闭合标签 |
| 模型看到的 input_ids | ✅ 17 vs 19 个 token |
| 采样参数（temperature 等） | ⚠️ 官方建议不同模式配不同值 |

两种模式走**完全相同的计算图**：

```
Thinking ON  input_ids ┐
                       ├─▶ token_embed() → forward() layer 0~63 → lm_head() → sample
Thinking OFF input_ids ┘
                       (权重、代码、路径全部相同，唯一变量是那两三个 token)
```

### 为什么这样设计

- **简洁**：不需要维护两套权重或两个 checkpoint
- **灵活**：同一个部署，API 层面按需切换
- **高效**：Non-Thinking 模式跳过推理过程，延迟更低、token 消耗更少
- **统一**：模型的推理能力是训练出来的（SFT + RL），模板只是引导它"想不想"

### 完整调用链路

从客户端发 API 请求到模型输出，`enable_thinking` 参数经历三层传递：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client (API 请求)                              │
│                                                                      │
│  POST /v1/chat/completions                                           │
│  {                                                                   │
│    "messages": [...],                                                │
│    "extra_body": {                                                   │
│      "chat_template_kwargs": {                                       │
│        "enable_thinking": true      ← 客户端指定本次请求要 thinking  │
│      }                                                               │
│    }                                                                 │
│  }                                                                   │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      vLLM 推理服务器                                  │
│                                                                      │
│  ① 解析 API 请求，提取 chat_template_kwargs                          │
│  ② 调用 Chat Template 序列化 messages                               │
│     apply_chat_template(messages, enable_thinking=true)              │
│  ③ Template 在 prompt 中注入 <thinking> 标签                         │
│     → 生成 input_ids (包含 thinking 引导 token)                      │
│  ④ 将 token 序列送入模型推理                                         │
│     model.generate(input_ids, ...) ← 模型不知道"API"的存在           │
│                                                                      │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        模型 (27B 权重)                                │
│                                                                      │
│  看到开放的 <thinking> 标签 → 开始推理                               │
│  生成: <thinking>推理过程</thinking> + 最终答案                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**三层含义：**

| 层级 | 参数 | 谁控制 | 作用 |
|------|------|--------|------|
| **API 层** | `extra_body.chat_template_kwargs.enable_thinking` | 客户端在请求中指定 | 告诉服务器"这次请求要 thinking" |
| **Template 层** | Chat Template 注入 `<thinking>` 标签 | 服务器内部，模板序列化阶段 | 把 thinking 指令变成模型看得懂的 token |
| **模型层** | 输入 token 序列（多/少几个 token） | 完全由模板决定 | 看到开放标签就推理，看到闭合标签就跳过 |

API 参数 → 控制 Chat Template → 改变模型看到的 token 序列。vLLM 服务器是中间层（请求 → 模板 → 序列化 → 推理 → 返回），权重和推理代码从未变化，变的只是那两三个 token。无论哪种推理服务器、哪种 API 格式，最终都收敛为 `input_ids` tensor 送入模型。

### API 支持 vs 模型支持

这是两个**独立的维度**：

| | **模型支持**（有 reasoning 训练） | **模型不支持**（无 reasoning 训练） |
|---|---|---|
| **API 支持 thinking** | ✅ 正常工作（如 Qwen3.6-27B + vLLM） | ⚠️ 模板注入了 token，但模型没经过推理训练，输出质量差 |
| **API 不支持 thinking** | ⚠️ 模型有能力，但无法通过 API 激活；可手动构造 token 绕过 | ❌ 都不行 |

- **模型支持** = 推理能力是**训练出来的**（SFT + RL）
- **API 支持** = 推理服务器提供**参数入口**，把客户端请求翻译成正确的 token 序列

Qwen3.6-27B 的情况：**模型有能力 + vLLM 有参数入口 + Chat Template 知道怎么注入**，三者缺一不可。缺 API 支持还能手动凑 token，缺模型训练就没戏。

### 不同客户端的用法

| 场景 | 如何传递 |
|------|---------|
| **OpenAI 兼容 API** (curl / SDK) | `extra_body: {"chat_template_kwargs": {"enable_thinking": true}}` |
| **Transformers** (Python 直接调用) | `processor.apply_chat_template(..., enable_thinking=True)` |
| **vLLM 原生** | `--enable-auto-tool-choice --reasoning-parser qwen3` |
| **Claude Code** (自托管) | `ANTHROPIC_MODEL` 指向 vLLM，由 vLLM 处理模板 |

直接用 Transformers 库时没有"API 支持不支持"的问题——你就是 API 层，`apply_chat_template()` 返回的 `input_ids` 直接进 `model.generate()`，全程透明。这就是 `qwen/low-automodel.py` 和 `qwen/low-automodel-thinking-prompt-diff.py` 的做法。

---

## 环境配置

### 硬件与软件速查

| 组件 | Host（物理机） | Dev Container |
|------|---------------|---------------|
| GPU | NVIDIA H100 80GB HBM3 | NVIDIA H100 80GB HBM3 |
| 驱动 | NVIDIA 580.173.02 | NVIDIA 580.173.02 |
| CUDA | 13.1（host） / 13.0（container） | 13.0 |
| NCCL | 可用 | (2, 30, 7) |
| Docker | 29.1.3 | -- |
| containerd | 2.2.1 | -- |
| nvidia-container-toolkit | 1.19.1 | -- |
| Python | -- | 3.12.3 |
| PyTorch | -- | 2.14.0+cu130 |
| Transformers | -- | 5.16.1 |

### Host

```bash
# GPU 状态
nvidia-smi
# GPU: NVIDIA H100 80GB HBM3, 0 MiB / 81559 MiB

# CUDA 编译工具
nvcc --version
# CUDA 13.1, V13.1.115

# NVIDIA Container Toolkit
nvidia-ctk --version
# 1.19.1

# Docker
docker version
# Client/Server: 29.1.3, API 1.52
# containerd: 2.2.1, runc: 1.3.4

# 拉取 PyTorch 镜像
docker pull pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel
# 镜像大小: 36.3 GB（内容 13.4 GB）
```

### 部署 Dev Container

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

# 查看日志
docker logs -f dev-server

# 进入容器
docker exec -it dev-server /bin/bash
```

容器内安装 VS Code CLI 并启动隧道：

```bash
# 在容器内
apt update && apt install curl

# 安装 VS Code CLI
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
  -o vscode_cli.tar.gz && tar -xf vscode_cli.tar.gz && \
  mv code /usr/local/bin/code && rm vscode_cli.tar.gz

# GitHub 认证
code tunnel user login --provider github

# 启动隧道
nohup code tunnel --accept-server-license-terms --name 001 &> output.log &
```

### 容器内环境验证

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
# 安装依赖
pip install -U transformers accelerate safetensors huggingface_hub --break-system-packages

# 验证
python -c "import transformers; print(transformers.__version__)"
# 5.16.1

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
# 2.14.0+cu130, True, 1
```

---

## vLLM

### 推理链路

vLLM 的推理核心可以概括为：

```
Scheduler → KV Cache Manager → Model Runner → forward() → Attention → KV Cache → Sampling
```

- **Scheduler**：决定哪些请求、哪些 token 需要计算
- **KV Cache Manager**：以 block 方式管理和分配 KV Cache
- **Model Runner**：调用模型的 `forward()`
- **Attention**：从 KV Cache 读取历史 K/V，将新生成的 K/V 写回 Cache
- **Sampling**：生成下一个 token

在 TP=8 场景下，KV Cache 分布在多个 GPU 上。像 MooncakeConnector 这样的组件，可以介入 KV Cache 层，将 Prefill 产生的 KV Cache 在不同 worker/GPU 之间通过 RDMA 等方式传输。

### 模型参数变化 vs 架构变化

如果新模型只是参数规模、层数、hidden size、vocab size 等发生变化，但 model architecture 和 KV Cache / Attention 机制不变，vLLM 的整体推理框架无需重新设计，已有的模型 implementation 通常可以复用，只需加载不同参数。

只有当新模型引入新的 Attention、KV/State Cache、MoE、Vision Encoder、MTP 等架构或执行机制时，才需要修改相应的 model `forward()`、Attention/KV Cache 或其他 runtime 部分。

### 框架生态适配

如果 Qwen 系列引入全新模型架构，为了让其在主流生态中正常运行，通常需要分别适配：

- **Hugging Face Transformers**：提供标准的模型、配置、Processor 和 `forward()` 实现
- **vLLM / SGLang**：针对各自的 inference runtime 适配模型执行、Attention、KV/State Cache、Tensor Parallel 等机制，实现高性能推理

具体代码不一定全部由阿里维护，也可以由 Qwen、框架团队和社区共同完成。

---

## 测试验证

在 Dev Container 中运行：

```bash
# 最低层：Pipeline API
python qwen/low-pipeline.py

# AutoModel + AutoProcessor（带 Thinking 开关）
python qwen/low-automodel.py

# 模型架构探索（配置、层数、参数、设备分布）
python qwen/low-model-arch.py

# Thinking Mode ON vs OFF — Chat Template 差异验证
python qwen/low-automodel-thinking-prompt-diff.py
```

### Thinking Mode 验证

`qwen/low-automodel-thinking-prompt-diff.py` 确认：Thinking 开关是 **Chat Template 行为**，不是模型权重或推理代码变化。详见上方「Thinking Mode 工作机制验证」一节。

---

## 参考资料

- [CoT (Chain of Thought)](https://www.nvidia.cn/glossary/cot-prompting/)
- [Qwen 官方博客](https://qwen.ai/blog?id=qwen3.6-27b)
- [Hugging Face - Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Transformers Model Doc](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen Code](https://github.com/QwenLM/qwen-code)
- [vLLM Recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [深度学习技术分析 (CN)](http://eadst.com/blog/325)
