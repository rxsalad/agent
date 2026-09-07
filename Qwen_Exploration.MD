# Qwen Exploration

Qwen3.6-27B 是阿里云Qwen系列开源的一个 27B 参数**稠密（dense）**多模态模型，具备原生的视觉语言理解能力（图像、视频与文本联合输入）。

它是一个统一 checkpoint 中同时内置 reasoning 能力的模型：通过在推理请求中显式设置参数（如聊天模板中的 `enable_thinking` 或类似的 `thinking` 布尔字段）来切换 **Thinking Mode** 与 **Non-Thinking Mode**，而非依赖 prompt 软性引导。

- **Thinking Mode**：启用 extended chain-of-thought 推理，模型先输出较长的内部推理过程，再给出最终答案，适用于复杂逻辑推理、数学、编程等任务。
- **Non-Thinking Mode**：跳过显式推理过程，直接生成响应，适用于高效的通用对话场景，延迟更低、成本更省。

两种模式共享同一套模型权重，无需切换到不同的checkpoint，这一设计与同系列的 Qwen3.6-35B-A3B（MoE架构）保持一致。

**架构特点**：作为稠密模型（而非MoE），Qwen3.6-27B 在推理时激活全部参数（active parameters = total parameters），不涉及专家路由（expert routing）的复杂性，因此部署和serving相对简单，是希望在中等规模下获得旗舰级能力、同时避免MoE部署复杂度的开发者的实用选择。据官方介绍，其在主要coding基准上的表现超过了上一代规模远大于它的开源旗舰 Qwen3.5-397B-A17B（397B总参数/17B激活参数的MoE模型）——但这个对比需要谨慎解读，因为一个是稠密模型全量激活27B，另一个MoE模型每次推理只激活17B，直接拿"参数量"做横向比较容易产生误导。

**实践注意事项**：
- Thinking Mode 与 Non-Thinking Mode 通常需要配套不同的采样参数（如 temperature、top_p、top_k 等），混用会影响输出质量，这是实际部署中容易踩的坑。
- 已有社区反馈指出，部分推理框架（如某些版本的 vLLM 插件）在两种模式间切换时存在兼容性问题，实际部署前建议先验证目标推理框架对该模型thinking开关的支持是否稳定。
- 模型已在 Hugging Face / ModelScope 开源权重，同时通过 Alibaba Cloud Model Studio API 及 Qwen Studio 提供在线访问。

## Documentation and Resources 

- [CoT](https://www.nvidia.cn/glossary/cot-prompting/)



## 几个核心概念

- CoT = 推理时的提示技巧，不需要专门训练
- Reasoning Model = 模型能力，靠SFT+RL训练出来的
- Thinking Mode = 把Reasoning Model能力包装成的一个可调用/可配置的产品功能

Reasoning Model + Thinking ON → inference 时进行显式/扩展 reasoning。
Reasoning Model + Thinking OFF → 不进行（或显著减少）这种显式 reasoning，直接生成答案。


### CoT（Chain of Thought，思维链）

本质上是一种推理时的技巧，2022年的论文首次系统提出。核心做法很简单：不让模型直接吐出答案，而是引导它先写出中间推理步骤，再给最终答案。最早的实现方式甚至只是在提示词里加一句"let's think step by step"，或者给几个带推理过程的示例（few-shot CoT）。这不需要专门训练——只要基础模型足够大、能力足够强，这种能力就是"涌现"出来的，靠的是模型预训练时学到的知识和模式匹配能力，CoT只是把这些能力"引导"出来。

### Reasoning Model（推理模型）

这是模型能力层面的东西，指像 OpenAI 的 o1/o3、DeepSeek-R1 这类专门针对"深度推理"训练过的模型。它们和"用CoT提示普通模型"的区别在于：这种长链推理能力是训练出来的，不是靠prompt诱导的。模型会自主学会探索、回溯、自我验证、纠错这些行为，而普通模型光靠prompt很难稳定做到这些。

训练方式大致是：

- 先有一个预训练好的基础模型
- 有时先用人工或合成的高质量推理轨迹做监督微调（SFT）"冷启动"
- 核心是用强化学习（如PPO、GRPO等），在有客观可验证答案的任务上训练——数学题、代码题、逻辑题这类"答案对错一目了然"的任务
- 奖励信号通常只看最终答案是否正确，不监督中间推理的具体内容，让模型自己摸索出"怎么想"更容易得到正确答案

训练数据主要就是这些可自动打分的题目：数学竞赛题、编程题（能跑测试用例判断对错）、逻辑推理题等，因为它们能提供干净的奖励信号，适合大规模RL训练。


Deep Reasoning 模型的训练数据，核心不是单纯收集大量人工编写的长 CoT，而是由高质量推理题目、可验证的答案以及模型生成的探索轨迹组成。训练通常先用人工或高质量合成的“问题→推理过程→答案”数据进行 SFT 冷启动，然后通过数学、代码、逻辑等具有明确正确答案的任务进行大规模强化学习，让模型自己探索不同解题路径，并通过最终答案或 verifier 给予奖励。经过反复训练，模型逐渐学会探索、回溯、自我验证和纠错等推理行为，而不是仅仅依赖 prompt 强制生成 CoT。


### Thinking Mode / Extended Thinking（思考模式）

这通常是产品/API层面的功能开关，是"Reasoning Model能力"的一种具体呈现形式：模型在给出最终回答前，会先生成一段可见（或可选择隐藏）的"思考"内容，通常还能调节思考的token预算（想得更久/更浅）。这既依赖底层模型本身具备推理能力（训练出来的），也是一个开发者/用户可以主动调用、配置的接口层功能。

## vLLM 

vLLM 的推理核心可以概括为：Scheduler 负责决定哪些请求、哪些 token 需要计算；KV Cache Manager 负责以 block 的方式管理和分配 KV Cache；Model Runner 调用模型的 forward()；Attention 从 KV Cache 中读取历史 K/V，并将新生成的 K/V 写回 Cache，最后通过 sampling 生成下一个 token。 在 TP=8 场景下，KV Cache 会分布在多个 GPU 上；而像 MooncakeConnector 这样的组件，则可以介入 KV Cache 层，将 Prefill 产生的 KV Cache 在不同 worker/GPU 之间通过 RDMA 等方式传输。因此，理解 vLLM 的 KV Cache，重点是沿着 Scheduler → KV Cache Manager → Model Runner → forward() → Attention → KV Cache → Connector 这条链路来看。


如果新模型只是参数规模、层数、hidden size、vocab size 等发生变化，但 model architecture 和 KV Cache / Attention 机制不变，那么 vLLM 的整体推理框架通常不需要重新设计：
```
Scheduler → KV Cache Manager → Model Runner → forward() → Attention → KV Cache → Sampling
```
已有的模型 implementation 也通常可以复用，只需要根据新的模型配置加载不同参数即可。

只有当新模型引入了新的 Attention、KV/State Cache、MoE、Vision Encoder、MTP 等架构或执行机制时，才可能需要修改相应的 model forward()、Attention/KV Cache 或其他 runtime 部分。


如果 Qwen3.8-27B 引入全新的模型架构，为了让其在主流生态中正常运行，通常需要分别适配 Hugging Face Transformers、vLLM 和 SGLang。

Transformers 需要提供标准的模型、配置、Processor 和 forward() 实现；vLLM 和 SGLang 则需要进一步针对各自的 inference runtime 适配模型执行、Attention、KV/State Cache、Tensor Parallel 等机制，以实现高性能推理。

具体代码不一定全部由阿里维护，也可以由 Qwen、框架团队和社区共同完成。


## Environment

### Host

``` bash

# nvidia-smi
Sun Sep  6 18:27:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:00:09.0 Off |                    0 |
| N/A   39C    P0             74W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_Dec_16_07:23:41_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0

# dpkg -l | grep -i nvidia-container-toolkit
ii  nvidia-container-toolkit         1.19.1-1                                         amd64        NVIDIA Container toolkit
ii  nvidia-container-toolkit-base    1.19.1-1                                         amd64        NVIDIA Container Toolkit Base

# apt list --installed 2>/dev/null | grep -i nvidia-container
libnvidia-container-tools/unknown,now 1.19.1-1 amd64 [installed,upgradable to: 1.20.0-1]
libnvidia-container1/unknown,now 1.19.1-1 amd64 [installed,upgradable to: 1.20.0-1]
nvidia-container-toolkit-base/unknown,now 1.19.1-1 amd64 [installed,upgradable to: 1.20.0-1]
nvidia-container-toolkit/unknown,now 1.19.1-1 amd64 [installed,upgradable to: 1.20.0-1]

# which nvidia-ctk
/usr/bin/nvidia-ctk

# nvidia-ctk --version
NVIDIA Container Toolkit CLI version 1.19.1
commit: 09ceee5dde66ba9ce25c7cc69b1ebd5e6e3266fa
 
# docker info | grep -i runtime
 Runtimes: io.containerd.runc.v2 runc
 Default Runtime: runc

# docker version
Client:
 Version:           29.1.3
 API version:       1.52
 Go version:        go1.24.4
 Git commit:        29.1.3-0ubuntu3~22.04.2
 Built:             Wed Apr 29 22:18:59 2026
 OS/Arch:           linux/amd64
 Context:           default

Server:
 Engine:
  Version:          29.1.3
  API version:      1.52 (minimum version 1.44)
  Go version:       go1.24.4
  Git commit:       29.1.3-0ubuntu3~22.04.2
  Built:            Wed Apr 29 22:18:59 2026
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          2.2.1
  GitCommit:        
 runc:
  Version:          1.3.4-0ubuntu1~22.04.1
  GitCommit:        
 docker-init:
  Version:          0.19.0
  GitCommit:   

# docker pull pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel

# docker image ls
IMAGE                                          ID             DISK USAGE   CONTENT SIZE   EXTRA
pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel   cce09744eb86       36.3GB         13.4GB        

```

### Deploy a Dev Contianer

``` bash
docker run --rm -d \
  --name dev-server \
  --gpus all \
  --privileged \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /bin/bash \
  pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel \
  -c '
set -ex
sleep infinity
'

# docker logs -f dev-server

# docker stop dev-server
```

``` bash
# docker exec -it dev-server /bin/bash

root@3f20a4f1c3ba:/workspace# apt update 
root@3f20a4f1c3ba:/workspace# apt install curl

root@3f20a4f1c3ba:/workspace# curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o vscode_cli.tar.gz && \
  tar -xf vscode_cli.tar.gz && \
  mv code /usr/local/bin/code && \
  rm vscode_cli.tar.gz

# Authenticate the tunnel with your GitHub account using the provided sign-in URL.
root@3f20a4f1c3ba:/workspace# code tunnel user login --provider github

# Start the VS Code tunnel (ID: 001) in the background
root@3f20a4f1c3ba:/workspace# nohup code tunnel --accept-server-license-terms --name 001 &> output.log &
```

### Dev Envrionment

``` bash
root@3f20a4f1c3ba:/workspace# nvidia-smi
Sun Sep  6 18:57:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:00:09.0 Off |                    0 |
| N/A   39C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

root@3f20a4f1c3ba:/workspace# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0

root@3f20a4f1c3ba:/workspace# python3 -c "import torch; print(torch.cuda.is_available())"
True

root@3f20a4f1c3ba:/workspace# python3 -c "import torch; print(torch.__version__)"
2.14.0+cu130

root@3f20a4f1c3ba:/workspace# python3 -c "import torch; print(torch.distributed.is_nccl_available())"
True

root@3f20a4f1c3ba:/workspace# python3 -c "import torch; print(torch.cuda.nccl.version())"
(2, 30, 7)

root@3f20a4f1c3ba:/workspace# python3 --version
Python 3.12.3
```

### Hugging Face Transformers

``` bash

root@3f20a4f1c3ba:/workspace# pip install -U transformers accelerate safetensors huggingface_hub --break-system-packages

root@3f20a4f1c3ba:/workspace# python -c "import transformers; print(transformers.__version__)"
5.16.1

root@3f20a4f1c3ba:/workspace# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
2.14.0+cu130
True
1
```

## Test

```
root@3f20a4f1c3ba:/workspace# python qwen/low-pipeline.py

root@3f20a4f1c3ba:/workspace# python qwen/low-automodel.py

root@3f20a4f1c3ba:/workspace# python qwen/low-model-arch.py
```