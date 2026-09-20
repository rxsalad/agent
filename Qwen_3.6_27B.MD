# Qwen3.6 27B

## Documentation and Resources 

- [vLLM Recipe Qwen3.6-27B](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)

- [vLLM Recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

- [vLLM Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

- [Hugging Face - Model](https://huggingface.co/Qwen/Qwen3.6-27B)

- [Hugging Face - Model Doc ](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

- [Official Blog](https://qwen.ai/blog?id=qwen3.6-27b)

- [Qwen Code](https://github.com/QwenLM/qwen-code)

- [In-Depth Technical Analysis - CHN](http://eadst.com/blog/325)

- [PyTorch Image](https://hub.docker.com/r/pytorch/pytorch/tags)

- [vLLM Image](https://hub.docker.com/r/vllm/vllm-openai/tags)

## Deploy a vLLM-Qwen Contianer 

``` bash
# BF16, TP1, Tool Calling, Reasoning, Encoding Parallel

mkdir ~/vllm-logs

docker run --rm -d \
  --name vllm-qwen3.6-27b \
  --gpus all \
  --privileged \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-logs:/logs \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -c '
set -ex

vllm serve Qwen/Qwen3.6-27B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --api-key ********** \
  > /logs/vllm.log 2>&1
'

# docker logs -f vllm-qwen3.6-27b

# tail -f ~/vllm-logs/vllm.log 

# docker stop vllm-qwen3.6-27b
```

``` bash
# watch -n 2 nvidia-smi

# nvidia-smi
Sun Sep  6 17:52:44 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200                    On  |   00000000:83:00.0 Off |                    0 |
| N/A   47C    P0            493W /  700W |  131981MiB / 143771MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          215144      C   VLLM::EngineCore                      13197... |
+-----------------------------------------------------------------------------------------+

# pwd
/root/.cache/huggingface/hub

# du -sh *
52G	models--Qwen--Qwen3.6-27B

# tree
.
├── CACHEDIR.TAG
└── models--Qwen--Qwen3.6-27B
    ├── blobs
    │   ├── 023756cfadf88e5bf69eefeee3e172f38c448d64
    │   ├── 03de44dc7e933025498d72f8d7ea32d5cde16eb7a85b35ff1fa49a737f4b2242
    │   ├── 0aa0ce0658d60ac4a5d609f4eadb0e8e43514176
    │   ├── 0bcd0ce28c7d2cc6f5ab2c21902cd032fec0cc5f29e0897e3bb0967add388e47
    │   ├── 160d914e2e4704a401a3eab9e9eec3380e5723a35981c9169bedfb82ea32e6d6
    │   ├── 26c114fb6d5d4131ab227552d489d7a97c8c958a9ed94f6d3d1584f6cfe4b9fb
    │   ├── 28d96ff303d1d20350185caf4bf037045916ed35
    │   ├── 2ea84a437d448ff71b08df68fdd949d5cc4ebb64
    │   ├── 33c5d7d18e1b3f661334dff736fc7d00561f1f1bb9d2970c669b40d2e582974e
    │   ├── 3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe
    │   ├── 44e8fe06d2d609bf20b16b1d2f42348ca2dc99179721f76035ff50db678f9a8a
    │   ├── 584a0ed8018d3b19ce8e533d08a460454650027901be6323b50b591da143dca6
    │   ├── 5c3a68304dabeaa5a0eb70a5e383d2b8134997ef6d5c1e8afdcd236091c847f2
    │   ├── 5f21d4e349aef6c74bedef7b3835dc8c11a16dd5ce72f4437e2284f1e83736e9
    │   ├── 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
    │   ├── 68db2ebb03231238c2a114f3642df967a970a1d96b029adc427838cb5cf7f27a
    │   ├── 9f3dd1b1c6d0b43f4b33819dbdd1b60369f9006e
    │   ├── a494e019ca1502219fd0128658b979e5f05ae8e8
    │   ├── a5abc1d5e9583409193e2cd58a671a4ed98467a6eaf883cd4377e3cc02021c22
    │   ├── a8755d827c0a7b614c246c4060dfd58ab352a8ff
    │   ├── b84b5b1315e865c9a19a444045d422a73e3e2e31ce3766797cffd3507c68c9c9
    │   ├── ba8b0849cb4c4c97e674709bfe56a9c008d7563a68ac57120ceb2855a73a9944
    │   ├── e7e3e1a17a2673340eb6bba95c3c6f07b12e7746ee738dde56551cbde27656f5
    │   ├── e8934789f4742c11da88f937da84bdbb8764c8a7f7ebe9e7b056bcbd41b14285
    │   └── ea32be691d17374c05b277246085e81b70e6ec0f
    ├── refs
    │   └── main
    ├── snapshots
    │   └── 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
    │       ├── chat_template.jinja -> ../../blobs/a8755d827c0a7b614c246c4060dfd58ab352a8ff
    │       ├── config.json -> ../../blobs/9f3dd1b1c6d0b43f4b33819dbdd1b60369f9006e
    │       ├── generation_config.json -> ../../blobs/023756cfadf88e5bf69eefeee3e172f38c448d64
    │       ├── merges.txt -> ../../blobs/a494e019ca1502219fd0128658b979e5f05ae8e8
    │       ├── model-00001-of-00015.safetensors -> ../../blobs/5f21d4e349aef6c74bedef7b3835dc8c11a16dd5ce72f4437e2284f1e83736e9
    │       ├── model-00002-of-00015.safetensors -> ../../blobs/03de44dc7e933025498d72f8d7ea32d5cde16eb7a85b35ff1fa49a737f4b2242
    │       ├── model-00003-of-00015.safetensors -> ../../blobs/5c3a68304dabeaa5a0eb70a5e383d2b8134997ef6d5c1e8afdcd236091c847f2
    │       ├── model-00004-of-00015.safetensors -> ../../blobs/ba8b0849cb4c4c97e674709bfe56a9c008d7563a68ac57120ceb2855a73a9944
    │       ├── model-00005-of-00015.safetensors -> ../../blobs/a5abc1d5e9583409193e2cd58a671a4ed98467a6eaf883cd4377e3cc02021c22
    │       ├── model-00006-of-00015.safetensors -> ../../blobs/160d914e2e4704a401a3eab9e9eec3380e5723a35981c9169bedfb82ea32e6d6
    │       ├── model-00007-of-00015.safetensors -> ../../blobs/0bcd0ce28c7d2cc6f5ab2c21902cd032fec0cc5f29e0897e3bb0967add388e47
    │       ├── model-00008-of-00015.safetensors -> ../../blobs/584a0ed8018d3b19ce8e533d08a460454650027901be6323b50b591da143dca6
    │       ├── model-00009-of-00015.safetensors -> ../../blobs/e7e3e1a17a2673340eb6bba95c3c6f07b12e7746ee738dde56551cbde27656f5
    │       ├── model-00010-of-00015.safetensors -> ../../blobs/e8934789f4742c11da88f937da84bdbb8764c8a7f7ebe9e7b056bcbd41b14285
    │       ├── model-00011-of-00015.safetensors -> ../../blobs/44e8fe06d2d609bf20b16b1d2f42348ca2dc99179721f76035ff50db678f9a8a
    │       ├── model-00012-of-00015.safetensors -> ../../blobs/33c5d7d18e1b3f661334dff736fc7d00561f1f1bb9d2970c669b40d2e582974e
    │       ├── model-00013-of-00015.safetensors -> ../../blobs/68db2ebb03231238c2a114f3642df967a970a1d96b029adc427838cb5cf7f27a
    │       ├── model-00014-of-00015.safetensors -> ../../blobs/26c114fb6d5d4131ab227552d489d7a97c8c958a9ed94f6d3d1584f6cfe4b9fb
    │       ├── model-00015-of-00015.safetensors -> ../../blobs/b84b5b1315e865c9a19a444045d422a73e3e2e31ce3766797cffd3507c68c9c9
    │       ├── model.safetensors.index.json -> ../../blobs/ea32be691d17374c05b277246085e81b70e6ec0f
    │       ├── preprocessor_config.json -> ../../blobs/2ea84a437d448ff71b08df68fdd949d5cc4ebb64
    │       ├── tokenizer.json -> ../../blobs/5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
    │       ├── tokenizer_config.json -> ../../blobs/28d96ff303d1d20350185caf4bf037045916ed35
    │       ├── video_preprocessor_config.json -> ../../blobs/3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe
    │       └── vocab.json -> ../../blobs/0aa0ce0658d60ac4a5d609f4eadb0e8e43514176
    └── trees
        └── 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9.json

6 directories, 53 files
```

## Test the Contianer 

``` bash
# curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer **********" \
  -d '{
    "model": "Qwen/Qwen3.6-27B",
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ]
  }'

# curl http://***.***.***.***:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer **********" \
  -d '{
    "model": "Qwen/Qwen3.6-27B",
    "messages": [
      {"role": "user", "content": "How to learn AI/ML, 200 words?"}
    ]
  }'

# curl http://***.***.***.***:8000/v1/models \
  -H "Authorization: Bearer **********"
```