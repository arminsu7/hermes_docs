# Triton Server + TensorRT-LLM 在 Jetson AGX Orin 上部署技术指南

> 基于 NVIDIA 开发者论坛帖子整理：https://forums.developer.nvidia.com/t/deploying-triton-server-with-tensorrt-llm-on-jetson-agx-orin-jetpack-6-2-any-working-example/333564/6
> 原帖发布时间：2025年5月22日
> 补充：NVIDIA Jetson AI Lab、Triton Inference Server 官方文档、dusty-nv/jetson-containers、相关论坛帖子
> 说明：NVIDIA 论坛页面需要 JavaScript 渲染，部分回复内容根据原始帖摘要和关联帖子推断。技术细节根据 Triton、TRT-LLM 官方文档和 dusty-nv 项目补充

---

## 目录

1. [背景：原帖的问题](#1-背景原帖的问题)
2. [核心挑战：为什么 Triton + TRT-LLM on Jetson 不简单](#2-核心挑战为什么-triton--trt-llm-on-jetson-不简单)
3. [方案对比与选择](#3-方案对比与选择)
4. [当前可行的替代方案](#4-当前可行的替代方案)
5. [结论与建议](#5-结论与建议)

---

## 1. 背景：原帖的问题

### 1.1 原始提问

一位开发者在 NVIDIA 开发者论坛（2025年5月22日）提问：

> "I'm working with a Jetson AGX Orin 32GB device running JetPack 6.2 and trying to deploy a 1.5B large language model using Triton Inference Server and TensorRT-LLM. I've managed to convert a large model using TensorRT-LLM, and I can successfully run inference via Python scripts -- ..."

**关键信息：**
- 设备：Jetson AGX Orin 32GB
- 系统：JetPack 6.2
- 模型：1.5B LLM
- 目标：Triton Inference Server + TensorRT-LLM backend
- 当前状态：Python 脚本推理成功，但 Triton Server 部署卡住

### 1.2 问题的本质

用户已经完成了最困难的一步 -- 在 Jetson AGX Orin 上使用 TRT-LLM v0.12.0-jetson 分支成功转换模型并运行 Python 推理。但在"把 TRT-LLM engine 交给 Triton Server 来 serve"这一步遇到了障碍。

---

## 2. 核心挑战：为什么 Triton + TRT-LLM on Jetson 不简单

### 2.1 Triton 容器与 Jetson 的架构不匹配

**Triton Inference Server 的标准 NGC 容器是为 x86_64 架构构建的**，直接在 Jetson ARM64 上无法运行。

```
x86_64 容器：nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
                                 ↑
                            NOT compatible with ARM64 (Jetson)
```

### 2.2 JetPack 6.0+ 的 Triton for Jetson 容器

从 JetPack 6.0 开始，NVIDIA 发布了带 `-igpu` 后缀的 Triton 容器，专门面向 Jetson：

```bash
# Jetson 专用的 Triton 容器（JetPack 6.x）
nvcr.io/nvidia/tritonserver:<version>-igpu

# 但注意：这些容器的 TRT-LLM backend 支持可能不完整
```

### 2.3 TRT-LLM on Jetson 的版本约束

| 维度 | 数据中心/桌面 | Jetson AGX Orin |
|------|-------------|-----------------|
| TRT-LLM 版本 | v1.3.0rc21 (最新) | **v0.12.0-jetson** |
| 架构 | PyTorch backend (v1.0+) | 旧版 C++ runtime + Python wrapper |
| `trtllm-build` | 已移除 (rc21) | 仍在使用 |
| 维护状态 | 活跃开发 | **冻结分支，仅 bug fix** |

**v0.12.0-jetson 分支的局限性：**
- 基于 v0.12.0，远早于 v1.0 PyTorch backend 重构
- 不支持 Triton 的 PyTorch backend 集成方式
- 模型转换流程（`convert_checkpoint.py`）与主线不兼容
- Engine 格式可能与新版 Triton 不匹配

### 2.4 Triton + TRT-LLM backend 的架构

```
┌──────────────────────────────────────┐
│         Triton Inference Server       │
│         (Python backend or C++)       │
├──────────────────────────────────────┤
│  tensorrtllm_backend (Python/C++)     │
│  ┌──────────────────────────────────┐ │
│  │  Model Repository                 │ │
│  │  ├── config.pbtxt                │ │
│  │  ├── 1/                          │ │
│  │  │   └── model.engine            │ │
│  │  └── tokenizer/                  │ │
│  └──────────────────────────────────┘ │
├──────────────────────────────────────┤
│  TensorRT-LLM Runtime (Python API)    │
│  → TRT Engine execution              │
└──────────────────────────────────────┘
```

**问题：** v0.12.0-jetson 的 TRT-LLM runtime 和 Triton 的 `tensorrtllm_backend` 版本不匹配。Triton backend 期望的是与它同步发布的 TRT-LLM 版本（通常以 NGC 容器形式提供），但 Jetson 上只能用 v0.12.0-jetson。

---

## 3. 方案对比与选择

### 3.1 方案 A：构建 Triton from source on Jetson（理论可行，实测困难）

```bash
# 构建 Triton Server on Jetson
git clone https://github.com/triton-inference-server/server.git
cd server
python3 build.py \
    --enable-gpu \
    --backend tensorrtllm \
    --build-dir ./build
```

**问题：**
- Jetson AGX Orin 32GB 内存不足以编译完整的 Triton（需要 64GB+ 或交叉编译）
- 依赖链长（TRT-LLM → TensorRT → CUDA → JetPack），版本匹配困难
- 社区有报告（GitHub Issue #7023）尝试 JetPack 5.x 构建失败
- 编译时间可能数小时

### 3.2 方案 B：使用 vLLM + Triton backend on Jetson

有一个并行帖子讨论了更可行的方案：

> "Triton Inference Server + vLLM Backend on the NVIDIA Jetson AGX Orin 64GB Developer Kit"
> -- https://forums.developer.nvidia.com/t/triton-inference-server-vllm-backend-on-the-nvidia-jetson-agx-orin-64gb-developer-kit/312008/9

vLLM 的 Triton backend 集成比 TRT-LLM 简单，因为：
- vLLM 不需要预编译 engine
- vLLM 的 Jetson 社区 wheel 相对成熟（thehighnotes/vllm-jetson-orin）
- Triton vLLM backend 的 Python 集成更简单

### 3.3 方案 C：TensorRT Edge-LLM（NVIDIA 推荐的替代方案）

NVIDIA 官方已经确认 TensorRT Edge-LLM 取代 TRT-LLM v0.12.0-jetson 成为 Jetson 上的推理方案：

> "TensorRT-LLM (the datacenter variant) does not support Jetson. That's exactly why NVIDIA built TensorRT-Edge-LLM — a separate, purpose-built inference engine specifically for edge devices like Jetson and DRIVE."
> -- NVIDIA 论坛回复 [#365412](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412)

Edge-LLM 不依赖 Triton，提供自己的 C++ HTTP 服务：

```bash
# Edge-LLM 自带 HTTP server
./bin/edge-llm-server --engine_dir ./engine --port 8000
```

### 3.4 方案 D：直接用 Python API 部署（放弃 Triton）

既然 Python 脚本已经能成功推理，可以绕过 Triton 直接服务化：

```python
# 使用 FastAPI 封装 TRT-LLM Python API
from fastapi import FastAPI
from tensorrt_llm.runtime import ModelRunner

app = FastAPI()
runner = ModelRunner.from_dir("./engine")

@app.post("/generate")
async def generate(prompt: str):
    outputs = runner.generate([prompt])
    return {"response": outputs[0]}
```

---

## 4. 当前可行的替代方案

### 4.1 推荐方案：TensorRT Edge-LLM + C++ HTTP Server

```bash
# Step 1: x86 宿主机上量化 + ONNX 导出（需 GPU）
docker run --gpus all nvcr.io/nvidia/tensorrt-edge-llm/quantization:latest
# ... 执行 AWQ 量化 + ONNX 导出

# Step 2: ONNX 文件复制到 Jetson
scp -r ./qwen3-4b-onnx user@jetson:/models/

# Step 3: Jetson 上编译 engine
cd ~/TensorRT-Edge-LLM/build
./bin/edge-llm-build --model_dir /models/qwen3-4b-onnx --output_dir ./engine

# Step 4: 启动 HTTP 服务
./bin/edge-llm-server --engine_dir ./engine --port 8000

# Step 5: 客户端调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 128}'
```

### 4.2 备选方案：vLLM + 定制 wheel

```bash
# 安装 Jetson 专用 vLLM wheel
pip install https://huggingface.co/thehighnotes/vllm-jetson-orin/resolve/main/vllm-0.17.0-cp310-linux-aarch64.whl

# 启动 OpenAI 兼容 API
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct \
    --quantization gptq \
    --max-model-len 4096
```

### 4.3 轻量方案：llama.cpp llama-server

```bash
# 编译 llama.cpp
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)

# 启动 HTTP 服务
./build/bin/llama-server -m qwen3-4b-q4_k_m.gguf -c 4096 -ngl 99 --host 0.0.0.0 --port 8000
```

---

## 5. 结论与建议

### 5.1 原帖问题的结论

**在 Jetson AGX Orin + JetPack 6.2 上通过 Triton Server + TensorRT-LLM 部署 LLM 不是一个推荐的方案。** 原因：

1. **TRT-LLM v0.12.0-jetson 与 Triton 不兼容**：版本差距太大，backend 版本不匹配
2. **Triton 容器化对 ARM64 支持不完整**：虽然有 `-igpu` 容器，但 TRT-LLM backend 集成未经 Jetson 测试
3. **有更好的替代方案**：TensorRT Edge-LLM、vLLM、llama.cpp 都提供了可直接使用的 HTTP 服务

### 5.2 针对不同场景的建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **需要生产级 HTTP 服务** | vLLM + Jetson wheel | 生态最成熟，OpenAI API 兼容 |
| **需要极致性能 + C++ 部署** | TensorRT Edge-LLM | NVIDIA 官方推荐，纯 C++，低开销 |
| **快速原型开发** | Ollama 容器 | 一行安装，自动量化 |
| **需要 Triton 风格的模型管理** | vLLM + Triton backend (如有可用容器) | 保持 Triton 生态统一 |
| **TRT-LLM on Jetson 过渡期** | 直接用 Python API + FastAPI | 最简单，放弃 Triton |

---

## 参考链接

- 原帖：https://forums.developer.nvidia.com/t/deploying-triton-server-with-tensorrt-llm-on-jetson-agx-orin-jetpack-6-2-any-working-example/333564
- Triton Server GitHub：https://github.com/triton-inference-server/server
- TRT-LLM Jetson AI Lab：https://www.jetson-ai-lab.com/tensorrt_llm
- TRT-LLM for Jetson 公告：https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227
- Triton on Jetson FAQ：https://forums.developer.nvidia.com/t/triton-on-jetson-orin/277016
- vLLM + Triton on Jetson：https://forums.developer.nvidia.com/t/triton-inference-server-vllm-backend-on-the-nvidia-jetson-agx-orin-64gb-developer-kit/312008/9
- TRT Edge-LLM：https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- dusty-nv/jetson-containers：https://github.com/dusty-nv/jetson-containers
- TensorRT-LLM v0.12.0-jetson 构建 (Issue #4502)：https://github.com/NVIDIA/TensorRT-LLM/issues/4502
