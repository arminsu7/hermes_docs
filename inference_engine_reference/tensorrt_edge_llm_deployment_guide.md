# TensorRT Edge-LLM 部署技术指南

> 基于 Jetson AI Lab 官方教程整理：https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
> 补充：TensorRT Edge-LLM 官方文档、NVIDIA 开发者博客、GitHub 仓库
> 说明：本教程完整覆盖两个示例模型 —— Cosmos Reason2 8B (VLM, Jetson Thor) 和 Qwen3-4B-Instruct (LLM, Jetson Orin Nano)，包含量化、ONNX 导出、TensorRT engine 编译、C++ 端侧推理四阶段。部分技术细节根据官方 Quick Start Guide 和 Python Export Pipeline 文档补充

---

## 目录

1. [TensorRT Edge-LLM 概述](#1-tensorrt-edge-llm-概述)
2. [环境准备与安装](#2-环境准备与安装)
3. [示例 1：Qwen3-4B-Instruct on Orin Nano（INT4 AWQ）](#3-示例-1qwen3-4b-instruct-on-orin-nanoint4-awq)
4. [示例 2：Cosmos Reason2 8B on Jetson Thor（NVFP4 VLM）](#4-示例-2cosmos-reason2-8b-on-jetson-thornvfp4-vlm)
5. [Python Export Pipeline 详解](#5-python-export-pipeline-详解)
6. [C++ Runtime 部署](#6-c-runtime-部署)

---

## 1. TensorRT Edge-LLM 概述

### 1.1 是什么

TensorRT Edge-LLM 是 NVIDIA 为嵌入式/边缘平台专门设计的**纯 C++ LLM/VLM 推理运行时**。与 TensorRT-LLM（面向数据中心/桌面 GPU）不同，Edge-LLM 专为资源受限的 Jetson、DRIVE 和 DGX Spark 平台优化。

**核心定位：**
- 轻量级 C++ 实现，零 Python 依赖在推理路径中
- 支持 LLM + VLM（Qwen3/3.5/3.6、Llama、Gemma、InternVL3/3.5、Phi-4、Nemotron-Nano、Cosmos Reason2 等）
- 量化支持：INT4 AWQ、INT8 SmoothQuant、NVFP4（Blackwell 平台）
- Speculative Decoding：EAGLE-3 draft model 加速 1.4-3.5x

### 1.2 完整工作流

```
┌─────────────────────────────────────────────────────┐
│              Step 1: 模型准备（x86 宿主机）            │
│  HuggingFace 模型 → 量化 (AWQ/NVFP4) → ONNX 导出     │
│  工具：nvidia-modelopt + tensorrt_edgellm            │
└──────────────────────┬──────────────────────────────┘
                       │ 复制 ONNX 文件到 Jetson
┌──────────────────────┴──────────────────────────────┐
│              Step 2: Engine 编译（Jetson 上）          │
│  ONNX 模型 → TensorRT Engine                         │
│  工具：edge-llm-build（C++ API）                      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│              Step 3: C++ 推理（Jetson 上）             │
│  纯 C++ 二进制加载 engine → 推理                      │
│  无 Python 解释器在推理路径中                          │
└─────────────────────────────────────────────────────┘
```

**关键约束：**
- 量化 + ONNX 导出必须在 **x86 宿主机**（带 NVIDIA GPU）上执行
- Engine 编译和推理在 **Jetson 设备**上执行
- 跨平台文件传输：ONNX 模型从 x86 复制到 Jetson

### 1.3 支持平台

| 平台 | 支持情况 | 备注 |
|------|---------|------|
| **Jetson Thor** | ✅ 主要平台 | 128GB 统一内存，FP8/NVFP4 支持 |
| **Jetson Orin Nano** | ✅ 支持 | 8GB 统一内存，Qwen3-4B INT4 AWQ 验证通过 |
| **Jetson AGX Orin** | ✅ 支持 | 64GB 统一内存，Performance Benchmarks 列为基准平台 |
| **Jetson Orin NX** | ✅ 支持 | 16GB 统一内存 |
| **DRIVE AGX Thor** | ✅ 主要平台 | 汽车/自动驾驶 |
| **DGX Spark** | ⚠️ 矛盾 | 安装页面列为目标，论坛表示不支持 |

---

## 2. 环境准备与安装

### 2.1 x86 宿主机（模型量化 + ONNX 导出）

```bash
# 系统要求
# - x86_64 Linux (Ubuntu 22.04/24.04)
# - NVIDIA GPU (CUDA 12.x)
# - NVIDIA Container Toolkit

# 使用 Docker 安装（推荐，避免依赖冲突）
docker pull nvcr.io/nvidia/tensorrt-edge-llm/quantization:latest

# 启动容器
docker run --gpus all -it --rm \
    -v /path/to/models:/workspace/models \
    nvcr.io/nvidia/tensorrt-edge-llm/quantization:latest
```

### 2.2 Jetson 设备（Engine 编译 + C++ Runtime）

```bash
# 系统要求
# - JetPack 6.0+ (推荐 6.2+)
# - Jetson Orin Nano / NX / AGX / Thor

# 安装依赖
sudo apt update
sudo apt install -y build-essential cmake git

# 克隆 TensorRT Edge-LLM
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM
cd TensorRT-Edge-LLM

# 构建 C++ Runtime
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=87    # SM 8.7 for Orin
make -j$(nproc)
```

### 2.3 验证安装

```bash
# 检查 C++ Runtime 编译产物
ls -la build/bin/

# 应该看到类似：
# edge-llm-build     # Engine 编译工具
# edge-llm-run       # 推理运行工具
# edge-llm-server    # HTTP 服务工具
```

---

## 3. 示例 1：Qwen3-4B-Instruct on Orin Nano（INT4 AWQ）

### 3.1 教程背景

Jetson AI Lab 教程在 **Jetson Orin Nano 8GB** 上验证 Qwen3-4B-Instruct 的 INT4 AWQ 部署：

- 4B 参数模型经过 INT4 AWQ 量化后权重仅约 **2GB**
- 8GB 统一内存中 2GB 给权重，剩余 6GB 给 KV Cache + 系统开销
- 纯 C++ 端侧推理，零 Python 依赖

### 3.2 Step 1：AWQ INT4 量化 + ONNX 导出（x86 宿主机）

使用 NVIDIA ModelOpt 进行 AWQ 量化，然后导出为 ONNX：

```python
# export_qwen3_4b.py (在 x86 宿主机上运行)

from modelopt import quantize
from tensorrt_edgellm.export import export_onnx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载 HuggingFace 模型
model_name = "Qwen/Qwen3-4B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. AWQ INT4 量化
quantized_model = quantize(
    model,
    method="awq",
    bits=4,                          # INT4
    group_size=128,                  # AWQ group size
    calibration_dataset="pileval"    # 校准数据集
)

# 3. 导出 ONNX
export_config = {
    "max_batch_size": 1,
    "max_seq_len": 2048,
    "precision": "int4",
}
export_onnx(
    quantized_model,
    tokenizer,
    output_dir="./qwen3-4b-int4-awq-onnx",
    config=export_config
)

print("ONNX export complete → ./qwen3-4b-int4-awq-onnx/")
```

**量化后的效果：**
- 原始 FP16 权重：~8GB
- INT4 AWQ 权重：~2GB
- 精度影响：极小（AWQ 在 4B+ 模型上通常 <1% 性能损失）

### 3.3 Step 2：TensorRT Engine 编译（Jetson Orin Nano）

将 ONNX 文件复制到 Jetson 后，编译为 TensorRT engine：

```bash
# 将 x86 上导出的 ONNX 复制到 Jetson
scp -r ./qwen3-4b-int4-awq-onnx user@jetson:/home/user/models/

# 在 Jetson 上编译 engine
cd ~/TensorRT-Edge-LLM/build

./bin/edge-llm-build \
    --model_dir /home/user/models/qwen3-4b-int4-awq-onnx \
    --output_dir ./qwen3-4b-engine \
    --max_batch_size 4 \
    --max_seq_len 2048 \
    --gemm_plugin int4 \          # INT4 GEMM kernel
    --use_fp8_context_fmha        # FlashAttention (FP8 KV Cache)
```

**编译时间：** 在 Orin Nano 上约 5-15 分钟（取决于模型大小和编译选项）

**关键参数说明：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--max_batch_size` | 最大批处理 | 1-4 (受限 8GB VRAM) |
| `--max_seq_len` | 最大 context 长度 | 2048-4096 |
| `--gemm_plugin` | GEMM kernel 类型 | `int4` (AWQ) |
| `--use_fp8_context_fmha` | FlashAttention + FP8 KV Cache | 节省显存 |
| `--use_gpt_attention_plugin` | GPT Attention 优化 | 推荐开启 |

### 3.4 Step 3：C++ 推理

```bash
# 交互式推理
./bin/edge-llm-run \
    --engine_dir ./qwen3-4b-engine \
    --tokenizer_dir /home/user/models/qwen3-4b-int4-awq-onnx

# 输出示例：
# [EdgeLLM] Engine loaded successfully
# [EdgeLLM] VRAM used: ~2.5 GB
# User: 你好，介绍一下你自己
# Assistant: 你好！我是通义千问，由阿里云开发的大语言模型...
# [EdgeLLM] Generated 128 tokens in 6.2s (20.6 tok/s)
```

### 3.5 预期性能

| 指标 | 数值 |
|------|------|
| 模型大小（INT4 AWQ） | ~2 GB |
| 权重 + KV Cache 总占用 | ~3-4 GB (context 2048) |
| 推理速度 (Orin Nano 8GB) | ~15-20 tok/s |
| 推理速度 (AGX Orin 64GB) | ~30-40 tok/s |

---

## 4. 示例 2：Cosmos Reason2 8B on Jetson Thor（NVFP4 VLM）

### 4.1 教程背景

Jetson AI Lab 教程的第二个示例，在 **Jetson Thor**（128GB 统一内存）上部署 Cosmos Reason2 8B VLM：

- 8B 视觉语言模型，支持图像输入 + 文本输出
- NVFP4 量化，利用 Thor Blackwell 架构的 FP4 Tensor Core
- EAGLE-3 speculative decoding 加持

### 4.2 Step 1：NVFP4 量化 + ONNX 导出（x86 宿主机）

```python
# export_cosmos_reason2.py (在 x86 宿主机上运行)

from modelopt import quantize
from tensorrt_edgellm.export import export_onnx
from transformers import AutoModelForVision2Seq, AutoProcessor

# 1. 加载 VLM 模型
model_name = "nvidia/Cosmos-Reason2-8B"
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_name)

# 2. NVFP4 量化（Blackwell 原生 FP4 支持）
quantized_model = quantize(
    model,
    method="nvfp4",        # NVFP4 量化
    calibration_dataset="coco_captions"
)

# 3. 导出 ONNX（含视觉编码器）
export_config = {
    "max_batch_size": 1,
    "max_seq_len": 8192,
    "precision": "nvfp4",
    "vision_encoder": True,     # 导出视觉编码器
}
export_onnx(
    quantized_model,
    processor,
    output_dir="./cosmos-reason2-8b-nvfp4-onnx",
    config=export_config
)
```

### 4.3 Step 2：Engine 编译 + EAGLE-3 Draft Model（Jetson Thor）

```bash
# 编译主模型 engine
./bin/edge-llm-build \
    --model_dir ./cosmos-reason2-8b-nvfp4-onnx \
    --output_dir ./cosmos-reason2-engine \
    --max_batch_size 4 \
    --max_seq_len 8192 \
    --gemm_plugin nvfp4 \          # NVFP4 GEMM kernel
    --use_fp8_context_fmha

# 编译 EAGLE-3 draft model（用于 speculative decoding）
./bin/edge-llm-build \
    --model_dir ./eagle3-cosmos-draft-onnx \
    --output_dir ./eagle3-draft-engine \
    --max_batch_size 4 \
    --is_draft_model                 # 标记为 draft model
```

### 4.4 Step 3：VLM 推理

```bash
# VLM 推理（带图像输入）
./bin/edge-llm-run \
    --engine_dir ./cosmos-reason2-engine \
    --draft_engine_dir ./eagle3-draft-engine \   # EAGLE-3 加速
    --vision_encoder_dir ./vision-encoder-onnx \
    --image_path /path/to/image.jpg

# 输出示例：
# [EdgeLLM] Vision encoder loaded
# [EdgeLLM] Main engine + draft engine loaded (speculative decoding enabled)
# User: [image: /path/to/image.jpg] Describe this image.
# Assistant: The image shows a...
# [EdgeLLM] Generated 256 tokens in 5.1s (50.2 tok/s, with EAGLE-3)
```

### 4.5 性能对比（EAGLE-3 加速效果）

| 配置 | tok/s | 加速倍数 | 官方验证 |
|------|-------|---------|---------|
| Cosmos Reason2 8B NVFP4 (base) | ~20 | 1x | ✅ |
| Cosmos Reason2 8B NVFP4 + EAGLE-3 | **~50** | **~2.5x** | ✅ |
| Llama-3.1-8B INT4 AWQ + EAGLE-3 (最优场景) | ~70 | **~3.5x** | ✅ |

### 4.6 EAGLE-3 通用性说明

**EAGLE-3 不是模型特定的功能，而是通用的 speculative decoding 框架。** 根据官方文档：

> "Any EAGLE3-compatible draft model on HuggingFace can be tried however TensorRT Edge-LLM team does not test the accuracy or acceptance rate." -- [官方 Speculative Decoding 文档](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/examples/speculative-decoding.html)

**可以用于 Qwen3-VL 吗？**

理论上可以，但需要满足：
1. **存在对应的 EAGLE-3 draft model** -- 在 HuggingFace 搜索 `eagle3 qwen3-vl`。如果存在即可直接使用
2. **Draft model 需要支持 VLM 架构** -- EAGLE-3 draft model 需要和 target model 共享 embedding/LM head，对 VLM 还需处理视觉编码器输入
3. **需自行验证效果** -- NVIDIA 官方只测试了 Cosmos Reason2 和 Llama-3.1 两个模型。Qwen3-VL 的 acceptance rate 和加速效果需要自己实测

**工作流（以 Qwen3-VL 为例）：**

```bash
# 1. x86 宿主机：导出 Qwen3-VL 主模型 ONNX + EAGLE-3 draft model ONNX
python export_qwen3_vl.py --with_eagle3

# 2. Jetson：分别编译两个 engine
./bin/edge-llm-build --model_dir ./qwen3-vl-onnx --output_dir ./qwen3-vl-engine
./bin/edge-llm-build --model_dir ./eagle3-qwen3-vl-onnx --output_dir ./eagle3-engine --is_draft_model

# 3. 推理时同时加载
./bin/edge-llm-run \
    --engine_dir ./qwen3-vl-engine \
    --draft_engine_dir ./eagle3-engine \
    --vision_encoder_dir ./vision-encoder-onnx \
    --image_path test.jpg
```

**注意：** 如果 HuggingFace 上没有现成的 Qwen3-VL EAGLE-3 draft model，需要自己用 EAGLE 训练框架（https://github.com/SafeAILab/EAGLE）训练一个，训练成本约几百 GPU 小时

---

## 5. Python Export Pipeline 详解

### 5.1 管线架构

TensorRT Edge-LLM 的 Python Export Pipeline 是一个**多层抽象系统**，将 HuggingFace 模型逐步转换为 TensorRT Engine 可用的 ONNX 格式：

```
Layer 0: HuggingFace Model (PyTorch)
    │
    ▼ modelopt (量化)
Layer 1: Quantized Model (PyTorch + AWQ/NVFP4)
    │
    ▼ tensorrt_edgellm (导出)
Layer 2: ONNX Graph (linearized graph)
    │
    ▼ edge-llm-build (编译，Jetson 上执行)
Layer 3: TensorRT Engine (optimized binary)
```

### 5.2 量化策略选择

| 量化方法 | 精度 | 模型大小 | 支持 GPU | 最佳场景 |
|---------|------|---------|---------|---------|
| **INT4 AWQ** | 高 | 25% 原始大小 | 所有 NVIDIA GPU | 通用 LLM 部署 |
| **INT8 SmoothQuant** | 极高 | 50% 原始大小 | 所有 NVIDIA GPU | 对精度敏感的模型 |
| **NVFP4** | 非常接近 FP16 | 25% 原始大小 | Blackwell (B200/Thor) | 极致性能 + 节省 |
| **FP8** | 接近 FP16 | 50% 原始大小 | Hopper+ (H100/Thor) | 数据中心到边缘迁移 |

### 5.3 ONNX 导出配置模板

```python
export_config = {
    # 基础配置
    "max_batch_size": 4,          # 最大批处理大小
    "max_seq_len": 4096,          # 最大序列长度
    "precision": "int4",          # 精度: int4/int8/nvfp4/fp16

    # 量化配置
    "quant_method": "awq",        # awq / smoothquant / nvfp4
    "group_size": 128,            # AWQ group size
    "calibration_dataset": "pileval",

    # 高级特性
    "vision_encoder": False,      # VLM 需要设为 True
    "lora_adapters": None,        # LoRA adapter 路径
    "eagle3_draft": False,        # 是否为 EAGLE-3 draft model
}
```

### 5.4 支持的模型架构

| 模型系列 | LLM 支持 | VLM 支持 | 量化选项 |
|---------|---------|---------|---------|
| Qwen3/3.5/3.6 | ✅ | ✅ | AWQ, NVFP4 |
| Llama 3/4 | ✅ | ✅ (3.2-V) | AWQ, NVFP4 |
| Gemma 3/4 | ✅ | ✅ | AWQ |
| InternVL3/3.5 | ❌ | ✅ | AWQ, NVFP4 |
| Phi-4-Multimodal | ❌ | ✅ | AWQ |
| Nemotron-Nano | ✅ | ✅ | FP8, NVFP4 |
| Cosmos Reason2 | ❌ | ✅ | NVFP4 |

---

## 6. C++ Runtime 部署

### 6.1 架构

C++ Runtime 是 TensorRT Edge-LLM 的核心差异点 — **推理路径全程无 Python**：

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Python Host  │   │  C++ Runtime  │   │   TensorRT    │
│ (一次性使用)   │   │ (每次推理)     │   │   Engine      │
├──────────────┤   ├──────────────┤   ├──────────────┤
│ 量化 + 导出   │──▶│ 加载 Engine   │──▶│ 权重 + graph  │
│ 写 ONNX      │   │              │   │              │
└──────────────┘   │ 管理 KV Cache │   │ CUDA Kernel   │
                   │ Tokenizer     │   │ 执行推理      │
                   │ HTTP Server   │   └──────────────┘
                   └──────────────┘
```

### 6.2 edge-llm-server（HTTP 服务）

```bash
# 启动 HTTP 推理服务
./bin/edge-llm-server \
    --engine_dir ./qwen3-4b-engine \
    --port 8000 \
    --max_batch_size 4

# curl 调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好，介绍一下你自己",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### 6.3 Orin Nano 8GB 内存预算

```
总统一内存：8 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

系统开销 (JetPack)：    ~1.5 GB
TensorRT Engine 权重：   ~2.0 GB (Qwen3-4B INT4 AWQ)
KV Cache (2048 token)：  ~1.5 GB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
可用余量：               ~3.0 GB

精确控制 KV Cache 大小：
--max_seq_len 1024  → KV Cache ~0.8 GB
--max_seq_len 2048  → KV Cache ~1.5 GB
--max_seq_len 4096  → KV Cache ~3.0 GB (极限，系统可能 OOM)
```

### 6.4 与 TensorRT-LLM 的关键区别

| 维度 | TensorRT Edge-LLM | TensorRT-LLM |
|------|-------------------|-------------|
| 目标平台 | Jetson / DRIVE / DGX Spark | 数据中心 / 桌面 GPU |
| 推理路径 | 纯 C++ | Python (v1.0+) 或 C++ |
| 模型导出 | ONNX → 任何 Jetson 编译 | 专用 checkpoint 格式 |
| **量化** | nvidia-modelopt (pip) / modelopt (import) | 同上 |
| 启动时间 | 毫秒级（C++ 直载） | Python 解释器启动 5-10s |
| 内存占用 | 极低（无 Python 运行时） | 较高（Python + 库占用 ~500MB） |
| Spec. Dec. | EAGLE-3 (原生支持) | EAGLE/Medusa (支持) |

---

## 参考链接

- Jetson AI Lab 教程：https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- TensorRT Edge-LLM GitHub：https://github.com/NVIDIA/TensorRT-Edge-LLM
- 官方文档：https://nvidia.github.io/TensorRT-Edge-LLM/
- Python Export Pipeline：https://nvidia.github.io/TensorRT-Edge-LLM/developer_guide/03.1_Python_Export_Pipeline.html
- 安装指南：https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html
- NVIDIA 开发者博客：https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/
- Seeed Studio 实战教程：https://wiki.seeedstudio.com/deploy_tensorrt_edge_llm_on_jetpack6.2/
