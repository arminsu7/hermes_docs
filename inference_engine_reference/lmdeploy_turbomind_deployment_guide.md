# LMDeploy TurboMind 部署技术指南

> 基于 Spheron 博客文章整理并补充：https://www.spheron.network/blog/deploy-lmdeploy-gpu-cloud-turbomind-inference/
> 原文发布于 2026年6月1日
> 说明：部分技术细节根据 LMDeploy 官方文档补充了标准操作流程

---

## 目录

1. [LMDeploy 概述](#1-lmdeploy-概述)
2. [安装与环境准备](#2-安装与环境准备)
3. [模型部署实战](#3-模型部署实战)
4. [AWQ INT4 量化部署](#4-awq-int4-量化部署)
5. [MXFP4 量化部署](#5-mxfp4-量化部署)
6. [生产调优](#6-生产调优)

---

## 1. LMDeploy 概述

### 1.1 是什么

LMDeploy 是 InternLM 团队开发的开源 LLM 部署工具，提供两个推理引擎：

| 引擎 | 语言 | 定位 |
|------|------|------|
| **TurboMind** | C++ | 极致推理性能，面向生产 |
| **PyTorch** | Python | 降低开发门槛，快速验证 |

本文聚焦 TurboMind 引擎的 GPU 云端部署。

### 1.2 核心性能数据（官方 benchmark）

| 场景 | 对比基准 | 提升 | 来源版本 |
|------|---------|------|---------|
| GPT-OSS 120B MXFP4 on H100 | vLLM | **5x** 吞吐量 | LMDeploy v0.10.0 |
| GPT-OSS 120B MXFP4 on H800 | vLLM | **1.5x** 吞吐量 | LMDeploy v0.10.0 |
| 通用场景（官方声明） | vLLM | **最高 1.8x** 吞吐量 | README |

> **注意：** 1.5x 是特定场景（H800 + MXFP4 + gpt-oss）的实测数据，1.8x 是官方 README 声明的通用上限。不同模型/硬件/量化组合的实际提升不同。

### 1.3 TurboMind vs PyTorch 后端对比

| 维度 | TurboMind (C++) | PyTorch |
|------|----------------|---------|
| 性能 | 最高 | 中等 |
| 安装 | pip install 即可 | pip install 即可 |
| 量化支持 | AWQ, GPTQ, MXFP4, KV Cache 量化 | 有限 |
| 适合场景 | 生产部署 | 开发测试 |
| 预热时间 | 首次启动稍慢（GEMM 调优） | 较快 |

---

## 2. 安装与环境准备

### 2.1 基本安装

```bash
# 创建 conda 环境（推荐）
conda create -n lmdeploy python=3.11 -y
conda activate lmdeploy

# pip 安装（CUDA 12.x）
pip install lmdeploy        # 默认 TurboMind + PyTorch
pip install lmdeploy[all]   # 含所有可选依赖

# 验证
lmdeploy --version
```

**CUDA 版本兼容性：**
- 截至 2026年7月，预编译 wheel 基于 CUDA 12.8
- 支持 Ampere (SM 8.0+)、Hopper (SM 9.0)、Blackwell (SM 12.0)
- v0.13.0+ 已通过 pip 直接支持 RTX 50 系列

### 2.2 GPU 选型建议（Spheron 推荐）

| 模型规模 | 推荐 GPU | 显存 | 说明 |
|---------|---------|------|------|
| 7B-13B | A100 40GB / RTX 4090 24GB | 24-40GB | 单卡 AWQ/MXFP4 轻松 |
| 20B-32B | A100 80GB / H100 80GB | 80GB | AWQ 可单卡 |
| 70B+ | H100 80GB x2 / B200 | 多卡 | 需要 TP |

### 2.3 验证 GPU 环境

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# TurboMind 快速测试
lmdeploy check_env
```

---

## 3. 模型部署实战

### 3.1 一键启动（PyTorch 后端）

最简单的启动方式——直接从 HuggingFace 拉取模型：

```bash
# InternLM2.5-7B
lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333

# Qwen3-8B
lmdeploy serve api_server Qwen/Qwen3-8B-Instruct --server-port 23333

# Qwen3-Coder-7B
lmdeploy serve api_server Qwen/Qwen3-Coder-7B --backend turbomind --tp 1 --server-port 8000
```

### 3.2 使用 TurboMind 后端

```bash
# 指定 TurboMind 后端 + tensor parallel
lmdeploy serve api_server internlm/internlm2_5-7b-chat \
    --backend turbomind \
    --tp 1 \
    --server-port 23333 \
    --max-batch-size 256
```

### 3.3 多卡 Tensor Parallel (TP)

DeepSeek-V3 等 MoE 大模型需要多卡：

```bash
# DeepSeek-V3 MoE（示例，实际需要大量 GPU）
lmdeploy serve api_server deepseek-ai/DeepSeek-V3 \
    --backend turbomind \
    --tp 8 \
    --max-batch-size 64 \
    --session-len 4096
```

### 3.4 验证 API 服务

```bash
# curl 测试
curl http://localhost:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm/internlm2_5-7b-chat",
    "messages": [{"role": "user", "content": "Hello! Introduce yourself."}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### 3.5 Python 客户端调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:23333/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="internlm/internlm2_5-7b-chat",
    messages=[
        {"role": "user", "content": "What is LMDeploy?"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

---

## 4. AWQ INT4 量化部署

### 4.1 什么是 AWQ

AWQ (Activation-aware Weight Quantization) 是 LMDeploy 主推的 4-bit 量化方案：
- 权重 4-bit，激活保持 FP16
- 精度损失极小（通常 <1% benchmark 差异）
- TurboMind 专为 AWQ 优化，C++ kernel 直接加载 awq 格式

### 4.2 使用预量化 AWQ 模型

```bash
# 直接使用 HuggingFace 上的 AWQ 量化模型
lmdeploy serve api_server Qwen/Qwen3-8B-Instruct-AWQ \
    --backend turbomind \
    --model-format awq \
    --server-port 23333
```

### 4.3 自己量化模型（auto_awq）

```bash
# 使用 LMDeploy 内置的 AWQ 量化工具
lmdeploy lite auto_awq \
    Qwen/Qwen3-8B-Instruct \
    --work-dir ./qwen3-8b-awq \
    --calib-dataset c4 \
    --calib-samples 128 \
    --calib-seqlen 2048
```

### 4.4 AWQ 性能数据

| 模型 | FP16 (基准) | AWQ INT4 | 提速 |
|------|------------|---------|------|
| InternLM2-7B | 1x | ~2.4x | **2.4x** |
| 通用 4-bit vs FP16 | 1x | ~2.4x | **2.4x**（LMDeploy 官方 PDF 文档数据） |

> 数据来源：LMDeploy 官方 PDF 文档 `lmdeploy.readthedocs.io/_/downloads/en/latest/pdf/` -- "4-bit inference performance is 2.4x higher than FP16"。在 memory-bound 场景（如小模型大 batch）下效果最显著。实际提速取决于模型大小、batch size、硬件带宽。

---

## 5. MXFP4 量化部署

### 5.1 什么是 MXFP4

MXFP4 (Microscaling FP4) 是 NVIDIA 与行业联合提出的 4-bit 浮点量化标准：
- 每 32 个权重共享一个 scaling factor
- 精度远高于普通 INT4，接近 FP8
- 需要 Blackwell (B200/B300) 的 FP4 Tensor Core 才能发挥硬件加速

### 5.2 MXFP4 支持的 GPU

| GPU | MXFP4 加速类型 | 效果 |
|-----|---------------|------|
| B200 / B300 (Blackwell 数据中心) | **原生 FP4 Tensor Core** | 最佳：同时获得显存节省 + 硬件加速 |
| H100 / H800 (Hopper) | 软件模拟 | 获得显存节省，无硬件加速 |
| RTX 5090 (Blackwell 消费级) | 软件模拟 | 获得显存节省，FP4 Tensor Core 加速有限 |
| V100+ (Volta+) | 软件模拟 | 仅显存节省，性能提升不明显 |

### 5.3 部署 MXFP4 模型

```bash
# 使用 MXFP4 量化（需要先用量化工具生成 MXFP4 权重）
lmdeploy serve api_server ./qwen3-8b-mxfp4 \
    --backend turbomind \
    --quant-policy 4 \
    --server-port 23333
```

### 5.4 MXFP4 性能对比（Spheron 数据）

| 模型 | GPU | 量化 | 与 vLLM 对比 |
|------|-----|------|-------------|
| GPT-OSS 120B | H100 | MXFP4 | **5x** 吞吐量 |
| GPT-OSS 120B | H800 | MXFP4 | **1.5x** 吞吐量 |
| GPT-OSS 120B | B200 | MXFP4 | >5x（含 FP4 Tensor Core 加速） |

---

## 6. 生产调优

### 6.1 关键参数

```bash
lmdeploy serve api_server Qwen/Qwen3-8B-Instruct \
    --backend turbomind \
    --tp 1 \                      # Tensor Parallel 数
    --max-batch-size 256 \         # 最大并发批处理
    --session-len 8192 \           # 最大 context length
    --cache-max-entry-count 0.8 \  # KV Cache 最大占用比例
    --model-format awq \           # 量化格式
    --server-port 23333
```

**参数详解：**

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--tp` | Tensor Parallel 卡数 | 按显存需求 |
| `--max-batch-size` | 最大批处理大小 | 256（高并发） / 64（低并发） |
| `--session-len` | 最大 context 长度 | 4096-8192（按需求） |
| `--cache-max-entry-count` | KV Cache 缓存比例 | 0.5-0.8 |
| `--model-format` | 模型格式 | `awq` / `hf` / `llama` |

### 6.2 首次启动预热

TurboMind C++ 后端的首次启动会自动进行 GEMM 调优（约 1-2 分钟），后续启动会缓存调优结果。

```bash
# 首次启动稍慢（GEMM 调优）
# 看到这条日志说明调优完成：
# [TM][INFO] GEMM tuning finished, config saved to ./workspace/triton_models/weights/

# 第二次启动会自动加载缓存
```

### 6.3 监控与 Metrics

```bash
# 启用 metrics（默认关闭）
lmdeploy serve api_server Qwen/Qwen3-8B-Instruct \
    --backend turbomind \
    --metrics
```

### 6.4 Docker 部署

```bash
# 使用官方 Docker 镜像
docker run --gpus all \
    -v /path/to/models:/models \
    -p 23333:23333 \
    internlm/lmdeploy:latest \
    lmdeploy serve api_server /models/qwen3-8b \
    --backend turbomind \
    --server-port 23333
```

### 6.5 冷启动延迟对比

| 引擎 | 冷启动（首次推理） |
|------|-------------------|
| LMDeploy TurboMind | ~30-60秒（含 GEMM 调优） |
| vLLM | ~5-10秒 |
| TensorRT-LLM (旧) | ~15-45分钟（engine 编译） |

> LMDeploy 无预编译步骤，启动时间远快于旧版 TensorRT-LLM，略慢于 vLLM。

---

## 参考链接

- 原文：https://www.spheron.network/blog/deploy-lmdeploy-gpu-cloud-turbomind-inference/
- LMDeploy GitHub：https://github.com/InternLM/lmdeploy
- LMDeploy 文档：https://lmdeploy.readthedocs.io
- LMDeploy PyPI：https://pypi.org/project/lmdeploy/
- Spheron LMDeploy 快速指南：https://docs.spheron.ai/quick-guides/llms/frameworks/lmdeploy
