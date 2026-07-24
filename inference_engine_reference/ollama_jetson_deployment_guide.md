# Jetson Orin 上运行 Ollama 完整部署指南

> 基于两篇文章整理：
> 1. [Jetson AI Lab - Ollama on Jetson](https://www.jetson-ai-lab.com/tutorials/ollama/) (NVIDIA 官方教程)
> 2. [Running Ollama on a Jetson Orin Nano: From Gemma 3 to Gemma 4 with GPU Acceleration](https://julien.cloud/blog/jetson-nano-ollama-edge-inference/) (Julien 社区实践, 2026-06-09)
> 说明：部分技术细节根据 Ollama 官方文档和 dusty-nv/jetson-containers 项目补充

---

## 目录

1. [Ollama on Jetson 概述](#1-ollama-on-jetson-概述)
2. [方式一：Jetson AI Lab 官方容器（推荐）](#2-方式一jetson-ai-lab-官方容器推荐)
3. [方式二：原生安装](#3-方式二原生安装)
4. [Julien 的实践：从 Gemma 3 到 Gemma 4](#4-julien-的实践从-gemma-3-到-gemma-4)
5. [模型运行与管理](#5-模型运行与管理)
6. [性能数据与选型建议](#6-性能数据与选型建议)

---

## 1. Ollama on Jetson 概述

### 1.1 为什么选择 Ollama

Ollama 是目前最简化的本地 LLM 部署工具：
- **一行命令安装**，零配置启动
- **自动量化选择**：自动适配 VRAM，选择最优 GGUF 量化级别
- **模型仓库**：类似 Docker Hub 的 pull/run 体验
- **OpenAI 兼容 API**：可直接对接 Open-WebUI、LangChain 等
- **基于 llama.cpp**：底层是 llama.cpp CUDA 后端，在 Jetson 上得到充分测试

### 1.2 Jetson 平台兼容性

| Jetson 设备 | JetPack 版本 | 推荐方式 | 容器镜像 |
|------------|-------------|---------|---------|
| **Orin Nano/NX/AGX** (JP6) | JetPack 6.x (r36.x) | Docker 容器 | `dustynv/ollama:r36.2.0` |
| **Orin Nano/NX/AGX** (JP6, CUDA 13) | JetPack 6.x | Docker 容器 | `dustynv/ollama:r36.4.0-cu130` |
| **AGX Orin (JP5)** | JetPack 5.x (r35.x) | Docker 容器 | `dustynv/ollama:r35.2.1` |
| **Jetson Thor / DGX Spark** | SBSA (ARM Server) | Docker 容器 | `ghcr.io/nvidia-ai-iot/ollama:r38.2.arm64-sbsa-cu130-24.04` |

---

## 2. 方式一：Jetson AI Lab 官方容器（推荐）

### 2.1 安装 Docker + NVIDIA Container Runtime

```bash
# 安装 Docker（如果还没有）
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo usermod -aG docker $USER

# 安装 NVIDIA Container Runtime
sudo apt install -y nvidia-container-runtime
sudo systemctl restart docker

# 验证
docker run --rm --runtime=nvidia nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
```

### 2.2 拉取并启动 Ollama 容器

```bash
# JetPack 6 (Jetson Orin Nano / NX / AGX)
docker pull dustynv/ollama:r36.2.0

# 启动 Ollama 服务
docker run -d \
    --runtime nvidia \
    --name ollama \
    --network host \
    -v ollama_data:/ollama/.ollama \
    dustynv/ollama:r36.2.0

# 或者一次性运行（前台）
docker run --runtime nvidia -it --rm \
    --network host \
    -v ollama_data:/ollama/.ollama \
    dustynv/ollama:r36.2.0
```

**参数说明：**
- `--runtime nvidia`：启用 GPU 加速（关键！否则 CPU-only）
- `--network host`：Ollama 默认监听 11434 端口，host 模式避免端口映射问题
- `-v ollama_data:/ollama/.ollama`：持久化下载的模型文件

### 2.3 运行模型

```bash
# 进入容器交互
docker exec -it ollama ollama run llama3.2:3b

# 或从宿主机直接调用
docker exec ollama ollama run qwen3:4b
```

### 2.4 GPU 加速验证

```bash
# 查看 GPU 使用情况
docker exec ollama nvidia-smi

# 下载并运行模型后，观察 GPU 内存是否被占用
docker exec ollama ollama run llama3.2:3b "Hello, are you using GPU?"
# 另开终端
watch -n 1 nvidia-smi
```

### 2.5 容器镜像版本选择

| 镜像标签 | JetPack | CUDA | 适用设备 |
|---------|---------|------|---------|
| `dustynv/ollama:r36.2.0` | 6.0/6.1 | 12.2 | Orin Nano/NX/AGX |
| `dustynv/ollama:r36.4.0` | 6.2 | 12.6 | Orin Nano/NX/AGX |
| `dustynv/ollama:r36.4.0-cu130` | 6.2+ | 13.0 | 同上（最新 CUDA） |
| `dustynv/ollama:r35.4.1` | 5.1.3 | 11.4 | AGX Orin (旧 JetPack) |

> dusty-nv/jetson-containers 项目持续更新这些镜像，最新版本查看：https://github.com/dusty-nv/jetson-containers

---

## 3. 方式二：原生安装

### 3.1 安装 Ollama

```bash
# Jetson 上直接用官方安装脚本（Ollama 会自动检测 aarch64）
curl -fsSL https://ollama.com/install.sh | sh

# 验证
ollama --version

# 启动服务
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 3.2 GPU 驱动确认

```bash
# 确保 JetPack 的 CUDA 驱动正常工作
nvidia-smi

# 如果显示 "NVIDIA-SMI has failed"，需要重新安装驱动
# 通常刷机时已自带，不需要额外操作
```

### 3.3 原生安装的限制

- Ollama 官方 ARM64 支持主要针对 Apple Silicon (M1/M2/M3/M4)
- Jetson 上原生安装可能缺少某些 GPU 优化
- **推荐使用 dusty-nv 的 Docker 容器**，因为：
  - 已经针对 Jetson 架构编译了 llama.cpp CUDA 后端
  - 自动配置了正确的 CUDA 库路径
  - 避免了 JetPack 版本不匹配的问题

---

## 4. Julien 的实践：从 Gemma 3 到 Gemma 4

### 4.1 背景

Julien 在 Jetson Orin Nano 上部署 Ollama 的历程，揭示了 Jetson 平台的一些典型坑。

### 4.2 第一阶段：Gemma 3 4B（CPU only）

```
设备：Jetson Orin Nano (JetPack 5)
模型：Gemma 3 4B (Q4_K_M)
性能：17.5 tok/s
状态：CPU only — GPU 完全空闲
```

**问题：** 官方 Ollama 二进制在 JetPack 5 上已经不再支持 GPU 加速。虽然模型能跑（17.5 tok/s 还算可用），但 GPU 完全浪费。

### 4.3 第二阶段：GPU 驱动折腾

```bash
# 尝试强制启用 GPU
# 发现 JetPack 5 的 GPU 驱动与新版 Ollama 的 llama.cpp 不兼容
# 需要升级到 JetPack 6 或使用 dusty-nv 的容器

# Julien 最终方案：使用 dusty-nv 容器
docker pull dustynv/ollama:r36.2.0
```

**教训：** JetPack 版本与 Ollama GPU 后端的兼容性是关键。JetPack 5 用户必须用容器，JetPack 6 用户原生安装也需要谨慎。

### 4.4 第三阶段：Gemma 4 E2B（GPU 加速成功）

```
设备：Jetson Orin Nano (JetPack 6, dusty-nv 容器)
模型：Gemma 4 E2B (2.3B effective, Q4_K_M)
性能：25.5 tok/s (GPU 加速)
```

**关键数据：**
- Gemma 4 E2B 是 Google 2026 年初发布的边缘专用变体
- 2.3B effective parameters（MoE 架构，实际只激活部分参数）
- 在 Orin Nano 8GB 上 GPU 加速达到 25.5 tok/s
- "from zero swap, stable for months" — 长期运行稳定，无内存泄漏

### 4.5 Gemma 4 边缘变体

| 模型 | 有效参数 | 推荐硬件 | 场景 |
|------|---------|---------|------|
| **Gemma 4 E2B** | 2.3B | Orin Nano 4GB+ | 轻量对话、边缘 AI |
| **Gemma 4 E4B** | 4.3B | Orin Nano 8GB+ | 复杂推理、RAG |
| Gemma 4 26B | 26B | AGX Orin 64GB | 本地 Agent |
| Gemma 4 31B | 31B | Jetson Thor 128GB | 生产级边缘 AI |

---

## 5. 模型运行与管理

### 5.1 下载与运行

```bash
# 拉取模型（自动选择最佳量化级别）
docker exec ollama ollama pull qwen3:4b
docker exec ollama ollama pull llama3.2:3b
docker exec ollama ollama pull gemma4:2b

# 交互式运行
docker exec -it ollama ollama run qwen3:4b

# 查看已下载模型
docker exec ollama ollama list
```

### 5.2 API 服务

Ollama 默认在 11434 端口提供 OpenAI 兼容 API：

```bash
# curl 测试
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Python 客户端
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### 5.3 集成 Open-WebUI

```bash
# 使用 jetson-containers 启动 Open-WebUI
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers

# 启动 Open-WebUI（自动连接本地 Ollama）
./run.sh --workdir=/opt/open-webui \
    --env OLLAMA_BASE_URL=http://localhost:11434 \
    $(./autotag open-webui)
```

### 5.4 Modelfile：自定义模型

```bash
# 创建 Modelfile
cat > Modelfile << 'EOF'
FROM qwen3:4b

# 设置 system prompt
SYSTEM "You are a helpful Jetson AI assistant."

# 设置参数
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
EOF

# 构建自定义模型
docker exec ollama ollama create my-jetson-assistant -f Modelfile

# 运行
docker exec ollama ollama run my-jetson-assistant
```

---

## 6. 性能数据与选型建议

### 6.1 Jetson Orin Nano 实测数据

| 模型 | 量化 | 引擎 | 性能 (tok/s) | 来源 |
|------|------|------|-------------|------|
| Gemma 3 4B | Q4_K_M | Ollama (CPU) | 17.5 | Julien 实测 |
| Gemma 4 E2B | Q4_K_M | Ollama (GPU) | **25.5** | Julien 实测 |
| Llama 3.2 3B | Q4_K_M | Ollama | ~20 | Jetson AI Lab |
| Qwen3 4B | Q4_K_M | Ollama | ~15 | 社区 benchmark |
| Gemma 3 1B | Q4_0 | Ollama | ~45 | Jetson AI Lab |
| Llama 3.1 8B | Q4_K_M | Ollama | ~12-15 | SpecPicks |

### 6.2 Orin Nano vs AGX Orin

| 设备 | Gemma 4 E2B (估) | Qwen3 4B (估) | Llama 3.1 8B (估) |
|------|-----------------|--------------|-------------------|
| **Orin Nano 8GB** | ~25 tok/s | ~15 tok/s | ~12 tok/s |
| **AGX Orin 64GB** | ~50 tok/s | ~30 tok/s | ~21 tok/s |

> AGX Orin 约 2x 速度提升（带宽 204.8 vs ~68 GB/s）。更大的优势是 64GB 统一内存可以加载更大的模型（14B+）。

### 6.3 Ollama vs 直接使用 llama.cpp

| 维度 | Ollama | llama.cpp 直接使用 |
|------|--------|-------------------|
| 安装 | 一行命令 | 需源码编译 |
| 模型管理 | `ollama pull/run`，自动量化选择 | 手动下载 GGUF 文件 |
| GPU 加速 | 自动检测 | 需编译时配置（-DGGML_CUDA=ON） |
| API | 内置 OpenAI 兼容 | llama-server 手动配置 |
| 灵活性 | 较低（封装层次高） | 高（直接控制所有参数） |
| Jetson 兼容 | 推荐用 dusty-nv 容器 | 从源码编译 + 正确设置 CUDA_ARCH |

**建议：** 快速验证用 Ollama 容器，追求极致性能或用自定义模型用 llama.cpp 直接编译。

### 6.4 选型决策

```
你的 Jetson 设备是什么？
│
├── Orin Nano 4GB/8GB
│   ├── 快速开始？ → dusty-nv/ollama 容器 + Gemma 4 E2B 或 Qwen3 4B
│   ├── 追求性能？ → llama.cpp 源码编译 + Q4_K_M 量化
│   └── CPU only 可行吗？ → 可以（~17 tok/s），但浪费 GPU
│
├── Orin NX 16GB
│   ├── 同 Nano，但可跑更大模型（8B 轻松）
│   └── 推荐 Qwen3 8B Q4_K_M 或 Llama 3.1 8B
│
└── AGX Orin 64GB
    ├── 64GB 统一内存是巨大优势
    ├── 可跑 14B-32B 模型（Q4 量化）
    ├── Ollama 容器或 llama.cpp 均可
    └── 推荐 Qwen3 14B/32B Q4 或 Gemma 4 26B
```

---

## 参考链接

- Jetson AI Lab - Ollama 教程：https://www.jetson-ai-lab.com/tutorials/ollama/
- Julien 实践 - Gemma 3 → Gemma 4：https://julien.cloud/blog/jetson-nano-ollama-edge-inference/
- dusty-nv/jetson-containers：https://github.com/dusty-nv/jetson-containers
- Ollama 官方文档：https://docs.ollama.com
- NVIDIA Gemma 4 on Jetson：https://developer.nvidia.com/blog/bringing-ai-closer-to-the-edge-and-on-device-with-gemma-4/
