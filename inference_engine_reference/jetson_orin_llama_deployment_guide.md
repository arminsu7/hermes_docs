# Jetson Orin 从零部署 LLM 完整技术指南

> 基于 Lucas Calje 两篇 Medium 文章整理
> 原文 1：[Flashing NVIDIA Jetson Orin: Nano and AGX (CLI + NVMe Guide)](https://calje.medium.com/flashing-nvidia-jetson-orin-nano-and-agx-cli-nvme-guide-11d95e08a65d) (2026-03-25)
> 原文 2：[Getting Started with LLMs on NVIDIA Jetson Orin](https://calje.medium.com/getting-started-with-llms-on-nvidia-jetson-orin-ee3a80096510) (2026-05-08)
> 说明：部分技术细节根据原文思路补充了标准操作流程，并非逐句翻译

---

## 目录

1. [Part 1：CLI 方式刷写 JetPack（Nano + AGX）](#part-1cli-方式刷写-jetpacknano--agx)
2. [Part 2：llama.cpp + Open-WebUI 部署 LLM](#part-2llamacpp--open-webui-部署-llm)
3. [关键性能数据](#关键性能数据)
4. [踩坑与注意事项](#踩坑与注意事项)

---

## Part 1：CLI 方式刷写 JetPack（Nano + AGX）

### 1.1 目标设备

| 设备 | 型号 | 存储方案 |
|------|------|---------|
| Jetson Orin Nano Super | 8GB | NVMe SSD |
| Jetson AGX Orin | 64GB | NVMe SSD |

作者使用 Jetson Linux 36.4.4 版本，两台设备用完全相同的版本和 CLI 工作流。

### 1.2 为什么要用 CLI 而不是 SDK Manager GUI？

- **CLI 更可控**：刷写过程的每一步都是显式的，出错容易排查
- **可复现**：命令行可以写成脚本，批量部署多台设备
- **不需要图形界面**：适合 headless 服务器、远程操作

### 1.3 准备工作

**宿主机要求：**
- Ubuntu 22.04 或 24.04 x86_64 主机
- USB-C 数据线（连接 Jetson 和宿主机）
- Jetson 进入 Recovery Mode（用跳线/按钮操作）

**下载 L4T BSP：**

```bash
# 下载 Jetson Linux 36.4.4 BSP 包
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/jetson_linux_r36.4.4_aarch64.tbz2

# 下载 rootfs
wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.4/release/tegra_linux_sample-root-filesystem_r36.4.4_aarch64.tbz2

# 解压
tar xf jetson_linux_r36.4.4_aarch64.tbz2
cd Linux_for_Tegra
sudo tar xpf ../tegra_linux_sample-root-filesystem_r36.4.4_aarch64.tbz2

# 安装 NVIDIA 驱动和工具
sudo ./apply_binaries.sh
```

### 1.4 刷写 Orin Nano Super 8GB 到 NVMe

Orin Nano Super 8GB 没有 eMMC，必须从 NVMe 或 SD 卡启动。作者选择了 NVMe。

```bash
cd Linux_for_Tegra

# Nano Super：使用 initrd flash 方式刷到 NVMe
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
    --external-device nvme0n1p1 \
    -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
    -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
    --showlogs \
    --network usb0 \
    jetson-orin-nano-devkit nvme0n1p1
```

**关键参数说明：**
- `--external-device nvme0n1p1`：目标存储为 NVMe SSD 第一个分区
- `flash_l4t_t234_nvme.xml`：T234 SoC 的 NVMe 分区表配置
- `flash_t234_qspi.xml`：QSPI 启动固件烧录
- `--network usb0`：通过 USB 虚拟网口传输
- `jetson-orin-nano-devkit`：设备配置文件，Nano 用这个（NX 也用这个）

### 1.5 刷写 AGX Orin 64GB 到 NVMe

AGX Orin 自带 64GB eMMC，但作者选择 NVMe 以获得更大的存储空间和更好的读写性能。

```bash
cd Linux_for_Tegra

# AGX Orin：刷到 NVMe
sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
    --external-device nvme0n1p1 \
    -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
    -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
    --showlogs \
    --network usb0 \
    jetson-agx-orin-devkit nvme0n1p1
```

**差异：** AGX 的设备配置文件是 `jetson-agx-orin-devkit`（Nano 用 `jetson-orin-nano-devkit`）。

### 1.6 刷写后的配置

```bash
# 首次登录（通过串口或 SSH）
# 默认用户名: jetson，密码: jetson

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3-pip git cmake curl

# 设置交换空间（Nano 8GB 可能不够）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 开启电源模式（Super / MAXN）
sudo nvpmodel -m 0         # 0 = MAXN Super (Nano), 0 = MAXN (AGX)
sudo jetson_clocks          # 锁定最高频率
```

### 1.7 验证刷写结果

```bash
# 检查 JetPack 版本
cat /etc/nv_tegra_release

# 检查 CUDA 版本
nvcc --version

# 检查 GPU 状态
tegrastats
```

---

## Part 2：llama.cpp + Open-WebUI 部署 LLM

### 2.1 架构总览

```
┌──────────────────────────────────────────────┐
│                Open-WebUI                     │
│         (浏览器前端 / Chat UI)                 │
└──────────────────┬───────────────────────────┘
                   │ OpenAI-compatible API
                   │ (HTTP POST /v1/chat/completions)
┌──────────────────┴───────────────────────────┐
│              llama-server                     │
│      (llama.cpp 内置 HTTP 服务器)              │
│      提供 OpenAI-compatible API               │
└──────────────────┬───────────────────────────┘
                   │ GGUF 模型加载 + CUDA 推理
┌──────────────────┴───────────────────────────┐
│           Jetson Orin (ARM64 + GPU)           │
│         Nano Super 8GB / AGX 64GB            │
└──────────────────────────────────────────────┘
```

### 2.2 编译 llama.cpp（CUDA 后端）

```bash
# 克隆仓库
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 编译（CUDA 后端，适应 Jetson ARM64）
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \    # SM 8.7 = Orin
    -DGGML_NATIVE=OFF                  # 跨平台编译

cmake --build build --config Release -j$(nproc)
```

**注意事项：**
- Orin 的 compute capability 是 SM 8.7（不是 8.6 或 8.9）
- 使用 `-DGGML_NATIVE=OFF` 避免 x86 宿主机编译时的不兼容指令
- 编译产物在 `build/bin/` 下，主程序是 `llama-server`

### 2.3 下载 GGUF 模型

作者选用的是 Qwen2.5 或 Llama 3.2 系列的量化模型（从 HuggingFace 下载 GGUF 格式）。

```bash
# 典型下载方式（以 Qwen2.5-7B Q4_K_M 为例）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir ./models

# 或者直接用 wget/curl 从 HuggingFace 下载
```

**GGUF 量化级别对照：**

| 量化级别 | 相对大小 | 精度 | 适用场景 |
|---------|---------|------|---------|
| Q4_0 | 最小 | 最低精度 | 内存受限（Nano 8GB） |
| Q4_K_M | 中等 | 较好 | **推荐折中选择** |
| Q5_K_M | 较大 | 更好 | 内存宽裕时 |
| Q8_0 | 大 | 接近 FP16 | AGX 64GB |

### 2.4 启动 llama-server

```bash
# 后台启动 llama-server，提供 OpenAI 兼容 API
./build/bin/llama-server \
    -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \              # 所有层加载到 GPU（99 足够覆盖全部层）
    -c 4096 \              # context length
    --cont-batching \      # 开启 continuous batching
    --parallel 2           # 允许 2 个并发请求
```

**参数说明：**
- `-ngl 99`：将 99 层加载到 GPU（7B 模型总共约 32 层，99 足够覆盖全部）
- `-c 4096`：context window 大小
- `--cont-batching`：continuous batching，多请求并发处理
- `--parallel 2`：允许同时处理 2 个请求

### 2.5 安装 Open-WebUI

**方式一：Docker（推荐）**

```bash
# 拉取 Open-WebUI
docker pull ghcr.io/open-webui/open-webui:main

# 启动（连接到本地 llama-server）
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
    -e OPENAI_API_KEY=not-needed \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

**注意：** Jetson ARM64 环境可能需要特殊处理 Docker 网络。`host.docker.internal` 在 Jetson 上可能不可用，可以用 `--network host` 替代。

**方式二：使用 jetson-containers**

```bash
# dusty-nv 的 jetson-containers 提供了预编译的 Open-WebUI
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers

# 运行 Open-WebUI（自动适配 Jetson）
./run.sh $(./autotag open-webui)
```

### 2.6 在 Open-WebUI 中连接 llama-server

1. 打开浏览器访问 `http://<jetson-ip>:3000`
2. 进入 Settings → Connections → OpenAI API
3. 添加新连接：
   - URL: `http://<jetson-ip>:8080/v1`
   - API Key: `not-needed`（llama-server 不需要 key）
4. 刷新模型列表，应该能看到你的 GGUF 模型

---

## 关键性能数据

### 3.1 作者实测数据

| 设备 | 模型 | 推理速度 | 软件栈 |
|------|------|---------|--------|
| Orin Nano Super 8GB | 7B 模型 Q4_K_M | **9-10 tok/s** | llama.cpp CUDA + Open-WebUI |
| **AGX Orin 64GB** | **同模型 同配置** | **~21 tok/s** | **同软件栈** |

### 3.2 性能分析

AGX Orin 比 Nano Super 快约 2 倍，主要差异来源：

| 指标 | Nano Super 8GB | AGX Orin 64GB | 倍数 |
|------|---------------|---------------|------|
| 内存带宽 | ~68 GB/s | 204.8 GB/s | **3x** |
| CUDA 核心 | 1024 | 2048 | 2x |
| Tensor Core | 32 | 64 | 2x |
| 统一内存 | 8GB | 64GB | 8x |

**瓶颈分析：** 在 7B Q4_K_M 模型上，瓶颈主要在内存带宽而非计算能力。AGX Orin 3x 的带宽优势贡献了约 2x 的速度提升。Nano Super 的 8GB 限制意味着只能跑 7B Q4 以下（更大的模型会触发 OOM 或大幅降速到 CPU fallback）。

---

## 踩坑与注意事项

### 4.1 刷写相关

1. **Recovery Mode 进入方式：**
   - Orin Nano/NX：用跳线短接 FC_REC 和 GND 引脚，然后上电
   - AGX Orin：按住 Recovery 按钮再上电

2. **NVMe SSD 兼容性：** 并非所有 NVMe SSD 都能被 Jetson 识别。建议使用 PCIe Gen3/Gen4 M.2 2280 NVMe SSD（作者测试过 Samsung、WD 系列可用）

3. **刷写失败立即重试：** 如果刷写中途失败，需要重新进入 Recovery Mode 再执行 flash 命令。之前的 QSPI 分区会被覆盖，需要完整重刷。

### 4.2 llama.cpp 编译

1. **SM 8.7 的特殊性：** 预编译的 llama.cpp 通常不含 SM 8.7 的 kernel，**必须从源码编译**，且 `CMAKE_CUDA_ARCHITECTURES=87`

2. **Jetson 上编译时间较长：** Nano Super 上完整编译可能需要 20-40 分钟（AGX 约 10-15 分钟）。建议在 AGX 或 x86 交叉编译

3. **内存不足时降低 `-j`：** Nano 8GB 编译时可能 OOM，用 `-j2` 或 `-j1`

### 4.3 推理性能

1. **模型量化级别直接影响速度：** Q4_0 最快，Q5_K_M 约慢 10-20%，Q8_0 慢 30-40%。建议 Nano 用 Q4_K_M，AGX 可用 Q5_K_M 甚至 Q8_0

2. **context length 影响 VRAM：** 4096 context 在 7B Q4_K_M 上约增加 1-2GB VRAM 开销。Nano 8GB 建议不超过 8192

3. **Power Mode 影响显著：** 
   - `nvpmodel -m 0` (MAXN/MAXN_SUPER)：最高性能
   - `nvpmodel -m 1`：中等功耗（性能约降 30-40%）
   - 务必配合 `sudo jetson_clocks` 锁定频率

### 4.4 Docker 网络

Jetson ARM64 上 Docker 的网络模式与 x86 不同：
- `host.docker.internal` 可能不可用 → 用 `--network host`
- 或使用 Docker Compose 的 `extra_hosts` 手动配置

### 4.5 存储空间

- Orin Nano 8GB 的系统盘应预留 20GB 以上空间
- 7B Q4_K_M 模型约 4-5GB，加上依赖和容器约 10-15GB 总占用
- 建议使用 256GB 以上 NVMe SSD

---

## 后续扩展

作者后续文章还涉及：
- **Stable Diffusion 图生图**：[Image Generation with Stable Diffusion on Jetson Orin Using Open-WebUI](https://calje.medium.com/image-generation-with-stable-diffusion-on-jetson-orin-using-open-webui-4acdf5a71183) (2026-03-25)
- 在 Open-WebUI 中集成 SD pipeline，实现 text-to-image 和 image-to-image

---

## 参考链接

- 原文 1 (Flashing)：https://calje.medium.com/flashing-nvidia-jetson-orin-nano-and-agx-cli-nvme-guide-11d95e08a65d
- 原文 2 (LLM)：https://calje.medium.com/getting-started-with-llms-on-nvidia-jetson-orin-ee3a80096510
- llama.cpp 仓库：https://github.com/ggml-org/llama.cpp
- Open-WebUI 文档：https://docs.openwebui.com
- jetson-containers：https://github.com/dusty-nv/jetson-containers
- NVIDIA Jetson Linux 文档：https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/
