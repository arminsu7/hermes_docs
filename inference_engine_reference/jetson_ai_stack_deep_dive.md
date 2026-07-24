# Jetson AI Stack 深度解析：JetPack 组件版本与硬件加速模块

> 基于 eLinux Jetson AI Stack 页面整理：https://elinux.org/Jetson/L4T/Jetson_AI_Stack#AGX_Orin
> 补充：NVIDIA JetPack 官方文档、Jetson Linux Developer Guide、Jetson AI Lab、ProventusNova
> 说明：eLinux 页面为社区维护的 JetPack 版本对照表，覆盖 JetPack 4.x ~ 7.x 的 AI 栈组件版本及 Jetson 硬件加速模块说明

---

## 目录

1. [JetPack 软件栈架构](#1-jetpack-软件栈架构)
2. [JetPack 版本对照表](#2-jetpack-版本对照表)
3. [AI Stack 核心组件详解](#3-ai-stack-核心组件详解)
4. [Jetson 硬件加速模块](#4-jetson-硬件加速模块)
5. [AGX Orin 专项](#5-agx-orin-专项)

---

## 1. JetPack 软件栈架构

### 1.1 三层结构

```
┌─────────────────────────────────────────────────┐
│               Developer Tools                    │
│  (SDK Manager, Nsight, VPI, Multimedia API...)   │
├─────────────────────────────────────────────────┤
│                  AI Stack                        │
│  CUDA │ cuDNN │ TensorRT │ DLA │ VPI │ DLFW     │
├─────────────────────────────────────────────────┤
│               Jetson Linux (L4T)                 │
│  Linux Kernel │ NVIDIA Drivers │ UEFI │ OP-TEE  │
│  Ubuntu RootFS │ Bootloader                      │
├─────────────────────────────────────────────────┤
│             Jetson Hardware                      │
│  GPU │ DLA │ PVA │ VIC │ NVENC │ NVDEC │ ISP    │
└─────────────────────────────────────────────────┘
```

### 1.2 各层职责

| 层级 | 内容 | 更新方式 |
|------|------|---------|
| **Jetson Linux (L4T)** | Linux Kernel 5.10/5.15/6.x, NVIDIA 驱动, UEFI, OP-TEE, Ubuntu RootFS | 需要刷机升级 |
| **AI Stack** | CUDA, cuDNN, TensorRT, DLA, VPI, DLFW | JetPack 包升级 |
| **Developer Tools** | SDK Manager, Nsight, VPI, Multimedia API, Isaac ROS | 独立安装 |

> **关键点：** JetPack 6 起 AI Stack 与 L4T 有更好的解耦，理论上可以在不刷机的情况下升级部分 AI Stack 组件。

---

## 2. JetPack 版本对照表

### 2.1 Orin 系列 (JetPack 5.x ~ 7.x)

| JetPack | L4T | CUDA | cuDNN | TensorRT | VPI | DLA | 支持设备 |
|---------|-----|------|-------|---------|-----|-----|---------|
| **7.1** | R38.2 | 13.0 | 9.7 | 10.13 | 3.3 | 3.2 | Thor + Orin |
| **7.0** | R38.1 | 13.0 | 9.6 | 10.11 | 3.3 | 3.2 | Thor + Orin |
| **6.2** | R36.4 | 12.6 | 9.3 | **10.3** | 3.2 | 3.1 | Orin (Super Mode) |
| **6.1** | R36.3 | 12.6 | 9.3 | 10.0 | 3.2 | 3.1 | Orin 全系列 |
| **6.0** | R36.2 | 12.2 | 8.9 | 8.6 | 3.1 | 3.0 | Orin 全系列 |
| **5.1.3** | R35.5 | 11.4 | 8.6 | 8.5 | 2.3 | 2.2 | Orin 全系列 |
| **5.1.2** | R35.4 | 11.4 | 8.6 | 8.5 | 2.3 | 2.2 | Orin 全系列 |

### 2.2 Xavier 系列 (JetPack 4.x)

| JetPack | L4T | CUDA | cuDNN | TensorRT | 支持设备 |
|---------|-----|------|-------|---------|---------|
| **4.6** | R32.7 | 10.2 | 8.2 | 8.2 | AGX Xavier / Xavier NX / TX2 |
| **4.5** | R32.5 | 10.2 | 8.0 | 7.1 | 同上 |

### 2.3 TensorRT 版本对 LLM 推理的影响

| TensorRT 版本 | JetPack | LLM 支持的 Engine 格式 | 备注 |
|-------------|---------|----------------------|------|
| 8.5 | 5.1.x | TRT-LLM v0.12.0-jetson | **仅 AGX Orin，Int4/FP16** |
| 8.6 | 6.0 | TRT-LLM v0.12.0-jetson | Int4/FP16 |
| 10.0 | 6.1 | TRT-LLM v0.12.0-jetson + Edge-LLM | Edge-LLM 开始支持 |
| 10.3 | 6.2 | Edge-LLM (推荐) | **AWQ/INT8/NVFP4 支持** |
| 10.11+ | 7.0+ | Edge-LLM (Thor 主力) | NVFP4 原生支持 |

> **对推理引擎选型的关键影响：** JetPack 6.2 (TensorRT 10.3) 是 Orin 上运行 TRT Edge-LLM 的最低推荐版本，因为 AWQ INT4 量化需要 TensorRT 10.x 的 INT4 GEMM kernel 支持。

---

## 3. AI Stack 核心组件详解

### 3.1 CUDA Toolkit

**作用：** GPU 通用计算平台，提供 CUDA C/C++ 编译器 (nvcc) 和 CUDA runtime 库

**Jetson 上的特殊点：**
- CUDA 版本绑定在 JetPack 中，不能像桌面那样独立升级
- SM 架构：Orin = SM 8.7 (Ampere 变体)，Thor = SM 10.0 (Blackwell 变体)
- 共享统一内存架构：`cudaMalloc` 和 `malloc` 分配同一块物理内存

**版本对应关系：**
```
JetPack 5.1.x → CUDA 11.4 → SM 8.7 support (Orin)
JetPack 6.0   → CUDA 12.2 → SM 8.7 + Blackwell preview
JetPack 6.1   → CUDA 12.6 → SM 8.7 + SM 10.0 (Thor)
JetPack 7.x   → CUDA 13.0 → SM 8.7 + SM 10.0 + SM 12.0 (Thor T5000)
```

### 3.2 cuDNN

**作用：** 深度神经网络 GPU 加速库，提供卷积、池化、归一化等优化的 CUDA kernel

**Jetson 上 LLM 推理的相关性：**
- 主要用于 CNN/Vision Transformer 的推理优化
- LLM 推理中，Vision Encoder（VLM 的视觉部分）依赖 cuDNN
- 纯文本 LLM 推理直接使用自研 kernel，不经过 cuDNN

### 3.3 TensorRT

**作用：** NVIDIA 深度学习推理优化器，将训练好的模型编译为 GPU 优化的推理 engine

**Jetson 上的关键限制：**
- **Engine 绑定特定 JetPack/CUDA/TensorRT 版本**，升级 JetPack 后必须重新编译 engine
- Orin 的 TensorRT 10.3+ 才完整支持 INT4 AWQ 量化
- DLA 作为独立推理加速器，不经过 TensorRT（有自己的编译器）

**TensorRT Engine 编译流程：**
```
ONNX 模型 → trtexec/edge-llm-build → .plan (TensorRT Engine)
                                         ↑
                                    绑定 GPU 型号 + JetPack 版本 + 精度
```

### 3.4 VPI (Vision Programming Interface)

**作用：** 计算机视觉/图像处理算法库，提供跨 GPU/CPU/PVA/VIC 的统一 API

**支持的硬件后端：**

| 后端 | 加速硬件 | 场景 |
|------|---------|------|
| **CUDA** | GPU Tensor Core | 深度学习预处理 |
| **CPU** | ARM Cortex-A78AE | 灵活度最高，兼容性最好 |
| **PVA** | Programmable Vision Accelerator | 低功耗视觉 pipeline |
| **VIC** | Video Image Compositor | 硬件图像格式转换 |

**Orin 上的 VPI 版本演进：**
```
JetPack 5.1.x → VPI 2.3 → 基础视觉算法
JetPack 6.0   → VPI 3.1 → 新增算法 + 改进的 PVA 调度
JetPack 6.1   → VPI 3.2 → DLA 集成 + 更多预处理
JetPack 7.x   → VPI 3.3 → Thor 适配 + FP16 预处理
```

### 3.5 DLA (Deep Learning Accelerator)

**作用：** 专用深度学习推理加速器，独立于 GPU，超低功耗，确定性延迟

**与 GPU 的对比：**

| 维度 | GPU (Tensor Core) | DLA |
|------|------------------|-----|
| **TOPS** | 170-275 (Orin) | ~21 (Orin AGX, 2x DLA) |
| **功耗** | 15-60W | ~2-5W |
| **精度** | FP32/FP16/INT8/INT4 | INT8/FP16 |
| **延迟** | 可变（调度开销） | **确定性** |
| **适合** | LLM 推理、大规模 CV | 轻量 CV（检测/分割）、always-on |
| **LLM 推理** | ✅ 主力 | ❌ 不支持 Transformer |

> **DLA 不能用于 LLM 推理。** DLA 专为 CNN 类网络设计（卷积、池化、激活），不支持 Transformer 的 Attention/GEMM 操作。

---

## 4. Jetson 硬件加速模块

### 4.1 硬件模块总览

```
Jetson AGX Orin SoC
┌──────────────────────────────────────────────────────┐
│                                                        │
│  ┌─────────┐  ┌──────┐  ┌──────┐  ┌──────────────┐   │
│  │   GPU   │  │ DLA  │  │ PVA  │  │  VIC (Video  │   │
│  │ 2048 CC │  │  x2  │  │  x1  │  │   Image      │   │
│  │ 64  TC  │  │21 TOPs│  │      │  │  Compositor) │   │
│  └─────────┘  └──────┘  └──────┘  └──────────────┘   │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │  NVENC   │  │  NVDEC   │  │   ISP (Image       │   │
│  │ (H.265)  │  │ (H.265)  │  │   Signal Processor)│   │
│  │ encoder  │  │ decoder  │  │   16-camera MIPI   │   │
│  └──────────┘  └──────────┘  └────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │    ARM Cortex-A78AE CPU (12 cores)               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  Unified Memory: 32/64 GB LPDDR5                      │
│  Memory Bandwidth: 204.8 GB/s                          │
└──────────────────────────────────────────────────────┘
```

### 4.2 各模块详解

#### GPU (Graphics Processing Unit)

| Orin 型号 | CUDA Cores | Tensor Cores | GPU Clock (MAXN) | GPU Clock (MAXN_SUPER) | GPU INT8 TOPS (Sparse/Dense) |
|-----------|-----------|-------------|-----------------|----------------------|------------------------------|
| **AGX Orin 64GB** | 2048 | 64 | 1.3 GHz | - | **170 / 85** |
| **AGX Orin 32GB** | 1792 | 56 | 0.93 GHz | - | **108 / 54** |
| **Orin NX 16GB** | 1024 | 32 | 918 MHz | 1.17 GHz | 100 / 50 |
| **Orin NX 8GB** | 1024 | 32 | 765 MHz | 1.17 GHz | 70 / 35 |
| **Orin Nano 8GB** | 1024 | 32 | 625 MHz | 1.02 GHz | 40 / 20 |
| **Orin Nano 4GB** | 512 | 16 | 625 MHz | 1.02 GHz | 20 / 10 |

> **说明：**
> - Sparse = 利用 2:4 结构化稀疏的理论峰值；Dense = 通用密集矩阵的实际可用算力
> - NVIDIA 营销材料中的"275 TOPS"是 AGX Orin 64GB 的**总平台算力**（GPU 170 + 2×DLA ~105），非 GPU 单独算力
> - MAXN_SUPER 需要 JetPack 6.2+ 且执行 `sudo nvpmodel -m <mode>` + `sudo jetson_clocks`（NX: mode 2, Nano: mode 2）
> - AGX Orin 无独立 SUPER 模式，MAXN 即最高（`nvpmodel -m 0`）

**Tensor Core 精度支持（架构差异）：**

Orin 全系列使用 Ampere 架构（3rd Gen Tensor Core），各代 Tensor Core 的精度支持如下：

| 架构 | 代表 GPU | 原生 Tensor Core 支持 | INT4 通用 GEMM |
|------|---------|---------------------|---------------|
| Ampere (3rd Gen) | **AGX Orin** | FP16, BF16, TF32, INT8, INT4(稀疏) | ❌ 仅 2:4 稀疏 |
| Ada Lovelace (4th Gen) | RTX 4090 | +FP8 | ❌ 仅 2:4 稀疏 |
| Hopper (4th Gen) | H100 | +FP8 | ❌ 仅 2:4 稀疏 |
| Blackwell (5th Gen) | Thor, RTX 5090 | +FP4 (NVFP4) | ✅ **原生 NVFP4** |

> **关键结论：AGX Orin 没有原生 INT4 Tensor Core 支持。**
>
> Ampere 的 "INT4" 仅指 2:4 结构化稀疏模式（每 4 个权重去掉 2 个），不是通用 INT4 矩阵乘法。
> Orin 上运行 INT4 AWQ 量化模型时：**权重以 INT4 存储（节省 75% 带宽），但计算前 dequantize 到 INT8/FP16，再用 INT8/FP16 Tensor Core 执行。**
> 带宽节省是 Orin 上 INT4 量化最大的收益（204.8 GB/s 的带宽本就是瓶颈），计算层面没有硬件加速。

#### DLA (Deep Learning Accelerator)

| Orin 型号 | DLA 数量 | DLA Sparse TOPS (总计) | 精度 |
|-----------|---------|----------------------|------|
| AGX Orin | **2** | **105** | INT8/FP16 |
| Orin NX | 2 | 未公开（总平台 TOPS 含 GPU+DLA） | INT8/FP16 |
| Orin Nano | 0 | 0 | **无 DLA** |

> AGX Orin 的 2 个 DLA 合计提供 105 INT8 Sparse TOPS（NVIDIA 技术简报 v1.2 数据）。DLA 仅支持 CNN 类网络（卷积、池化、激活），**不支持 Transformer/Attention/GEMM 操作，不能用于 LLM 推理。** Orin NX 的 DLA 规格未在公开 datasheet 中独立列出。Orin Nano **没有 DLA**，所有推理必须走 GPU。

#### PVA (Programmable Vision Accelerator)

- AGX Orin：1 个 PVA
- Orin NX：1 个 PVA
- Orin Nano：**无 PVA**

用于 VPI 视觉 pipeline 的硬件加速（图 resize、CSC 转换、透视变换等）。

#### VIC (Video Image Compositor)

- 硬件图像处理引擎
- 支持格式转换（NV12 ↔ RGB ↔ YUV）、缩放、裁剪
- 所有 Orin 型号都有 VIC

#### NVENC / NVDEC (Video Encoder / Decoder)

| 功能 | Orin 支持 |
|------|---------|
| H.265 (HEVC) Encode | ✅ 4K60 |
| H.265 (HEVC) Decode | ✅ 4K60 |
| H.264 (AVC) Encode | ✅ 4K60 |
| H.264 (AVC) Decode | ✅ 4K60 |
| AV1 Decode | ✅ (Orin) |

#### ISP (Image Signal Processor)

- 支持最多 **16 路 MIPI CSI 摄像头**
- 硬件去马赛克 (demosaicing)、降噪、白平衡
- AGX Orin 独有（Nano/NX 使用简化版或软件 ISP）

---

## 5. AGX Orin 专项

### 5.1 硬件规格（eLinux 页面数据）

| 参数 | AGX Orin 64GB | AGX Orin 32GB |
|------|-------------|-------------|
| **GPU** | 2048 CUDA Cores, 64 Tensor Cores | 1792 CUDA Cores, 56 Tensor Cores |
| **GPU Max Clock** | 1.3 GHz | 0.93 GHz |
| **DL Accelerator** | 2x NVDLA 2.0 | 2x NVDLA 2.0 |
| **Vision Accelerator** | 1x PVA 2.0 | 1x PVA 2.0 |
| **CPU** | 12x ARM Cortex-A78AE | 8x ARM Cortex-A78AE |
| **Memory** | 64 GB LPDDR5 | 32 GB LPDDR5 |
| **Memory Bandwidth** | 204.8 GB/s | 204.8 GB/s |
| **Storage** | 64 GB eMMC + NVMe | 64 GB eMMC + NVMe |
| **TOPS (INT8)** | 275 | 200 |
| **Power** | 15-60W | 15-40W |

### 5.2 Power Mode 设定

```bash
# 查看所有可用的 Power Mode
sudo nvpmodel -q

# AGX Orin Power Mode 示例
# 0: MAXN (最高性能, 60W)
# 1: 40W 平衡
# 2: 30W 低功耗
# 3: 15W 超低功耗

# 锁定 GPU 最高频率
sudo jetson_clocks
```

### 5.3 统一内存架构的优势

```
传统 GPU (RTX 4090)：
  CPU RAM (24+ GB) ← PCIe ~32 GB/s → GPU VRAM (24 GB)
  模型必须先加载到 VRAM，KV Cache 也在 VRAM

Jetson AGX Orin：
  Unified Memory (64 GB LPDDR5, 204.8 GB/s)
  CPU 和 GPU 共享同一物理内存，零拷贝
```

**对 LLM 推理的影响：**
- 可以加载超过 GPU VRAM 的模型（用 CPU 部分做部分 offload）
- KV Cache 的 CPU-GPU 换入换出延迟为零
- 但总带宽远低于独立显卡（204.8 vs 1008 GB/s）

---

## 参考链接

- eLinux Jetson AI Stack：https://elinux.org/Jetson/L4T/Jetson_AI_Stack#AGX_Orin
- NVIDIA JetPack SDK：https://developer.nvidia.com/embedded/jetpack
- Jetson Linux Developer Guide：https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/
- Jetson AI Lab 入门教程：https://www.jetson-ai-lab.com/tutorials/intro-to-jetson/
- JetPack 版本与 L4T 对照表：https://proventusnova.com/blog/jetpack-versions-l4t-compatibility-table/
- Jetson Software Architecture：https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/AR/JetsonSoftwareArchitecture.html
