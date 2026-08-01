# ffs_deploy.sh — FFS ONNX → TRT Engine 一键部署

快速入门：在当前目录运行 `./ffs_deploy.sh`。
详细用法见下文。

## 概述

`ffs_deploy.sh` 是一键 ONNX→TensorRT engine 部署脚本，自动检测 GPU/CUDA/TRT 环境，编译 V3 Custom Plugin 并构建 engine（plugin 已序列化，加载时无需 .so）。

支持平台：RTX 30/40/50 系列、AGX Orin、Orin NX 等。

## 用法

```bash
./ffs_deploy.sh [WORKDIR]
```

WORKDIR 默认当前目录 (`$PWD`)。

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TRT_ROOT` | 自动检测 | TensorRT 安装路径 |
| `CUDA_ARCH` | 自动检测 | SM 架构号 (86/87/89/90...) |
| `ENGINE_NAME` | `<onnx>_fp16.engine` | 输出 engine 文件名 |
| `NO_BUILD=1` | 否 | 跳过 plugin 编译 |
| `FORCE_REBUILD=1` | 否 | 强制重新转换（覆盖已有 engine） |

### 示例

```bash
# 当前目录，自动检测一切
./ffs_deploy.sh

# 指定工作目录
./ffs_deploy.sh /root/workspace/ffs_trt_deploy

# RTX 3060 + hub tarball
./ffs_deploy.sh /root/workspace/ffs_trt_deploy

# Orin NX（JetPack TRT）
CUDA_ARCH=87 ./ffs_deploy.sh /home/nvidia/ffs_deploy

# 显式指定 TRT 路径
TRT_ROOT=/root/repos/hub/TensorRT-10.3.0.26 ./ffs_deploy.sh .

# 跳过编译（plugin 已编译）
NO_BUILD=1 ./ffs_deploy.sh .
```

## 目录结构

```
ffs_trt_deploy/              # WORKDIR (任意路径)
├── cpp-v3/                  # V3 Plugin 源码 (必选)
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── ffs_gwc_plugin.hpp
│   └── src/
│       ├── gwc_volume_plugin_v3.cpp
│       └── depth_kernels.cu
├── *.onnx                   # ONNX 模型 (必选，自动发现)
└── *.yaml                   # (可选)
```

## 流程

```
1. [检测] GPU → 名称 / Compute Cap / SM Arch
2. [检测] CUDA → nvcc / 版本
3. [检测] TRT → trtexec / libnvinfer / 头文件 / 版本
   (以上信息全部打印到终端)
4. [编译] cmake + make ffs_gwc_plugin
5. [转换] trtexec --dynamicPlugins --setPluginsToSerialize --fp16
6. [打印] Summary: GPU / SM / CUDA / TRT / Plugin / Engine
```

## 输出文件

```
ONNX_DIR/
├── xxx_fp16.engine          # engine 文件 (~38 MB)
├── xxx_fp16.log             # trtexec 日志
├── xxx_fp16_time.cache      # 时序缓存
├── xxx_fp16_profile.json    # layer 耗时
└── xxx_fp16_layerinfo.json  # layer 信息
```

## 加载 engine

```bash
trtexec --loadEngine=xxx_fp16.engine --useSpinWait --iterations=100
# 无需 --staticPlugins / --dynamicPlugins —— plugin 已序列化
```

## 验证记录

| 平台 | GPU | SM | TRT | 结果 |
|---|---|---|---|---|
| WSL2 Docker | RTX 3060 | 86 | 10.3.0.26 | PASS (37 MB) |

> **注**：trtexec 构建后会自行验证反序列化，此时 "exists already" 错误是预期行为（`--dynamicPlugins` 已注册 + engine 内嵌 library 重复注册），非致命，engine 文件已正确生成。
