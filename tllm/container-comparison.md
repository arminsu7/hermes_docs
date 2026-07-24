# TensorRT-LLM 容器环境对比

> 检查日期：2026-07-21
> 宿主机：WSL2, RTX 3060 12GB, Driver 596.49

## 容器概览

| 容器名 | 镜像 | 用途 |
|------|------|------|
| tllm130rc20-rel | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` | 运行时部署 |
| tllm130rc21-rel | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc21` | 运行时部署 |
| tllm130rc20-dev | `nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc20` | 源码编译开发 |
| tllm130rc21-dev | `nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc21` | 源码编译开发 |

## 核心组件版本

| 组件 | rc20-rel | rc21-rel | rc20-dev | rc21-dev |
|------|------|------|------|------|
| tensorrt_llm | 1.3.0rc20 | 1.3.0rc21 | 未安装 | 未安装 |
| tensorrt | 10.15.1.29 | 10.16.1.11 | 10.15.1.29 | 10.16.1.11 |
| torch | 2.11.0a0 | 2.12.0a0 | 2.11.0a0 | 2.12.0a0 |
| nvidia-modelopt | 0.37.0 | 0.37.0 | 0.41.0 | 0.42.0 |
| transformers | 5.5.4 | 5.5.4 | - | - |
| CUDA (nvcc) | 13.1 | 13.2 | 13.1 | 13.2 |
| flash_attn | 2.7.4 | 2.7.4 | 2.7.4 | 2.7.4 |
| flash-attn-4 | 4.0.0b11 | 4.0.0b11 | 4.0.0b11 | 4.0.0b11 |
| cudnn-frontend | 1.18.0 | 1.22.1 | 1.18.0 | 1.22.1 |
| onnx | 1.22.0 | 1.21.0 | 1.18.0 | 1.21.0 |
| mpi4py | 3.1.5 | 3.1.5 | 3.1.5 | 3.1.5 |

## 功能支持矩阵

| 能力 | rc20-rel | rc21-rel | rc20-dev | rc21-dev |
|------|:------:|:------:|:------:|:------:|
| trtllm-build（TRT backend） | ✅ | ❌ | ❌ | ❌ |
| trtllm-serve（PyTorch backend） | ✅ | ✅ | ❌ | ❌ |
| tensorrt_llm.builder 模块 | ✅ | ❌ | ❌ | ❌ |
| Qwen3VLModel | ✅ | ✅ | ❌ | ❌ |
| trtllm-bench | ✅ | ✅ | ❌ | ❌ |
| trtllm-eval | ✅ | ✅ | ❌ | ❌ |
| cmake / make / nvcc | ✅ | ✅ | ✅ | ✅ |

## 关键变化：rc20 → rc21

### 移除的内容

- `trtllm-build` 命令
- `tensorrt_llm.builder` 模块
- `tensorrt_llm.commands.build` 模块
- 整个 TensorRT engine build 流程

### 保留的内容

- `trtllm-serve`（PyTorch backend 推理服务）
- `trtllm-bench`（性能测试）
- `trtllm-eval`（模型评估）
- Qwen3VLModel 模型定义

### 升级的组件

| 组件 | rc20 | rc21 |
|------|------|------|
| TensorRT | 10.15.1 | 10.16.1 |
| PyTorch | 2.11.0 | 2.12.0 |
| CUDA | 13.1 | 13.2 |
| cudnn-frontend | 1.18.0 | 1.22.1 |
| nvidia-modelopt (dev) | 0.41.0 | 0.42.0 |

## 结论

### 部署推荐

| 场景 | 推荐容器 | 原因 |
|------|------|------|
| 快速部署，PyTorch backend | **tllm130rc21-rel** | 直接可用，最新版本 |
| 极致性能，TRT engine | **tllm130rc20-rel** | 唯一支持 TRT backend 的版本 |
| 从源码开发 | 任意 dev 容器 | 需自行编译安装 tensorrt_llm |

### 重要说明

GitHub releases 声明：v1.3.0rc20 是最后一个支持 TensorRT backend 的版本，v1.3.0rc21 起 TRT backend 已移除，PyTorch backend 成为唯一执行后端。

- 官方声明：https://github.com/NVIDIA/TensorRT-LLM/releases
- 迁移指南：https://nvidia.github.io/TensorRT-LLM/legacy/tensorrt-backend-removal.html