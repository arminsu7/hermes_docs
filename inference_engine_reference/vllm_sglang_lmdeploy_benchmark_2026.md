# vLLM vs SGLang vs LMDeploy：2026年LLM推理引擎性能对比

> 基于 Prem AI (Arnav Jalan) 在 n1n.ai 发布的文章整理：https://explore.n1n.ai/blog/vllm-vs-sglang-vs-lmdeploy-fastest-inference-2026-2026-03-05
> 原文发布于 2026年3月5日（dev.to 同步转载于 https://dev.to/jaipalsingh/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026-5h04）
> 说明：部分技术细节补充自官方文档

---

## 目录

1. [核心结论](#1-核心结论)
2. [Benchmark 数据](#2-benchmark-数据)
3. [vLLM 架构分析](#3-vllm-架构分析)
4. [SGLang 架构分析](#4-sglang-架构分析)
5. [LMDeploy 架构分析](#5-lmdeploy-架构分析)
6. [三引擎对比总结](#6-三引擎对比总结)
7. [选型建议](#7-选型建议)

---

## 1. 核心结论

**SGLang 和 LMDeploy 并列 2026 年最快推理引擎**，在 H100 上 Llama 3.1 8B benchmark 中：

| 引擎 | 吞吐量 (tok/s) | 相对 vLLM |
|------|---------------|-----------|
| **SGLang** | ~16,200 | +29% |
| **LMDeploy** | ~16,100 | +29% |
| vLLM | ~12,500 | 基准 |

**差距来源不是"谁的 kernel 写得更好"，而是架构设计的根本差异。** vLLM 在 2026 年仍在追赶 SGLang 最早 2024 年引入的 RadixAttention 前缀缓存机制。

---

## 2. Benchmark 数据

### 2.1 测试配置

| 参数 | 值 |
|------|-----|
| 模型 | Llama 3.1 8B |
| GPU | H100 80GB |
| 输入长度 | 1024 tokens |
| 输出长度 | 512 tokens |
| 量化 | FP16（无量化） |

### 2.2 原始数据（多 batch size 对比）

| Batch Size | vLLM (req/s) | SGLang (req/s) | LMDeploy (req/s) |
|-----------|-------------|----------------|-----------------|
| 1 | 8.5 | 8.5 | 8.5 |
| 8 | 55 | 62 | 62 |
| 16 | 95 | 115 | 114 |
| 32 | 155 | 195 | 194 |
| 64 | 220 | 290 | 288 |
| 128 | 270 | 370 | 365 |

> 低 batch size 下三引擎性能接近。差异主要在 batch size ≥16 时显现，且随 batch 增大而拉大。

### 2.3 峰值吞吐量

```
tokens/second (Llama 3.1 8B, H100, 1024 in / 512 out)

SGLang     ████████████████████ 16,200
LMDeploy   ███████████████████▌ 16,100
vLLM       ████████████████▌    12,500

SGLang/LMDeploy 比 vLLM 快 ~29%
```

---

## 3. vLLM 架构分析

### 3.1 核心设计

```
请求 → Scheduler (抢占式) → Model Runner → GPU
         ↑
    PagedAttention
    KV Cache 管理
```

**vLLM 的调度器的特点是"抢占式"（preemptive）：** 当一个高优先级请求到达时，vLLM 会暂停当前正在处理的 batch 中的低优先级请求，将它们的 KV Cache 换出到 CPU 内存，插入高优先级请求，完成后再恢复。

### 3.2 优势

- **PagedAttention**：KV Cache 虚拟内存管理，碎片化程度最低，显存利用率最高
- **抢占式调度**：适合混合优先级场景（如在线服务中紧急请求优先处理）
- **生态最成熟**：模型支持最广，社区最大，文档最完善
- **Model Runner V2**（v0.25+）：统一内核，减少框架开销

### 3.3 劣势

- **无原生前缀缓存**：vLLM 在 2025 年底才加入 automatic prefix caching（APC），比 SGLang 的 RadixAttention 晚了约一年。对于共享前缀的 workload（如 multi-turn 对话、RAG、system prompt 复用），vLLM 的 APC 不如 SGLang 的 RadixAttention 高效
- **调度器激进性不足**：在纯吞吐量场景下，vLLM 的抢占式调度引入了额外开销
- **PyTorch 层封装较厚**：相比 LMDeploy TurboMind 的 C++ 直接操作，vLLM 的 Python 层有一定开销

---

## 4. SGLang 架构分析

### 4.1 核心设计

```
请求 → RadixAttention (前缀树) → Scheduler (连续批处理) → GPU
         ↓
    自动前缀匹配 & KV Cache 复用
```

**RadixAttention 是 SGLang 的核心竞争力。** 它是一个基于 Radix Tree（基数树/前缀树）的 KV Cache 管理器：自动检测多个请求之间的共享前缀，复用已有的 KV Cache，避免重复计算。

### 4.2 RadixAttention 工作原理

```
请求1: "You are a helpful assistant. What is Python?"
请求2: "You are a helpful assistant. What is Rust?"
请求3: "You are a helpful assistant. Explain quantum computing."

共享前缀 "You are a helpful assistant. " → 只计算一次
剩余部分各自计算
```

对于 system prompt 固定 + user prompt 变化的典型场景，RadixAttention 可节省 30-50% 的 prefill 计算。

### 4.3 优势

- **RadixAttention**：实际生产中最有效的前缀缓存方案（vLLM 2025年底才追赶）
- **零开销批处理调度**：无抢占开销，纯连续批处理
- **Day-0 模型支持**：DeepSeek V3/R1、MiniMax M2、Mistral Large 3 等最新模型
- **已加入 PyTorch 生态**（2025年3月）
- **SGLang Diffusion**：扩展到视频/图像生成领域

### 4.4 劣势

- **文档和社区不如 vLLM**：虽然增长迅速（40万+ GPU 部署），但中文资料较少
- **部分高级功能不如 vLLM 完善**：如 LoRA hot-swap、speculative decoding 等
- **对旧架构 GPU 优化不如 vLLM**：主要优化目标为 H100/A100 等数据中心 GPU

---

## 5. LMDeploy 架构分析

### 5.1 核心设计

```
请求 → TurboMind Engine (C++) → 连续批处理 + AWQ Kernel → GPU
         ↓
    无 Python 运行时开销
```

**TurboMind 是 LMDeploy 的 C++ 推理引擎，完全绕过 Python 解释器。** 相比之下，vLLM 和 SGLang 的 runtime 都有 Python 层。

### 5.2 TurboMind 的技术栈

| 层级 | vLLM/SGLang | LMDeploy TurboMind |
|------|------------|-------------------|
| API 层 | Python (FastAPI) | Python (FastAPI) |
| 调度层 | Python | **C++** |
| 推理层 | CUDA Kernel + PyTorch | **C++ 直接调用 CUDA** |
| 量化 Kernel | Marlin / CUTLASS | **自研 AWQ TurboMind Kernel** |

关键差异：LMDeploy 把性能敏感的计算全部放在 C++ 层，仅用 Python 做 API 暴露。这在大 batch 高并发场景下减少了 Python GIL 和对象序列化开销。

### 5.3 优势

- **C++ 原生引擎**：无 Python 运行时开销，在 batch size ≥32 时优势明显
- **AWQ/MXFP4 量化原生支持**：TurboMind 为 AWQ 格式定制了 C++ 加载器，`--model-format awq` 可直接加载
- **KV Cache 量化**：支持 INT4/INT8 KV Cache 量化，进一步节省显存
- **1.5x-1.8x vLLM 吞吐量**（官方数据）：在 MXFP4 + gpt-oss 场景下可达 5x
- **pipeline API**：提供 Python pipeline 接口，适合自定义推理流程

### 5.4 劣势

- **模型支持不如 vLLM 广泛**：主要为 InternLM、Qwen、Llama、DeepSeek 等主流架构优化
- **社区规模最小**：相比 vLLM 和 SGLang，GitHub stars 和贡献者数量更少
- **Blackwell 适配曾滞后**（v0.13.0 起已支持）
- **Jetson/ARM 平台无官方支持**

---

## 6. 三引擎对比总结

### 6.1 架构对比

| 维度 | vLLM | SGLang | LMDeploy |
|------|------|--------|---------|
| **核心语言** | Python + C++/CUDA | Python + C++/CUDA | Python API + **C++ 引擎** |
| **KV Cache** | PagedAttention | **RadixAttention** (前缀树) | Paged KV Cache |
| **调度器** | 抢占式 | 连续批处理 | 连续批处理 |
| **前缀缓存** | APC (2025年底) | RadixAttention (2024) | - |
| **量化** | AWQ/GPTQ/FP8 | AWQ/GPTQ/FP8 | **AWQ/MXFP4/KV Cache 量化** |
| **多卡** | TP/PP/DP/EP/CP | TP/DP/EP | TP |
| **模型支持** | **最广** | 广 | 主流 |

### 6.2 性能对比

| 场景 | 最佳选择 | 原因 |
|------|---------|------|
| **共享前缀 workload** (RAG, multi-turn) | **SGLang** | RadixAttention 节省 30-50% prefill |
| **高并发吞吐量** (batch ≥32) | **LMDeploy 或 SGLang** | C++ 引擎 / 连续批处理 |
| **混合优先级** (在线服务) | **vLLM** | 抢占式调度 |
| **极致量化性能** | **LMDeploy** | AWQ TurboMind kernel + MXFP4 |
| **模型覆盖 / 快速验证** | **vLLM** | 最广模型 + 最大社区 |
| **低并发单用户** | 三者接近 | batch=1 差异 <5% |

### 6.3 吞吐量 vs 批处理大小的关系

```
tok/s
400 ↑                                    ● SGLang
    |                              ●───── LMDeploy
300 |                         ●───
    |                    ●───
200 |               ●───
    |          ●───              ● vLLM
100 |     ●───              ●───
    |●───              ●───
  0 ├────┬────┬────┬────┬────┬───→ batch size
    0    32   64   96   128  160

低 batch：三者接近
高 batch：SGLang ≈ LMDeploy >> vLLM
差距随 batch 增大而拉大
```

---

## 7. 选型建议

### 7.1 选型决策树

```
你的场景是？
│
├── 需要最广模型支持 + 最成熟生态？
│   └── vLLM（模型覆盖最广，社区最大，文档最好）
│
├── 大量共享前缀（RAG / multi-turn / system prompt 固定）？
│   └── SGLang（RadixAttention 节省 30-50% prefill 计算）
│
├── 高并发 + 追求极限吞吐量 + 接受稍窄模型覆盖？
│   └── LMDeploy TurboMind（C++ 引擎，AWQ 原生支持）
│      或 SGLang（吞吐量几乎持平，但社区更大）
│
├── 需要抢占式调度/混合优先级在线服务？
│   └── vLLM（唯一支持抢占式调度的引擎）
│
└── 低并发 / 开发测试 / 快速验证？
    └── 三者都可以，vLLM 安装最简单（pip install vllm）
```

### 7.2 推荐组合

| 硬件 | 场景 | 推荐引擎 | 理由 |
|------|------|---------|------|
| RTX 4090 单卡 | 开发测试 | vLLM | pip install 即用，模型覆盖最广 |
| RTX 4090 单卡 | 追求吞吐量 | LMDeploy | C++ TurboMind + AWQ 量化 |
| H100 多卡 | 在线服务(RAG) | SGLang | RadixAttention 前缀缓存 |
| H100 多卡 | 高并发纯推理 | LMDeploy 或 SGLang | 峰值吞吐量最高 |
| RTX 3060 12GB | 单用户 | vLLM 或 Ollama | 低并发差异不大 |

---

## 参考链接

- 原文：https://explore.n1n.ai/blog/vllm-vs-sglang-vs-lmdeploy-fastest-inference-2026-2026-03-05
- dev.to 转载：https://dev.to/jaipalsingh/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026-5h04
- Prem AI 博客：https://blog.premai.io/author/arnav/
- vLLM GitHub：https://github.com/vllm-project/vllm
- SGLang GitHub：https://github.com/sgl-project/sglang
- LMDeploy GitHub：https://github.com/InternLM/lmdeploy
