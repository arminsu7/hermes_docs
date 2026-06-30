# Qwen3.5-4B 本地部署量化选择学习笔记

## 1. 这份文档解决什么问题

这份笔记把两件事放在一起：

1. `compressed-tensors` 里 `preset_schemes` 的含义
2. 针对你手上的几台机器，部署 `Qwen/Qwen3.5-4B` 时该优先选什么量化方案

目标不是把所有格式都背下来，而是形成一个可执行的选型脑图：

- 机器显存/内存够不够
- 是优先质量、吞吐还是兼容性
- 该选 `W4A16 / W8A8 / FP8_DYNAMIC`，还是直接走 `GGUF Q4_K_M`

---

## 2. 本文使用到的已核实信息

### 2.1 Qwen3.5-4B 模型信息

根据 Hugging Face 搜索结果：

- 模型：`Qwen/Qwen3.5-4B`
- 默认上下文长度：`262,144 tokens`
- 兼容生态：Transformers、vLLM、SGLang 等
- 另有现成 GGUF：`unsloth/Qwen3.5-4B-GGUF`

### 2.2 4B 模型不同精度下的原始权重体积（仅权重，不含 KV cache / runtime buffer）

实际计算结果：

- BF16/FP16：`7.45 GiB`
- FP8/INT8：`3.73 GiB`
- INT4/FP4：`1.86 GiB`

这只是“裸权重”。真正部署时，还要给：

- KV cache
- 激活
- runtime workspace
- allocator fragment
- 框架额外开销

预留空间。

### 2.3 KV cache 的粗略量级提醒

按一个 4B 级 dense decoder 模型的粗略估算，KV cache 大致是：

- BF16/FP16 KV：约 `360 MiB / 1K tokens`
- FP8/INT8 KV：约 `180 MiB / 1K tokens`

所以要特别注意：

- 原生 `262K` context 几乎不可能在这些本地设备上直接开满
- 真正实用的本地部署上下文，通常是 `2K / 4K / 8K / 16K / 32K`
- 量化解决的是“权重体积”，但长上下文很多时候是被 KV cache 卡死的

一句话：

权重像仓库货物，KV cache 像不断堆高的周转箱。模型能放进仓库，不代表你有足够通道和周转区长期运行。

---

## 3. `preset_schemes` 速读：这些名字到底是什么意思

源码里 `PRESET_SCHEMES` 包含：

- `UNQUANTIZED`
- `W8A16`
- `W4A16`
- `W4A16_ASYM`
- `W8A8` / `INT8`
- `W4A8`
- `W4AFP8`
- `FP8`
- `FP8_DYNAMIC`
- `FP8_BLOCK`
- `NVFP4A16`
- `NVFP4`
- `MXFP4A16`
- `MXFP4`
- `MXFP8A16`
- `MXFP8`

### 3.1 命名规则

- `W` = weights（权重）
- `A` = activations（激活）
- 数字 = bit 数
- `A16` 通常表示：激活不量化，保持高精度运行（FP16/BF16 语义）
- `ASYM` = 非对称量化
- `DYNAMIC` = 推理时动态生成激活量化参数

例如：

- `W4A16`：4bit 权重，激活保持高精度
- `W8A8`：8bit 权重 + 8bit 激活
- `W4AFP8`：4bit 权重 + FP8 激活
- `FP8_DYNAMIC`：FP8 权重 + 动态 FP8 激活

### 3.2 常见策略字段怎么理解

源码里的量化粒度包括：

- `TENSOR`：整个 tensor 共用一套 scale
- `CHANNEL`：每个输出通道一套 scale
- `GROUP`：每 `group_size` 一组 scale
- `BLOCK`：二维 block（如 `128x128`）一组 scale
- `TOKEN`：按 token 动态生成激活 scale
- `TENSOR_GROUP`：局部分组 + 全局尺度，偏 NVFP4 特化

粗略理解：

- `TENSOR` 最粗，最省元数据
- `CHANNEL / GROUP` 是最常见实战路线
- `BLOCK` 更偏 tile/block 优化 kernel
- `TOKEN` 常见于 dynamic activation quant

---

## 4. 每个 preset 的实际含义

### 4.1 `UNQUANTIZED`

- 不量化
- 作为 baseline / 对照组最合适

### 4.2 `W8A16`

源码特征：

- weight: `INT8`
- `GROUP`
- `group_size=128`
- `symmetric=True`
- 只量化权重，不量化激活

实际含义：

- 8bit weight-only
- 激活走 FP16/BF16 高精度路径
- 比较稳，但压缩率不如 W4A16

### 4.3 `W4A16`

源码特征：

- weight: `INT4`
- `GROUP`
- `group_size=128`
- `symmetric=True`

实际含义：

- 4bit weight-only
- 是最经典、最实用的一类低比特部署格式
- AWQ / GPTQ 很多最终落地形态都接近这一路

### 4.4 `W4A16_ASYM`

和 `W4A16` 的主要区别：

- `symmetric=False`

实际含义：

- 4bit 非对称 weight-only
- 对偏移分布更灵活
- 但某些 kernel / runtime 支持可能比对称量化更挑

### 4.5 `W8A8` / `INT8`

源码特征：

- weights: `INT8 + CHANNEL + static`
- activations: `INT8 + TOKEN + dynamic`

实际含义：

- 权重 8bit，激活 8bit
- 但不是全静态，是“静态权重 + 动态激活”
- 这是非常常见的实战 W8A8 设计

### 4.6 `W4A8`

源码特征：

- weights: `INT4 + GROUP(128) + static`
- activations: `INT8 + TOKEN + dynamic`

实际含义：

- 权重压到 4bit
- 激活保留 INT8
- 比 W8A8 更激进，显存更省，但对 kernel 支持要求更高

### 4.7 `W4AFP8`

源码特征：

- weights: `INT4 + GROUP(128)`
- activations: `FP8 + TOKEN + dynamic`

实际含义：

- 权重 INT4
- 激活不是 INT8，而是 FP8
- 更偏“压权重 + 保留激活动态范围”的 mixed int/float 路线

### 4.8 `FP8`

源码特征：

- weights: `FLOAT8 + TENSOR + static`
- activations: `FLOAT8 + TENSOR + static`

实际含义：

- 比较朴素的全静态 FP8
- 粒度较粗
- 对激活动态范围变化较大的场景不一定最稳

### 4.9 `FP8_DYNAMIC`

源码特征：

- weights: `FLOAT8 + CHANNEL + static`
- activations: `FLOAT8 + TOKEN + dynamic`

实际含义：

- 更实战的 FP8 路线
- 权重 per-channel
- 激活 per-token 动态
- 一般比全静态 `FP8` 更好用

### 4.10 `FP8_BLOCK`

源码特征：

- weights: `FLOAT8 + BLOCK + block_structure=[128,128]`
- activations: `FLOAT8 + GROUP(128) + dynamic`

实际含义：

- block-wise FP8
- 更偏特定 kernel / 特定模型结构优化
- 适合 tile/block 友好的实现

### 4.11 `NVFP4A16` / `NVFP4`

源码特征：

- 4bit float
- `TENSOR_GROUP`
- `group_size=16`
- scale / zero-point 使用 `FP8_E4M3`
- `NVFP4` 的激活侧带 `DynamicType.LOCAL`

实际含义：

- NVIDIA 风格 FP4 路线
- 不是 INT4，而是低位浮点
- 更偏特定硬件 / 特定 kernel / 实验性格式

### 4.12 `MXFP4A16` / `MXFP4` / `MXFP8A16` / `MXFP8`

源码特征：

- `GROUP`
- 常见 `group_size=32`
- scale/zp 以 `uint8` 等形式存储

实际含义：

- 属于 microscaling 风格的低位浮点格式
- 更偏特定格式、特定 runtime 与特定 kernel 配套
- 不是默认通用首选

---

## 5. 先给结论：Qwen3.5-4B 在你这些机器上的量化建议

### 5.1 总表

| 机器 | 资源判断 | 第一推荐 | 第二推荐 | 不建议优先选 | 推荐后端 |
|---|---|---|---|---|---|
| RTX 4090 | 24GB 显存，带宽充足 | BF16/FP16（短中上下文） | `W4A16` 或 `FP8_DYNAMIC` | `Q3/Q2` 这类过低比特 | vLLM / TensorRT-LLM / SGLang |
| RTX 5060 | 8GB 显存 | `W4A16` 或 GGUF `Q4_K_M / IQ4_NL` | `W8A8`（短上下文） | BF16、全静态 FP8 | llama.cpp / Ollama / 轻量 vLLM |
| RTX 3060 | 12GB 显存，Ampere | `W4A16` | `W8A8`，或 BF16 仅短上下文 | FP8、NVFP4、MXFP* | vLLM / llama.cpp |
| AGX Orin | 统一内存，带宽远弱于桌面 4090 | GGUF `Q4_K_M / Q5_K_M` 或 `W4A16` | `W8A8`（看内存版本） | FP8、NVFP4、MXFP* | llama.cpp 优先，其次 TensorRT-LLM / vLLM |
| Orin NX | 统一内存更紧 | GGUF `Q4_K_M / IQ4_NL`，8GB 型号可退到 `Q3_K_M` | `W4A16`（16GB 型号） | BF16、W8A8、FP8 | llama.cpp 优先 |

---

## 6. 分机器详细建议

## 6.1 RTX 4090

### 结论

这是你手上最“自由”的一台机器。Qwen3.5-4B 在 4090 上不必为了“能跑起来”而激进压缩。

### 推荐顺序

#### 方案 A：BF16 / FP16

适合：

- 你想先拿到最准的基线
- 想做 prompt、采样、服务链路验证
- 上下文控制在 `4K ~ 16K` 为主

理由：

- 4B 模型 BF16 裸权重大约 `7.45 GiB`
- 24GB 显存完全能承受模型本体
- 对 4090 来说，真正限制通常变成 KV cache 和 batch，不是权重本身

#### 方案 B：`W4A16`

适合：

- 想腾更多显存给 KV cache / batch
- 想提高多并发或长上下文可用性
- 对极致精度没有基线级执念

理由：

- 权重压到约 `1.86 GiB` 级别
- 是成熟、稳定、兼容性好的方案
- 在 vLLM / AWQ / GPTQ 生态中最实用

#### 方案 C：`FP8_DYNAMIC`

适合：

- 你就是想研究 FP8 路线
- 使用支持较好的 CUDA / Tensor Core / runtime 组合
- 更关心吞吐和实验路线

注意：

- `FP8_DYNAMIC` 不是“永远比 W4A16 更好”
- 对 4B 这种小模型，本体已经不大，量化收益很多时候没有 7B/14B/32B 那么戏剧化

### 不建议优先选

- `Q3/Q2`、过于激进的 GGUF 低比特
- 没必要为了 4B 模型在 4090 上牺牲太多质量

### 推荐后端

- 生产/服务：vLLM
- 做极致性能实验：TensorRT-LLM
- 做快速验证：SGLang / Transformers

---

## 6.2 RTX 5060（按 8GB 版本看）

### 结论

这台卡的核心矛盾不是算力，而是显存只有 8GB。Qwen3.5-4B 本体不大，但 native 262K context 绝对不现实，重点要放在“给 KV cache 留空间”。

### 推荐顺序

#### 方案 A：`W4A16`

适合：

- CUDA 推理
- 你想尽量保住模型质量
- 上下文主要在 `4K ~ 8K`

理由：

- INT4 权重体积合理
- 激活保持高精度，质量/稳定性更平衡
- 是 8GB 显存上很实用的甜点位

#### 方案 B：GGUF `Q4_K_M` / `IQ4_NL`

适合：

- 本地单机聊天、轻量服务
- 追求部署简单、鲁棒性高
- 不想和 HF/vLLM 的依赖栈较劲

理由：

- 对 8GB 卡尤其友好
- llama.cpp / Ollama 路线更轻便
- `Q4_K_M` 通常是非常稳的起点；想更抠内存可以试 `IQ4_NL`

#### 方案 C：`W8A8`

适合：

- 你有明确的 INT8 kernel 路径
- 只跑短上下文

不足：

- 对 8GB 卡来说，`W8A8` 权重仍明显大于 `W4A16`
- 真正痛点是总显存预算，不是只看算子整数化

### 不建议优先选

- BF16：模型本体接近把 8GB 吃满，几乎不给 KV 留空间
- `FP8`/`FP8_DYNAMIC`：这台卡的首要矛盾是容量，不是研究 FP8

### 推荐后端

- 最省心：llama.cpp / Ollama（GGUF）
- 想走 CUDA 压缩模型：vLLM + AWQ/GPTQ 类 `W4A16`

---

## 6.3 RTX 3060（按 12GB 版本看）

### 结论

3060 12GB 是很典型的“够用型”本地部署卡。它比 5060 多出来的最大优势是显存余量，不是架构先进性。

### 推荐顺序

#### 方案 A：`W4A16`

这是 3060 上最推荐的默认解。

理由：

- 4B 模型很轻松
- 比 BF16 留出更多 KV cache 空间
- 比 `W8A8` 更省显存
- 兼容性和质量通常都不错

#### 方案 B：`W8A8`

适合：

- 你明确需要 INT8 activation 路线
- 上下文主要在 `4K ~ 8K`
- 想测试不同 kernel 的延迟/吞吐差异

#### 方案 C：BF16（仅短上下文）

适合：

- 你要做最原始的质量基线
- 上下文控制在 `2K ~ 4K`，最多到 `8K` 左右

不建议作为长期日用方案的原因：

- 3060 的 12GB 能放下 BF16 权重，但 KV cache 很快吃满
- 长一点的上下文就不划算

### 不建议优先选

- `FP8` / `FP8_DYNAMIC`
- `NVFP4` / `MXFP4` / `MXFP8`

原因：

- 3060 是 Ampere，优先走成熟的 INT4/INT8 路线更稳
- FP8/NVFP4/MX 系列在这台卡上的工程收益通常不如 W4A16 直接

### 推荐后端

- 默认：vLLM（AWQ/GPTQ 路线）
- 简洁本地：llama.cpp（GGUF）

---

## 6.4 AGX Orin

### 结论

AGX Orin 的问题不是“能不能放下 4B”，而是“统一内存 + 带宽 + edge runtime 复杂度”共同制约。它不是桌面 4090 那种大水管，而更像一台带 GPU 的高性能嵌入式系统。

### 推荐顺序

#### 方案 A：GGUF `Q4_K_M` / `Q5_K_M`

这是最推荐的 edge 落地方案。

适合：

- 本地对话 / edge service
- 希望部署链路最短
- 需要高稳定性和较少折腾

理由：

- llama.cpp 在 Jetson/ARM 侧常常是最省心的
- `Q4_K_M` 通常是性能/质量平衡点
- 如果内存是 64GB 或比较宽裕，`Q5_K_M` 可以作为更高质量选项

#### 方案 B：`W4A16`

适合：

- 你明确想走 HF/vLLM/TensorRT-LLM 生态
- 你愿意多处理 CUDA / runtime 兼容性

理由：

- 对 4B 模型已经足够轻
- 比 BF16 更符合 edge 设备现实

#### 方案 C：`W8A8`

适合：

- 你的 runtime 对 INT8 支持很好
- 机器是 32GB/64GB 版本
- 上下文需求不算特别长

### 如果你的 AGX Orin 是 64GB

- 可以跑 BF16 做 baseline
- 但仍然不建议把 BF16 当长期默认部署格式
- 原因不是放不下，而是带宽和能耗利用率不如低比特方案划算

### 如果你的 AGX Orin 是 32GB

- 更推荐 `Q4_K_M / Q5_K_M` 或 `W4A16`
- BF16 也能跑，但没有太大必要

### 不建议优先选

- `FP8` / `FP8_DYNAMIC`
- `NVFP4` / `MXFP4` / `MXFP8`

原因：

- AGX Orin 还是 Ampere 代 Jetson 思维方式
- 优先选成熟、兼容、维护成本低的路线更值当

### 推荐后端

- 第一优先：llama.cpp（GGUF）
- 第二优先：TensorRT-LLM / vLLM（如果你明确要服务化）

---

## 6.5 Orin NX

### 结论

Orin NX 是你这些设备里最需要“量入为出”的。这里别想着优雅，先想怎么稳。

### 推荐顺序

#### 16GB 版本

第一推荐：GGUF `Q4_K_M / IQ4_NL`

理由：

- 质量和内存占用比较平衡
- 比 HF 原生链更适合边缘设备
- `IQ4_NL` 有时能在质量/尺寸上给出不错折中

第二推荐：`W4A16`

前提：

- 你明确要 CUDA/HF runtime
- 你接受部署栈更重

#### 8GB 版本

第一推荐：GGUF `Q3_K_M` 或更省内存的 `IQ3_*`

理由：

- 8GB 是统一内存，不是独占显存
- OS、CUDA、allocator、page cache 都要抢空间
- 看似能装下 `Q4`，真正长期稳定跑时往往更紧

如果质量优先、上下文很短，也可以试：

- `Q4_K_M`

但要接受：

- 上下文要保守
- batch 要小
- 稳定性余量不大

### 不建议优先选

- BF16
- `W8A8`
- `FP8` / `FP8_DYNAMIC`

原因：

- Orin NX 的首要矛盾是总内存和带宽
- 不是高端桌面卡那种“先追求更精细的数值格式”场景

### 推荐后端

- 基本就是：llama.cpp 优先
- 如果不是为了做框架实验，不建议把 Orin NX 当复杂 HF 服务栈主战场

---

## 7. 机器选型背后的核心原则

### 7.1 桌面大卡：先考虑质量与吞吐

- RTX 4090：先 BF16 / W4A16，再看 FP8

### 7.2 中端卡：先考虑显存容量

- RTX 5060 8GB：优先 W4A16 / GGUF Q4
- RTX 3060 12GB：优先 W4A16，次选 W8A8

### 7.3 Jetson：先考虑统一内存和部署复杂度

- AGX Orin：优先 GGUF Q4/Q5 或 W4A16
- Orin NX：优先 GGUF Q3/Q4

### 7.4 Native 262K context 不要当成本地默认目标

Qwen3.5-4B 的原生上下文虽然是 `262,144`，但本地部署时更务实的做法是：

- 8GB 级：`2K ~ 4K` 起步
- 12GB 级：`4K ~ 8K` 起步
- 24GB 级：`8K ~ 16K` 起步，再按场景往上试
- Jetson：优先短上下文，先把稳定性做出来

---

## 8. 最后给你的直接建议

如果你现在就要把 `Qwen3.5-4B` 在这些机器上跑起来，我建议默认路线如下：

- RTX 4090：
  - 先跑 BF16 baseline
  - 日用部署转 `W4A16`
  - 要研究吞吐再试 `FP8_DYNAMIC`

- RTX 5060：
  - 默认走 `W4A16`
  - 想省心就 GGUF `Q4_K_M`

- RTX 3060：
  - 默认走 `W4A16`
  - 想研究 INT8 再试 `W8A8`

- AGX Orin：
  - 默认走 GGUF `Q4_K_M`
  - 内存宽裕且想保质量试 `Q5_K_M`
  - 想接 HF 服务栈再试 `W4A16`

- Orin NX：
  - 16GB：GGUF `Q4_K_M / IQ4_NL`
  - 8GB：GGUF `Q3_K_M / IQ3_*`

一句话总括：

- 桌面卡优先 `W4A16`
- 4090 才有资格认真考虑 BF16/FP8
- Jetson 优先 GGUF
- Orin NX 要把“能稳跑”放在“格式漂亮”前面

---

## 9. 一份非常短的速查表

| 设备 | 最推荐格式 |
|---|---|
| RTX 4090 | BF16 baseline；部署选 `W4A16` |
| RTX 5060 8GB | `W4A16` 或 GGUF `Q4_K_M` |
| RTX 3060 12GB | `W4A16` |
| AGX Orin | GGUF `Q4_K_M / Q5_K_M` |
| Orin NX 16GB | GGUF `Q4_K_M / IQ4_NL` |
| Orin NX 8GB | GGUF `Q3_K_M / IQ3_*` |

---

## 10. 备注

如果后面要继续深入，下一步最值得做的是：

1. 为每台机器写一份“建议启动命令”
   - vLLM 版
   - llama.cpp / Ollama 版
2. 统一做一次小 benchmark：
   - 4K context
   - 8K context
   - 128 output tokens
   - 对比首 token 延迟、tok/s、显存占用、温度/功耗

这样你会从“知道选什么格式”，升级到“知道为什么这台机器该这么选”。
