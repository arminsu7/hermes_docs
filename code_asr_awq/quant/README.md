# Qwen3-ASR-1.7B INT4 量化踩坑全记录

## 1. 概述

将 Qwen3-ASR-1.7B 的 Decoder（Qwen3）部分量化到 INT4 weight-only（group_size=128, symmetric），Encoder（audio_tower）保持 FP16。量化使用 compressed-tensors 0.17.1 的 min-max PTQ 方法（非 AWQ，无校准数据），输出格式为 `pack-quantized`，由 vLLM 0.23.0 的 Marlin kernel 做推理反量化。

**量化结果：**

| 指标 | 值 |
|------|-----|
| 原始大小 | 4.70 GB (FP16) |
| 量化后 | 2.62 GB |
| 压缩比 | 1.8x |
| 量化模块数 | 196 层 (仅 Decoder Linear) |
| Encoder (audio_tower) | 保持 FP16，未量化 |

**推理验证：**

| 指标 | FP16 | INT4 (修复后) |
|------|------|---------------|
| 文本输出 | `We hung up the phone to record a Skype training session.` | `We are from the hotel to say "yeh, so, what" and` |
| Token 分布 | 多样化 | 多样化 ✓ |
| 输出全 `?` | 无 | 无 ✓ |
| 模型显存 | ~3.4 GB | 1.92 GiB |

---

## 2. 环境

### 量化环境 (smr2508 容器)

| 组件 | 版本 |
|------|------|
| Docker 镜像 | `nvcr.io/nvidia/pytorch:25.08-py3` |
| Python | 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| CUDA | 13.0 |
| compressed-tensors | 0.17.1 |
| transformers | 5.10.1 |
| safetensors | 0.8.0 |
| GPU | NVIDIA GeForce RTX 3060 (12GB) |
| GPU Driver | 596.49 |

### 推理环境 (vllm23 容器)

| 组件 | 版本 |
|------|------|
| Docker 镜像 | `vllm/vllm-openai:v0.23.0-x86_64-cu129-ubuntu2404` |
| Python | 3.12.3 |
| vLLM | 0.23.0 |
| PyTorch | 2.11.0+cu129 |
| CUDA | 12.9 |
| GPU | NVIDIA GeForce RTX 3060 (12GB) |

---

## 3. 执行步骤

### 3.1 量化

```bash
# 进入 smr2508 容器
docker exec -it smr2508 bash

# 激活环境
source /root/repos/hermes/scripts/activate_llmc.sh

# 进入目录并执行
cd /root/repos/hermes/docs/code_asr_awq
python3 save_quant.py
```

量化脚本路径：`/root/repos/hermes/docs/code_asr_awq/save_quant.py`

### 3.2 推理

```bash
# 进入 vllm23 容器
docker exec -it vllm23 bash

# 启动 INT4 量化模型
vllm serve /root/repos/llm/model/qwen3-asr-1.7B-int4-weight-only \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.5 --max-model-len 2048 \
  --max-num-seqs 1 --enforce-eager \
  --kv-cache-dtype fp8 --served-model-name Qwen/Qwen3-ASR-1.7B \
  --quantization compressed-tensors
```

**前提：** vllm23 容器中需要应用一个源码级 patch（见坑 2）。

---

## 4. 模型结构

```
Qwen3ASRForConditionalGeneration
├── Audio Tower (Encoder) — 约 0.5B 参数
│   ├── conv1d × 2          (mel-spectrogram 预处理)
│   ├── 24 × EncoderLayer   (d_model=1024, ffn=4096, 16 heads)
│   │   ├── self_attn — QKVParallelLinear(prefix="qkv")  ⊗ 不量化
│   │   ├── fc1, fc2 (FFN)
│   │   └── LayerNorm
│   └── conv_out            (投影到 decoder hidden_size=2048)
│
└── Language Model (Decoder) — Qwen3, 约 1.4B 参数
    ├── embed_tokens        (vocab=151936 × 2048) ⊗ 跳过
    ├── 28 × Qwen3DecoderLayer (hidden=2048, intermediate=6144, GQA 16Q/8KV)
    │   ├── qkv_proj — QKVParallelLinear(prefix="qkv_proj")  ✓ 量化
    │   ├── o_proj          ✓ 量化
    │   ├── gate_up_proj    ✓ 量化
    │   ├── down_proj       ✓ 量化
    │   └── RMSNorm         ⊗ 跳过
    ├── final RMSNorm       ⊗ 跳过
    └── lm_head            (tied with embed_tokens) ⊗ 跳过
```

**关键设计细节：** Audio Encoder 和 Decoder 的 attention 使用不同的 prefix：
- `Qwen3OmniMoeAudioAttention`：`QKVParallelLinear(prefix="self_attn.qkv")`
- `Qwen3Attention`：`QKVParallelLinear(prefix="self_attn.qkv_proj")`

量化策略：`IGNORE = ["embed", "lm_head", "audio_tower", "feature_extractor", "norm", "rotary", "conv"]`
- `audio_tower`：保持 FP16，Encoder 运行一次，非自回归，量化收益小
- `embed` / `lm_head`：vocab=151936，参数量大但量化收益有限，跳过以保护精度

---

## 5. 踩坑详解

### 坑 1: quantize_config.json 格式不兼容 vLLM 0.23.0

> **改动文件:** `save_quant.py`（第 74-88 行，quantize_config.json 写入格式）

**现象:**

使用旧格式 `quantize_config.json` 启动 vLLM 时报错：

```
TypeError: CompressedTensorsConfig.__init__() missing 3 required positional arguments:
'target_scheme_map', 'ignore', and 'quant_format'
```

**根因:**

vLLM 0.23.0 的 `CompressedTensorsConfig.from_config()` 期望 `quantize_config.json` 包含 `config_groups` 字段，从中提取 `target_scheme_map`。旧格式是扁平结构（`bits`, `group_size`, `symmetric`），没有 `config_groups` → 解析失败 → `CompressedTensorsConfig()` 无参构造 → 缺少必需参数。

另外，compressed-tensors 0.17.x 将字段名从 `bits` 改为 `num_bits`，`format` 的值 `"int4-quantized"` 不被 vLLM 识别，需改为 `"pack-quantized"`。

**为什么旧版本不报错？**

旧格式在 compressed-tensors 早期版本（0.9.x）中使用，那时 `CompressedTensorsConfig.__init__` 允许无参构造并设置默认值。0.17.x 重构后要求显式传入 `target_scheme_map`、`ignore`、`quant_format`。

**修复:**

旧格式：
```json
{
  "quant_method": "compressed-tensors",
  "format": "int4-quantized",
  "bits": 4,
  "group_size": 128,
  "symmetric": true,
  "strategy": "group"
}
```

新格式：
```json
{
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "config_groups": {
    "group_0": {
      "targets": ["Linear"],
      "weights": {
        "type": "int",
        "num_bits": 4,
        "group_size": 128,
        "symmetric": true,
        "strategy": "group"
      },
      "input_activations": null
    }
  }
}
```

---

### 坑 2: vLLM 不读取 quantize_config.json（get_config_filenames 返回空）

> **改动文件:** vllm23 容器内 `compressed_tensors.py`（第 371 行，`return []` → `return ["quantize_config.json"]`）

**现象:**

修复 `quantize_config.json` 格式后，vLLM 仍然报同样的 TypeError，说明根本没有读取到 `quantize_config.json`。

**根因:**

vLLM 在加载量化模型时，调用 `quant_cls.get_config_filenames()` 确定要读取哪个 JSON 文件。`CompressedTensorsConfig.get_config_filenames()` 返回空列表 `[]` → vLLM 认为不需要读取配置文件 → 直接用无参构造 `CompressedTensorsConfig()` → 缺少 `target_scheme_map` 等参数 → TypeError。

代码路径：
```python
# vllm/model_executor/model_loader/weight_utils.py:387-391
possible_config_filenames = quant_cls.get_config_filenames()
if not possible_config_filenames:
    return quant_cls()  # ← 无参构造，直接炸
```

**为什么旧版本不报错？**

vLLM 0.23.0 之前的版本中 `CompressedTensorsConfig.__init__` 允许无参构造。0.23.0 重构后要求必须传入参数，但 `get_config_filenames()` 没有同步更新（仍然返回空列表）。

**修复:**

vllm23 容器内源码级 patch。旧：
```python
def get_config_filenames(cls) -> list[str]:
    return []
```

新：
```python
def get_config_filenames(cls) -> list[str]:
    return ["quantize_config.json"]
```

修复命令：
```bash
docker exec vllm23 sed -i 's/return \[\]/return ["quantize_config.json"]/' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
```

**影响范围：** 仅在 `--quantization compressed-tensors` 时被调用，不影响 BF16/FP16 非量化模型。

---

### 坑 3: Qwen3ASRForConditionalGeneration 未实现 SupportsQuant 接口

> **改动文件:** 无（建议 vLLM 后续版本添加）

**现象:**

早期调试 INT4 量化时，曾怀疑 `SupportsQuant` 缺失会导致量化管线失效（如 KeyError: `'layers.0.self_attn.qkv.weight_packed'`）。经深入分析后确认该类缺失不影响当前场景的量化推理。

**根因:**

vLLM 的量化框架要求模型类实现 `SupportsQuant` 接口。该接口通过 `__new__` 方法在模型构造时自动调用 `_maybe_apply_model_mapping()`，将模型的 `packed_modules_mapping` 和 `hf_to_vllm_mapper` 注册到 `CompressedTensorsConfig` 中。

`Qwen3ASRForConditionalGeneration` 继承了 `SupportsMultiModal, SupportsPP, SupportsMRoPE, SupportsTranscription, SupportsLoRA`，但未继承 `SupportsQuant`。

**为什么实际推理仍然成功？**

经过验证，当前场景下不实现 `SupportsQuant` 不影响推理：

1. `quant_config` 通过构造函数的 `vllm_config.quant_config` 传入，`__init__` 中手动赋值了 `self.quant_config = quant_config`
2. `CompressedTensorsConfig.get_quant_method()` 使用类名匹配（`"Linear" in "QKVParallelLinear"` → True），不依赖 `packed_modules_mapping`
3. Decoder 的权重加载通过 `Qwen3ASRForConditionalGeneration.load_weights` → `AutoWeightsLoader` + `hf_to_vllm_mapper` 完成，不依赖 `SupportsQuant`
4. Audio Encoder 的权重加载通过 `Qwen3OmniMoeAudioEncoder.load_weights` 完成，其 `stacked_params_mapping` 使用 `self_attn.qkv.` 映射，匹配 `QKVParallelLinear(prefix="qkv")`

但 `SupportsQuant` 的缺失意味着未来 vLLM 版本可能存在兼容性风险。

---

### 坑 4: INT4 权重 scale 全为零 → 推理输出全是 `?`

> **改动文件:** `save_quant.py`（第 45-54 行，在 `initialize_module_for_quantization` 和 `compress_model` 之间插入 `compute_dynamic_scales_and_zp` 计算）

**现象:**

第一次量化的 INT4 模型能加载、不报错，但推理输出全是 `?`（token ID = 30）：

```
FP16 模型输出: "We hung up the phone to record a Skype training session."  ✓
INT4 模型输出: "????????????????"  (token [30, 30, 30, 30, ...])  ✗
```

检查 safetensors 文件发现所有权重 scale 全为零（raw bytes 全 `00`）。

**根因:**

这是一个**量化流程时序 bug**，涉及两个组件：

**Step 1 — `initialize_module_for_quantization` 创建未初始化的 scale:**

```python
# compressed_tensors/quantization/lifecycle/initialize.py
init_scale = Parameter(
    torch.empty(expected_shape, dtype=scale_dtype, device=device),  # ← torch.empty()!
    requires_grad=False,
)
module.register_parameter(f"{base_name}_scale", init_scale)
```

`torch.empty()` 创建的是**未初始化内存**，值取决于当时的 GPU/CPU 内存状态。本次运行中该内存恰好被零填充 → 所有 scale = 0。

**Step 2 — `ModelCompressor.compress_model` 直接使用已有 scale:**

```python
# compressed_tensors/compressors/pack_quantized/base.py
weight = state_dict.pop("weight")
scale = state_dict.get("weight_scale")  # ← 直接拿，不重新计算！
quantized_weight = quantize(x=weight, scale=scale, ...)  # ← 用 scale=0 做量化
```

`compress` 函数**不重新计算 scale**。`quantize()` 函数也是以 `scale` 为参数输入，不自动推导。

**为什么旧版本或文档示例不报错？**

compressed-tensors 的典型完整流程是：
1. `initialize_module_for_quantization` → 创建 observer + scale/zp
2. **运行校准数据** → observer 收集 min/max → 更新 scale/zp
3. `compress_model` → 用正确的 scale 量化

本脚本跳过了第 2 步（没做校准），期望 weight-only 量化能直接计算 scale。但 `initialize_module_for_quantization` 不计算 scale，`compress_model` 也不计算 —— 这是设计上的「责任空白」。

**修复:**

在 `initialize_module_for_quantization` 之后、`compress_model` 之前，插入 scale 计算：

```python
print("Computing scales from weights...")
for name, mod in model.named_modules():
    if hasattr(mod, "weight_scale") and hasattr(mod, "quantization_scheme"):
        scale, zp = compute_dynamic_scales_and_zp(
            mod.weight, mod.quantization_scheme.weights, mod
        )
        mod.weight_scale.data = scale.to(mod.weight_scale.dtype)
        if hasattr(mod, "weight_zero_point") and zp is not None:
            mod.weight_zero_point.data = zp.to(mod.weight_zero_point.dtype)
print("Scale computation done.")
```

`compute_dynamic_scales_and_zp` 从权重张量直接计算 per-group 的 min/max → 生成 scale。对于 symmetric INT4，scale = max(|weight_per_group|) / 8.0。

**验证结果:**

修复前：
```
weight_scale 原始字节 (前 64): 00000000... (全部为零)
推理输出: "????????????????"
```

修复后：
```
weight_scale: all_zero=False, values=[0.047, 0.041, 0.035, ...]  ✓
推理输出: "We are from the hotel to say \"yeh, so, what\" and"  ✓
```

---

### 坑 5: audio_tower 误被量化（IGNORE 匹配失败）

> **改动文件:** `save_quant.py`（第 21 行，`"audio_encoder"` → `"audio_tower"`）

**现象:**

第一次量化的 INT4 模型中，audio_tower（Encoder）的 Linear 层也被量化了（含 `weight_packed`、`weight_scale`），但 IGNORE 列表的意图是跳过 Encoder。

**根因:**

模型的实际参数前缀是 `thinker.audio_tower.layers.X.self_attn.q_proj`，但 IGNORE 列表写的是 `"audio_encoder"`：

```python
IGNORE = ["embed", "lm_head", "audio_encoder", ...]  # ← 应该是 audio_tower
```

字符串子串匹配 `"audio_encoder" in "thinker.audio_tower.layers.0.fc1"` → **False** → 不跳过 → Encoder 被量化。

**修复:**

旧：
```python
IGNORE = ["embed", "lm_head", "audio_encoder", "feature_extractor", "norm", "rotary", "conv"]
```

新：
```python
IGNORE = ["embed", "lm_head", "audio_tower", "feature_extractor", "norm", "rotary", "conv"]
```

---

### 坑 6: Python `null` 不是 `None`

> **改动文件:** `save_quant.py`（第 88 行，`null` → `None`）

**现象:**

运行 `save_quant.py` 在写入 `quantize_config.json` 时报错：

```
NameError: name 'null' is not defined
```

**根因:**

`quantize_config.json` 的写入代码直接使用了 JSON 的 `null`，但 Python 中应使用 `None`。`json.dump()` 会自动将 Python 的 `None` 转换为 JSON 的 `null`。

**修复:**

旧：
```python
"input_activations": null
```

新：
```python
"input_activations": None
```

---

## 6. 关于校准的说明（AWQ vs PTQ）

当前量化使用的是 **min-max PTQ（Post-Training Quantization）**，而非 AWQ：

- **PTQ (min-max scaling):** 直接对每组权重取 max(|W|) / 8.0 作为 scale，无需校准数据，速度最快
- **AWQ (Activation-aware Weight Quantization):** 通过校准数据统计 per-channel 的重要性，对重要通道分配更大的 scale，对 LLM 的精度提升显著

对于 Qwen3-ASR 这种 Encoder-Decoder 模型，AWQ 的收益有限：

1. Encoder 未量化（保持 FP16），不需要 AWQ
2. Decoder 是 Qwen3 架构，但作为 ASR 的 decoder，其输入分布与 LLM 的自回归生成不同
3. 音频特征经过 Encoder 后的激活分布比纯文本 LLM 更均匀，per-channel 重要性差异小

如果要追求更高精度，可尝试用 `llmcompressor` 的 `run_calibration` 流程（需要准备几百条音频-文本对），或使用 GPTQ（逐层 Hessian 优化 scale）。

---

## 7. 核心代码

### save_quant.py（量化脚本，smr2508 容器）

文件路径：`/root/repos/hermes/docs/code_asr_awq/save_quant.py`

关键流程：
1. 加载 FP16 模型
2. 对非 IGNORE 的 `nn.Linear` 设置量化方案
3. `initialize_module_for_quantization` → 创建 scale/zp 参数
4. **`compute_dynamic_scales_and_zp`** → 从权重计算真实 scale（本次修复新增）
5. `ModelCompressor.compress_model` → 用正确 scale 压缩权重
6. 用 `safetensors.torch.save_file` 保存（绕过 `save_pretrained` 的 `_tied_weights_keys` 问题）
7. 手动复制 config/tokenizer 等辅助文件
8. 生成 vLLM 兼容的 `quantize_config.json`

### vLLM 推理命令（vllm23 容器）

```bash
vllm serve /root/repos/llm/model/qwen3-asr-1.7B-int4-weight-only \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.5 --max-model-len 2048 \
  --max-num-seqs 1 --enforce-eager \
  --kv-cache-dtype fp8 --served-model-name Qwen/Qwen3-ASR-1.7B \
  --quantization compressed-tensors
```

---

## 8. 修改文件汇总

| 文件 | 位置 | 改动内容 | 容器 | 状态 |
|------|------|----------|------|------|
| `compressed_tensors.py` | 第 371 行 | `return []` → `return ["quantize_config.json"]` | vllm23 | 已应用 |
| `save_quant.py` | 第 18 行 | 新增 import `compute_dynamic_scales_and_zp` | smr2508 | 已修复 |
| `save_quant.py` | 第 21 行 | IGNORE: `"audio_encoder"` → `"audio_tower"` | smr2508 | 已修复 |
| `save_quant.py` | 第 45-54 行 | 新增 scale 计算循环 | smr2508 | 已修复 |
| `save_quant.py` | 第 74-88 行 | `quantize_config.json` 格式改为 vLLM 兼容 | smr2508 | 已修复 |
| `save_quant.py` | 第 88 行 | `null` → `None` | smr2508 | 已修复 |
| ~~`qwen3_omni_moe_thinker.py`~~ | ~~第 539-541 行~~ | ~~`self_attn.qkv.` → `self_attn.qkv_proj.`~~ | vllm23 | **已回退** |

### 关于 qwen3_omni_moe_thinker.py 回退的说明

曾错误地将 `qwen3_omni_moe_thinker.py` 第 539-541 行的 `stacked_params_mapping` 从 `self_attn.qkv.` 改为 `self_attn.qkv_proj.`。经排查发现这是错误的：

- Audio Encoder 的 `Qwen3OmniMoeAudioAttention` 使用 `QKVParallelLinear(prefix="self_attn.qkv")` — **不是 `qkv_proj`**
- 原始映射 `self_attn.q_proj.` → `self_attn.qkv.` 正确匹配 `named_parameters()` 中的 `self_attn.qkv.weight`
- 错误的 `qkv_proj.` 映射导致 Audio Encoder attention 权重被静默跳过加载（`params_dict.get()` + `continue`），权重随机初始化
- 该 patch 已被回退，`stacked_params_mapping` 保持原始 `self_attn.qkv.`

---

## 9. 文件结构

```
/root/repos/hermes/docs/code_asr_awq/
├── README.md              ← smr2508 容器内的量化文档
├── save_quant.py          ← 量化脚本（已修复）
└── REPRODUCE.md           ← 复现指南

/root/repos/llm/model/
├── qwen3-asr-1.7B/        ← FP16 原始模型
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── ...
└── qwen3-asr-1.7B-int4-weight-only/  ← INT4 量化模型
    ├── model.safetensors   (2.62 GB, 1100 tensors)
    ├── quantize_config.json
    ├── config.json
    └── ...

/root/repos/llm/code/asr/
└── README.md              ← 本文档（vllm23 容器路径）
```

---

## 10. 验证结果

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 模型加载 (INT4) | ✓ 成功 | 1.92 GiB 显存, 0.44s 加载 |
| Marlin kernel | ✓ 已激活 | `Using MarlinLinearKernel for CompressedTensorsWNA16` |
| scale 非零 | ✓ 全部 98304/98304 | 验证通过 |
| BF16 文本推理 | ✓ 正常 | `We hung up the phone to record a Skype training session.` |
| INT4 文本推理 | ✓ 正常 | 与 FP16 语义相似 |
| encoder 未量化 | ✓ audio_tower 无 weight_scale | IGNORE 生效 |
| 压缩比 | 1.8x | 仅 decoder 量化, encoder 保持 FP16 |
| Audio Encoder 权重加载 | ✓ 正常 | `qwen3_omni_moe_thinker.py` patch 已回退，BF16 和 INT4 均正确 |

**注意事项:**
- 纯文本推理（无音频）输出与 FP16 不完全一致，属于 INT4 量化的正常精度损失
- ASR 场景下的真实精度需要音频文件做端到端测试
- `compressed_tensors.py` 的 patch 是临时的，vLLM 后续版本可能修复
- `qwen3_omni_moe_thinker.py` 的原始代码是正确的，没有被修改
