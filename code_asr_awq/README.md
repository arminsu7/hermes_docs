# Qwen3-ASR-1.7B INT4 Weight-Only PTQ 量化完整流程

## 概述

使用 `llmcompressor` + `compressed-tensors` 对 Qwen3-ASR-1.7B 语音识别模型进行
INT4 权重压缩 (W4A16)，group_size=128。

**量化结果:**
- 原始模型: 4.70 GB (FP16)
- 量化后: 2.17 GB (INT4 + scales)
- 压缩比: 2.2x
- 量化层数: 342 层 Linear (占总参数 73%)

**重要说明:** 本量化是 weight-only INT4 PTQ（min-max 对称量化），**不是 AWQ**。
AWQ（Activation-aware Weight Quantization）需要通过校准数据收集激活值统计来优化
per-channel 缩放因子，而本流程跳过了校准步骤（原因见下方「关于校准」一节）。

## 环境

| 组件 | 版本 |
|------|------|
| Docker 镜像 | nvcr.io/nvidia/pytorch:25.08-py3 (容器 smr2508) |
| Python | 3.12.3 |
| llmcompressor | 0.12.0 |
| transformers | 5.10.1 |
| compressed-tensors | 0.17.1 |
| accelerate | 1.13.0 |
| qwen-asr | 0.0.6 |
| GPU | NVIDIA RTX 3060 (12.9GB VRAM, SM 8.6) |

## 执行步骤

### 1. 进入容器并激活环境

```bash
docker exec -it smr2508 bash
source /root/repos/hermes/scripts/activate_llmc.sh
```

### 2. 安装 qwen-asr 支持

```bash
pip install qwen-asr
```

### 3. 恢复 transformers 版本（qwen-asr 会降级它）

```bash
pip install "transformers>=5.9.0,<=5.10.1" "accelerate==1.13.0"
```

---

## 踩坑详解

以下是量化过程中遇到的 8 个兼容性问题及其根因分析。
每个问题都说明了**现象、根因、为什么 transformers 4.x 不报错、修复方法**。

---

### 坑 1: qwen-asr 安装时降级了 transformers

> **改动文件:** 无（pip 依赖冲突，通过重装解决）

**现象:**

```
pip install qwen-asr
-> Uninstalling transformers-5.10.1
-> Installing transformers-4.57.6
-> Uninstalling accelerate-1.13.0
-> Installing accelerate-1.12.0
```

**根因:**

`qwen-asr` 0.0.6 的 `pyproject.toml` 中声明了精确版本依赖：

```
transformers==4.57.6
accelerate==1.12.0
```

pip 为了满足这个约束，强制卸载了 llmcompressor 环境中已有的 transformers 5.10.1 和 accelerate 1.13.0。

而 llmcompressor 0.12.0 的依赖是：

```
transformers>=5.9.0,<=5.10.1
```

两者版本范围**完全不重叠**，形成依赖死锁。因为 qwen-asr 基于 transformers 4.x 开发
（使用了 4.x 独有的 API），开发者锁定 4.57.6 确保稳定；llmcompressor 要求 5.x 的新特性。

**为什么这是必然发生的？**

qwen-asr 发布时 transformers 的最新稳定版是 4.57.6。它的代码依赖了 4.x 的 API
（如 `check_model_inputs()` 无参调用、`ROPE_INIT_FUNCTIONS` 包含 `"default"` 类型等）。
当 transformers 升级到 5.x 后做了大量 breaking changes，qwen-asr 没有跟进适配，
所以依赖声明只能用精确版本锁定来「自保」。

llmcompressor 的维护者（Neural Magic / vLLM 团队）紧跟着 transformers 5.x 更新了 API。
两个包的维护节奏完全错位，导致用户侧必然遇到依赖冲突。

**修复:**

先装 qwen-asr（接受降级），再强制装回 transformers 5.10.1 + accelerate 1.13.0：

```bash
pip install qwen-asr
pip install "transformers>=5.9.0,<=5.10.1" "accelerate==1.13.0"
```

pip 会输出警告，但这只是**警告**，不影响运行。真正的后果是 qwen-asr 的代码会在 5.x 上
出兼容性问题——这就是后续坑 2~5 要解决的。

---

### 坑 2: `@check_model_inputs()` 装饰器语法错误

> **改动文件:** `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`（第 1009 行，去掉括号）

**现象:**

```python
import qwen_asr
# TypeError: check_model_inputs() missing 1 required positional argument: 'func'
```

**根因:**

这是 transformers 4.x -> 5.x 的 API breaking change，涉及装饰器设计模式的变更。

在 **transformers 4.x** 中，`check_model_inputs` 是一个**装饰器工厂**：

```python
def check_model_inputs(*args, **kwargs):
    def decorator(func): ...
    return decorator
```

`@check_model_inputs()` = 调用工厂 -> 返回 decorator -> 作用于函数。

在 **transformers 5.x** 中，签名改为**直接装饰器**：

```python
def check_model_inputs(func):
    logger.warning_once("deprecated, use merge_with_config_defaults")
    return merge_with_config_defaults(func)
```

现在 `@check_model_inputs()` = 无参调用 -> 缺少 func 参数 -> TypeError。

**为什么 5.x 要改？**

`check_model_inputs` 在 4.x 中设计为装饰器工厂，但实际上几乎总是无参调用。
5.x 简化设计，同时也标记它为 deprecated——推荐用 `merge_with_config_defaults` 替代。

**修复:**

```bash
sed -i "s/@check_model_inputs()/@check_model_inputs/g" \
  qwen_asr/core/transformers_backend/modeling_qwen3_asr.py
```

去掉括号，让 `@check_model_inputs` 作为直接装饰器工作。

---

### 坑 3: `rope_type="default"` 导致 KeyError

> **改动文件:** `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`（第 785-804 行，添加 fallback + 改默认值）

**现象:**

```python
self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
# KeyError: 'default'
```

**根因:**

`ROPE_INIT_FUNCTIONS` 是一个模块级字典，将 rope_type 映射到初始化函数。

在 **4.x** 中有 `"default"` 条目（标准 RoPE），在 **5.x** 中被移除：

```
4.x: {"default", "linear", "dynamic", "yarn", "longrope", "llama3"}
5.x: {"linear", "dynamic", "yarn", "longrope", "llama3", "proportional"}
```

**为什么移除 "default"？**

5.x 重新设计了 RoPE 架构。标准 RoPE 不再通过注册表分发，改为各模型实现
`compute_default_rope_parameters` 静态方法（见坑 5）。这是一种**去中心化**转变：
从「全局字典管所有类型」变成「每个类自描述能力」。

**修复（两处）:**

**3a.** 在 `__init__` 中添加 fallback：

```python
if self.rope_type in ROPE_INIT_FUNCTIONS:
    self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
else:
    # 标准 RoPE: inv_freq[i] = 1 / base^(2i/dim)
    self.rope_init_fn = None
    self.attention_scaling = 1.0
    base = self.config.rope_theta
    dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
```

**3b.** 把 fallback 默认值从 `"default"` 改为 `"llama3"`。

---

### 坑 4: `Qwen3ASRThinkerConfig` 缺少 `pad_token_id`

> **改动文件:** `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`（第 1112 行，用 getattr 替代直接访问）

**现象:**

```python
self.config.pad_token_id
# AttributeError: 'Qwen3ASRThinkerConfig' object has no attribute 'pad_token_id'
```

**根因:**

qwen-asr 代码中对配置层级结构的误判。模型配置是三层嵌套：

```
Qwen3ASRConfig                    <- 顶层
  +-- thinker_config              <- 中间层 (Qwen3ASRThinkerConfig)
        +-- audio_config          <- 音频编码器
        +-- text_config           <- pad_token_id 在这里!
```

`self.config` 是中间层的 `Qwen3ASRThinkerConfig`，它只有子配置的引用，不直接持有 `pad_token_id`。
代码错误地在中间层访问了深层属性。

**修复:**

```python
# 原来:
self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

# 改为:
_pad_id = getattr(self.config, "pad_token_id", None)
self.pad_token_id = _pad_id if _pad_id is not None else -1
```

---

### 坑 5: `compute_default_rope_parameters` 方法缺失

> **改动文件:** `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`（第 807-818 行，新增静态方法）

**现象:**

```python
module.compute_default_rope_parameters
# AttributeError: not found
```

**根因:**

5.x 新增约定：所有自定义 RoPE 类**必须**提供 `compute_default_rope_parameters` 静态方法。
调用链：`from_pretrained -> _init_weights -> 检查 compute_default_rope_parameters`。

这是配合坑 3 的配套设计——5.x 不再通过 `ROPE_INIT_FUNCTIONS["default"]` 来查找标准 RoPE，
改为每个 RoPE 类自声明能力。qwen-asr 基于 4.x，自然没有。

**修复:**

在 `Qwen3ASRThinkerTextRotaryEmbedding` 中添加：

```python
@staticmethod
def compute_default_rope_parameters(config=None, device=None, seq_len=None):
    base = config.rope_theta
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
    return inv_freq, 1.0  # (inv_freq, attention_scaling=1.0)
```

---

### 坑 6: `_tied_weights_keys` 导致 `save_pretrained` 崩溃

> **改动文件:** 未修改 qwen-asr 源码。绕过 `model.save_pretrained()`，改用 `safetensors.torch.save_file` 直接保存（见 `save_quant.py` 第 48-56 行）

**现象:**

```python
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
# AttributeError: 'list' object has no attribute 'keys'
```

错误栈定位到 transformers 5.x 的 `_get_tied_weight_keys` 函数（modeling_utils.py 第 391-396 行）：

```python
def _get_tied_weight_keys(module):
    tied_weight_keys = []
    for name, submodule in module.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", {}) or {}
        tied_weight_keys.extend([
            f"{name}.{k}" if name else k
            for k in tied.keys()  # <- list 没有 .keys()!
        ])
    return tied_weight_keys
```

**根因:**

`_tied_weights_keys` 是 nn.Module 的可选属性，标记「权重绑定」（weight tying）。
常见于 LLM 中 `lm_head.weight` 和 `embed_tokens.weight` 共享权重，保存时需去重。

这个属性的**类型约定**在 4.x -> 5.x 之间变了：

```
4.x: list[str]   例: ["lm_head.weight", "embed.weight"]
5.x: dict[str, str]  例: {"lm_head.weight": "embed.weight"}
```

5.x 的 `_get_tied_weight_keys` 假设类型是 dict，调用了 `.keys()` 方法。

**具体到 qwen-asr 源码**：`Qwen3ASRThinkerForConditionalGeneration` 的类定义
（modeling_qwen3_asr.py 第 1093 行）：

```python
class Qwen3ASRThinkerForConditionalGeneration(...):
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    #                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 非空 list!
```

这是一个**非空 list 类属性**。`save_pretrained` 内部调用 `_get_tied_weight_keys`
遍历子模块时遇到这个类实例：

```python
tied = getattr(submodule, "_tied_weights_keys", {}) or {}
# -> 取到 list ["model.embed_tokens.weight", "lm_head.weight"]
# -> or {} 不触发（非空 list 是 truthy，不会 fallback）
# -> tied 仍然是 list
# -> tied.keys() -> AttributeError!
```

（如果是 `None` 或空 `[]`，`or {}` 会 fallback 到空 dict 安全跳过——但这里不是。）

**为什么 4.x 不报错？**

4.x 的 `_get_tied_weight_keys` 也同样调用 `.keys()`，但 4.x 中 qwen-asr 的
这个类属性格式可能是 dict，或者 4.x 的 `save_pretrained` 没有调用这个函数。
当模型以 5.x 格式保存时，属性类型没有同步更新。

**修复:**

由于问题出在 `save_pretrained` 内部函数调用，且修改 qwen-asr 类属性可能影响
其他功能或被下次升级覆盖，最稳妥的方式是**绕过 `save_pretrained`**，
直接用 safetensors 底层 API 保存权重：

```python
from safetensors.torch import save_file

# 手动收集 state_dict
state_dict = {}
for name, param in model.state_dict().items():
    state_dict[name] = param.contiguous().cpu()

# 绕过 save_pretrained，直接写 safetensors
save_file(state_dict, os.path.join(OUTPUT_DIR, "model.safetensors"))
```

然后手动复制 config.json、tokenizer.json 等辅助文件到输出目录。

**备选方案**（在根源上修复 class 属性，不推荐——qwen-asr 升级会覆盖）：

```python
# list -> dict 转换，使其兼容 5.x 格式
cls = type(model.thinker)  # Qwen3ASRThinkerForConditionalGeneration
cls._tied_weights_keys = {k: k for k in cls._tied_weights_keys}
# ["lm_head.weight", "embed.weight"] -> {"lm_head.weight": "lm_head.weight", ...}
```

---

### 坑 7: generation_config 中 temperature 与 do_sample 冲突

> **改动文件:** `/root/repos/llm/model/qwen3-asr-1.7B/generation_config.json`（删除 `temperature` 字段）

**现象:**

```
ValueError: temperature is set but do_sample=False.
temperature is only used in sample-based generation.
```

**根因:**

5.x 新增严格配置验证。`temperature` 只在采样模式有效，但 qwen3-asr 配置中有
`temperature=1e-06 + do_sample=False`，逻辑矛盾。

**修复:**

删除 `temperature` 字段（do_sample=False 时不需要）。

---

### 坑 8: `strategy="channel"` + `group_size` 被 pydantic 拒绝

> **改动文件:** `/root/repos/hermes/docs/code_asr_awq/quantize_awq.py`（`strategy="channel"` -> `strategy="group"`）

**现象:**

```python
QuantizationArgs(num_bits=4, group_size=128, strategy="channel")
# ValidationError: group_size requires strategy='group'
```

**根因:**

compressed-tensors 的语义：
- `strategy="channel"`: per-channel，整个 channel 一组 scale，不需要 group_size
- `strategy="group"`: per-group，每 group_size 个元素一组 scale

pydantic validator 拦截了矛盾组合。

**修复:**

`strategy="channel"` -> `strategy="group"`。

---

### 坑 1bis: `thinker_config` 初始化时序问题

> **改动文件:** `qwen_asr/core/transformers_backend/configuration_qwen3_asr.py`（第 402 行预置属性、第 423-425 行 None 检查）

**现象:**

```python
config = Qwen3ASRConfig.from_pretrained(model_path)
# AttributeError: 'Qwen3ASRConfig' object has no attribute 'thinker_config'
```

**根因:**

经典构造时序问题。调用链：

```
Qwen3ASRConfig.__init__()
  +-- super().__init__(**kwargs)
        +-- validate_token_ids()           # 5.x 新增的验证
              +-- self.get_text_config()   # 多态，走子类覆盖
                    +-- self.thinker_config.get_text_config()  # 崩! thinker_config 还没赋值
  +-- self.thinker_config = ...   # 太晚了
```

`validate_token_ids` 需要 text_config 来验证 token id，触发了 `get_text_config()`。
子类覆盖的 `get_text_config` 直接访问 `self.thinker_config`，但它在 `super().__init__()` **之后**才赋值。

**为什么 4.x 不报错？**

`validate_token_ids` 是 5.x 新增的验证逻辑，4.x 的 `PretrainedConfig.__init__` 没有这个步骤。

**修复（两处）:**

**1bis-a.** 在 `super().__init__()` 之前预置属性：

```python
def __init__(self, ...):
    object.__setattr__(self, "thinker_config", None)  # 绕过 __setattr__ 拦截
    super().__init__(**kwargs)
```

用 `object.__setattr__` 而非 `self.x = None`，因为 `PretrainedConfig` 重写了 `__setattr__`，
直接赋值可能被拦截或触发额外验证。

**1bis-b.** `get_text_config` 加 None 检查：

```python
def get_text_config(self, decoder=False):
    if self.thinker_config is not None:
        return self.thinker_config.get_text_config()
    return super().get_text_config(decoder=decoder)
```

初始化阶段 think_config 为 None 时 fallback 到父类；初始化完成后走正常路径。

---

## 关于校准：为什么不是 AWQ？

**真正的 AWQ** 需要：

1. 用一批校准数据跑模型，收集每层 Linear 输入的激活值统计（per-channel magnitude）
2. 根据激活值判断每个 channel 的重要性：激活大的 channel 对量化误差更敏感
3. 对重要 channel 用更保守的缩放因子，不重要 channel 可以更激进

**本流程跳过了 AWQ 校准**，原因：

1. **ASR 模型吃音频，没有现成的文本校准集。** llmcompressor 的 oneshot API
   和 AWQ workflow 默认用文本数据集（wikitext、c4 等），而 qwen3-asr 的输入是
   mel-spectrogram 音频特征，需要音频校准数据。

2. **compressed-tensors 直接路径本身就只做 min-max 量化。** 我们使用的
   initialize_module_for_quantization + ModelCompressor.compress_model
   这条路径没有校准阶段，直接对权重做 symmetric min-max 量化到 INT4。

**本流程实际做的是：weight-only INT4 PTQ（Post-Training Quantization）。**

### 精度影响

对 ASR 模型的预期精度损失：

- **保守策略。** 我们跳过了 audio_encoder 的全部层（卷积 + 编码器 attention），
  只量化了 text_model（transformer decoder）的 Linear 层。音频特征提取部分
  保持 FP16 精度，这是对 ASR 精度影响最大的环节。

- **group_size=128 的细粒度量化。** 每 128 个权重共享一组 scale，相比 per-channel
  或 per-tensor 量化，能更精确地捕捉权重分布，减少量化误差。对称量化范围 [-8, 7]
  对零附近对称分布的解码器权重通常精度损失 < 1% 困惑度。

- **对比 AWQ：** AWQ 通过激活值校准能再减少 10-30% 的量化误差（相比 min-max PTQ），
  但前提是有合适的校准数据。对于 ASR 模型，AWQ 的增益可能不如 LLM 显著，
  因为语音模型的解码器权重分布和纯文本模型不同。

- **最终判断需要 WER 测试。** 精度应该在具体 ASR 基准（如 LibriSpeech test-clean
  的 WER 或 AISHELL 中文 CER）上验证。如果 WER/CER 劣化 < 0.5%，可以认为量化成功。

**如果需要恢复 AWQ 校准**，需要：

1. 准备音频校准集（几十条音频，覆盖目标语言）
2. 修改脚本：用校准音频跑 forward，hook 每层 Linear 的输入，记录 per-channel max
3. 计算缩放因子替换 min-max scale，然后执行量化 + 保存

---

## 量化配置

```python
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, initialize_module_for_quantization,
)
from compressed_tensors.compressors import ModelCompressor

weight_args = QuantizationArgs(
    num_bits=4, type="int", symmetric=True,
    group_size=128, strategy="group",
)
scheme = QuantizationScheme(targets=["Linear"], weights=weight_args, input_activations=None)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and not should_ignore(name):
        module.quantization_scheme = scheme
        initialize_module_for_quantization(module, scheme)

compressor = ModelCompressor.from_pretrained_model(model)
compressor.compress_model(model)
```

### 跳过层

| Pattern | 原因 |
|---------|------|
| `embed` | Embedding 层, 通常不量化 |
| `lm_head` | 输出投影, 对精度敏感 |
| `audio_encoder` | 音频编码器, 对量化敏感 |
| `feature_extractor` | 特征提取层 |
| `norm` | LayerNorm/RMSNorm |
| `rotary` | RoPE 位置编码 |
| `conv` | Conv1d 层 |

---

## vLLM 验证

vLLM v0.23.0 可以开始加载模型，但 GPU 显存不足（RTX 3060 12GB 被 smr2508 占用）。
释放 GPU 后预期可加载。

**注意:** qwen3-asr 是 ASR 模型，非标准 LLM。vLLM 主要优化 decoder-only 架构。
推荐使用 transformers 原生推理。

## 文件结构

```
/root/repos/hermes/docs/code_asr_awq/
+- README.md          # 本文档
+- quantize_awq.py    # 量化脚本
+- save_quant.py      # 保存脚本

/root/repos/llm/model/
+- qwen3-asr-1.7B/        # 原始 (FP16, 4.70 GB)
+- qwen3-asr-1.7B-awq/    # 量化 (INT4, 2.17 GB)
```
