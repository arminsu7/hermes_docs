# 任务1：VLA 推理 TRT 加速方案

> 容器：whh_trtllm_r130r20（TRT-LLM 环境，ViT engine 构建）/ whh_vla_trt2（TRT 10.11，LLM engine 构建 + 部署）
> 代码目录：/root/workspace/embodichain/hpc_opt/trtllm/
> 不修改原始 dexechain/ 代码
> 所有 engine 导出统一使用 checkpoint-46000 权重
> ⚠️ **VLA 真实推理模型是 FP32**（config `mixed_precision: "no"`），不是 FP16——详见 5.11 修正

---

## 一、方案概述

### 目标

ViT + LLM 全部用 TRT engine 加速，不走 PyTorch 混合推理。所有 engine 导出统一使用 checkpoint-46000 权重。

---

## 二、checkpoint-46000 权重分析

### 2.1 LLM 部分

VLA checkpoint-46000 的 LLM 与 HF Qwen3-VL-2B-Instruct **完全一致**（max_diff=0.0，逐 bit 相同）。VLA 训练时冻结了 LLM（`llm_trainable_idx=0`）。

| 对比项 | VLA checkpoint-46000 | HF Qwen3-VL-2B-Instruct | 一致？ |
|--------|----------------------|-------------------------|--------|
| type | Qwen3VLTextModel | Qwen3VLTextModel | ✅ |
| num layers | 28 | 28 | ✅ |
| hidden_size | 2048 | 2048 | ✅ |
| num_heads / kv_heads | 16 / 8 | 16 / 8 | ✅ |
| head_dim | 128 | 128 | ✅ |
| intermediate_size | 6144 | 6144 | ✅ |
| 参数值 max_diff | 0.0 | 0.0 | ✅ 逐 bit 一致 |

### 2.2 ViT 部分

VLA checkpoint-46000 的 ViT 与 HF Qwen3-VL-2B-Instruct **不完全一致**——layer 12-23 经过微调（`vision_tower_trainable_idx=12`）。

| 对比项 | 结果 |
|--------|------|
| 总参数 | 315 |
| 完全相同 | 193（layer 0-11 + patch_embed + merger + deepstack_merger） |
| 有差异 | 122（layer 12-23 的 attn/mlp/norm） |
| 差异幅度 | max_diff ~0.03, mean_diff ~0.001-0.002（轻微微调） |

> **重要**：ViT engine 必须用 checkpoint-46000 权重导出，否则 image_hidden_state max_diff=249。

---

## 三、Qwen3-VL DeepStack 机制

Qwen3-VL 的 ViT 不只输出最终 pooler_output，还输出 3 个 deepstack_features（来自 ViT layer 5/11/17）。LLM 的 forward 需要：
- `inputs_embeds`：embed_tokens(input_ids) → masked_scatter(image_embeds)
- `visual_pos_masks`：image token 位置 mask
- `deepstack_visual_embeds`：3 个中间层特征，注入 LLM layer 0/1/2

### TRT-LLM 官方做法

TRT-LLM 的 Qwen3-VL model 实现（`tensorrt_llm/_torch/models/modeling_qwen3vl.py`）有完整的 DeepStack 支持：

1. ViT 输出 pooler_output [288, 2048] + 3× deepstack [288, 2048] each
2. 在 dim=1 拼接成 [288, 8192]（hidden_dim × (1 + num_deepstack_levels)）
3. 通过 prompt_embedding_table 传给 LLM engine
4. LLM forward 内部 split 成 4 份，pooler 注入 inputs_embeds，3× deepstack 分别注入 layer 0/1/2

> serve.py 只输出 1 个 pooler_output 是因为它简化了（只做文本生成），不代表架构不需要 DeepStack。

---

## 四、prepare_inputs（输入预处理，待优化）

### 4.1 代码位置

文件：`dexechain/agents/dexforce_vla/models/multimodal_encoders/vlms/qwen2_5_vl.py`
方法：`prepare_inputs()`（L352~398）

### 4.2 功能

`prepare_inputs` 是 ViT 和 LLM 共用的输入准备步骤，在 `forward_for_dexforcevla`（L432）中被调用：

```python
# forward_for_dexforcevla L432
inputs = self.prepare_inputs(images, instructions)
# ...
outputs = self.vlm(**inputs, output_hidden_states=True)  # L441
```

### 4.3 处理流程

1. BGR → RGB（图像通道翻转）
2. tensor → PIL Image（每张相机图像转成 PIL）
3. `processor.apply_chat_template`（构造 chat 文本模板）
4. `process_vision_info`（图像 patch 信息）
5. `processor(text, images)`（tokenize + 图像预处理）
6. `pixel_values → to(vlm.device), to(dtype=self.dtype)`（**self.dtype = FP32**，VLA 是 FP32 模型）

### 4.4 输出

返回 `inputs` dict，包含 ViT 和 LLM 共需的输入：

| key | shape | dtype | 用途 | 给谁 |
|-----|-------|-------|------|------|
| pixel_values | [1152, 1536] | **float32** | 图像 patchified 输入（576 patches × 2 图 = 1152，VLA FP32 模型） | ViT engine（内部 cast 到 FP16） |
| image_grid_thw | (2, 3)，内容 `[[1,24,24],[1,24,24]]` | int64 | 图像网格尺寸（T=1帧, H=W=24 patches，384/16=24；已烘焙进 ViT engine） | ViT（常量） |
| input_ids | [1, 316] | int64 | token 序列（含 image_pad 占位，288 视觉 + 28 文本，测试数据 316） | LLM engine |
| attention_mask | [1, 316] | int64 | attention mask（全 1，无 padding） | LLM engine |
| mm_token_type_ids | [1, 316] | int64 | 多模态 token 类型（M-RoPE position_ids 计算） | LLM engine |

> 全部 shape/dtype 由 `llm_pt_reference_with_ds.safetensors` 实测确认（17 tensor）。注意 cos/sin/position_ids 不在 prepare_inputs 产出里——它们是 LLM forward 内部由 rotary_emb 计算的（M-RoPE 3D）。

### 4.5 在 TRT pipeline 中的角色

走 TRT ViT engine + TRT LLM engine 路线时，`prepare_inputs` 仍在 CPU 侧执行：
- 产出的 `pixel_values`（fp32）喂给 ViT engine——**TrtEngine hook 内部自动 cast 到 FP16**（engine 是 FP16 的，见 task1c）
- 产出的 `input_ids` 喂给 LLM engine

> 待优化：当前 `prepare_inputs` 在 VLA profile 中耗时 ~11ms（实测 prepare_inputs 11.29ms，CPU 侧 PIL 转换 + HF processor tokenize），GPU 化优化见 **4.6**。

### 4.6 prepare_inputs GPU 化优化（2026-08-31，进行中）

> 代码：`hpc_opt/trtllm/prepare_inputs/gpu_prepare_inputs.py` + 验证 `verify_gpu_prepare5.py`
> 耗时剖析（profile 打点实测，prepare_inputs 11.29ms）：

| 子模块 | 耗时 | 占比 | 设备 | 优化 |
|--------|------|------|------|------|
| processor（PIL resize + tokenize） | 6.45ms | 57% | CPU | **GPU resize/normalize** |
| to_pil（tensor→PIL） | 2.89ms | 26% | CPU | **跳过（直接 GPU tensor）** |
| to_gpu | 0.79ms | 7% | GPU | 保留 |
| vision_info | 0.17ms | 1.5% | CPU | 保留（小） |
| template | 0.16ms | 1.4% | CPU | 保留（小） |

**GPU 化方案**（已实现 + 验证）：
1. **图像不走 to_pil_image**——以 GPU tensor 直接处理
2. `smart_resize`（对齐 processor，factor=patch_size*merge_size=16*2=32）
3. `tvF.resize`（GPU，BICUBIC + antialias）——实测 CPU/GPU max_diff=0.0
4. `rescale_normalize`（GPU）——实测与 processor 一致 1.19e-7
5. `patch_merge`（view/permute/reshape，GPU）——对齐 processor
6. 文本部分（apply_chat_template + tokenize）**保留 CPU**（文本量小且依赖可变指令）

**关键参数**（Qwen3-VL-2B，从 config 确认）：
- patch_size=16, merge_size=2, temporal_patch_size=2 → patch_dim = 3×2×16×16 = 1536
- smart_resize factor = 16×2 = 32
- 真实输入：4 相机（hand/high × left/right），每相机 6 帧历史，每帧 384×1152 uint8（hdf5 实测）
- VLA config：image_size=384, num_patches=144（(384/16/2)² = 12² = 144）

**验证结果**（verify_gpu_prepare5.py，随机图 384×1152 BGR）：
```
原始 processor 输出: [1728, 1536]  gthw [1,24,72]
GPU 化输出:        [1728, 1536]  gthw [1,24,72]
max_diff = 1.19e-7（float32 舍入级）✅ 数值等价
```

**关键结论**：
- prepare_inputs 的 11.29ms 里 **9.3ms（82%）是 CPU 图像处理**，GPU 化省 ~8ms
- 文本和图像**每帧都变** → 无"值"级预计算（input_ids/position_ids/cos/sin 全依赖每帧内容）
- 可预计算的只有静态结构：engine IO buffer 复用 + CUDA graph 捕获（省固定 launch/分配开销）

### 4.6.1 hook 集成（2026-08-31 已完成）

> 代码：`prepare_inputs/gpu_prepare_inputs.py` + `prepare_inputs/prepare_inputs_gpu_hook.py` + `trt_infer_shim.py`

**集成方式**（hook 替换，不改原始代码）：
- `trt_infer_shim.py` 的 `_patched_forward_for_dexforcevla` 里，对 `self`（Qwen25VLEncoder）调 `install_prepare_inputs_gpu_hook(self)`
- hook 用**代理对象**替换 `processor.image_processor`（`__call__` 走 GPU 化），文本 tokenize 保留原始
- 幂等 + 静默失败（不影响 ViT/LLM hook）

**踩坑**：
1. **Python `__call__` 特殊方法陷阱**：查类不查实例，不能改 `image_processor.__call__` 属性 → 必须整体替换为代理对象
2. **模型结构**：真实链路 `install_trt_hooks` 传的是 `self.vlm`（Qwen3VL），prepare_inputs 在 Qwen25VLEncoder 上 → 由 shim 对 `self` 直接装
3. **静默原则**：shim 完全静默（print=0），GPU 化失败静默降级

**对齐验证**（verify_prepare_gpu_hook.py，真实模型 + 正方形 384×384 双图）：
```
input_ids / attention_mask / mm_token_type_ids : max_diff = 0.0
pixel_values  : max_diff = 1.19e-7（float32 舍入级）
image_grid_thw: max_diff = 0.0
✅ 与原始 prepare_inputs 数值等价
```

**端到端验证**（真实链路 infer_trt.sh，V1+L2 TRT）：
```
prepare_inputs GPU 化已启用
Input preparation time : 0.00s（GPU 化前 0.01~0.06s）
VLM forward            : 0.01s（TRT 正常）
EXIT = 0
```

**性能收益**：prepare_inputs **11.29ms → ~3.5ms**（中位数 0ms，加速 ~4 倍）

**进度**：✅ 图像 GPU 化 + hook 集成 + 对齐验证已完成（见 4.6.1）；**延迟隐藏（4.7）待实现**

### 4.6.2 self.vlm 内部耗时锚点分析（2026-09-01）

> 目的：精细分析 `self.vlm(**inputs)`（L472）内部耗时——ViT / LLM 是否完全由 engine 替代，还有哪些 torch 边界步骤。
> 实现：transformers `modeling_qwen3_vl.py` 的 Qwen3VLModel.forward + Qwen3VLForConditionalGeneration.forward 加 time() 锚点，
> 写入 `_profile_speed`，qwen2_5_vl.py 采集，runner 合并（vlm_model./vlm_cgm. 前缀）。
> ⚠️ transformers 文件已改（备份 .bak_profile），分析后可回滚。

**实测（原始 torch 链路，100 iter avg/min/max）**：

```
vlm.forward                        : 50.00 / 44.32 / 59.48 ms  ← self.vlm(**inputs) 总
└ vlm_cgm.model                    : 49.72 / 44.10 / 59.21 ms  ← Qwen3VLForCGM → self.model(...)
    ├ vlm_model.embed              :  0.10 ms  (embed_tokens)
    ├ vlm_model.get_image_features : 29.58 ms  (★ ViT 图像编码, 59%)
    ├ vlm_model.mask_inject        :  0.41 ms  (masked_scatter 注入)
    ├ vlm_model.compute_position_ids: 0.84 ms  (M-RoPE 3D)
    └ vlm_model.language_model     : 18.73 ms  (LLM 前 27 层, 37%)
└ vlm_cgm.lm_head                  :  0.03 ms  (lm_head logits)
```

**结论**：
1. **vlm.forward 50ms = ViT 29.58ms（59%）+ LLM 18.73ms（37%）+ torch 边界 ~1.4ms**
2. **被 engine 替代**：`get_image_features`（ViT）→ ViT engine；`language_model`（LLM 前 27 层）→ LLM engine
3. **engine 边界开销（始终 torch，不可消除）**：embed 0.10 + mask_inject 0.41 + M-RoPE 0.84 + lm_head 0.03 ≈ **1.38ms**
4. **TRT 化收益**：48.3ms（ViT+LLM）被 engine 替代 → 省 ~30ms（实测 TRT 链路 ~17ms），剩 1.38ms torch 边界
5. **锚点层级**：vlm_cgm.model（外层）包住 vlm_model.*（内层），lm_head 最后——已修正（曾误并列）

### 4.6.3 TrtEngine 输出 buffer 复用（2026-09-01）

> 目的：消除 `TrtEngine.__call__` 每次推理的**输出 buffer 分配 + set_tensor_address** 重复开销。
> 代码：`trt_hooks.py` 的 `TrtEngine`（`reuse_output_buffers` 参数 + `TRT_REUSE_OUTPUT` 环境变量）。

**问题**：`__call__` 每次推理都做 `torch.empty` 分配输出 + `set_tensor_address` 重绑地址。实测分项：

| 引擎 | 完整 __call__ | 输入处理 | set shape+addr | 输出分配 | 输出 set_addr | execute_v2 | sync |
|------|--------------|---------|---------------|---------|--------------|-----------|------|
| ViT | 4.43ms | 0.38ms | 0.43ms | **0.46ms** | **0.46ms** | 2.70ms | 0.003ms |
| LLM | 5.20ms | 0.02ms | 0.04ms | 0.04ms | 0.04ms | 5.06ms | 0.003ms |

→ **ViT 引擎输出分配 + 地址设置 = 0.92ms（占 21%）**，是主要可优化空间（LLM 几乎全在 execute_v2，优化空间小）。

**方案**：输出 shape 固定（单指令 + 多帧图像 shape 一致）→ **预分配一次，后续复用**：

```python
# trt_hooks.py
def __init__(self, plan_path, reuse_output_buffers=True):
    ...
    self.reuse_output_buffers = reuse_output_buffers
    self._out_buffers = {}   # name -> tensor（复用缓存）

def __call__(self, inputs):
    ...
    if self.reuse_output_buffers and name in self._out_buffers \
            and self._out_buffers[name].shape == shape \
            and self._out_buffers[name].dtype == tdtype:
        out = self._out_buffers[name]   # 复用（省分配 + 重设地址）
    else:
        out = torch.empty(shape, dtype=tdtype, device="cuda")
        self.ctx.set_tensor_address(name, out.data_ptr())
        self._out_buffers[name] = out
```

**开关**（出问题时关闭，不改代码）：
```bash
bash infer_trt.sh                    # 默认开（TRT_REUSE_OUTPUT=1）
TRT_REUSE_OUTPUT=0 bash infer_trt.sh # 关闭复用
```

**验证**（verify_reuse.py）：
```
ViT: reuse vs noreuse max_diff=0.0 ✅  复用收益 0.39ms
LLM: reuse vs noreuse max_diff=0.0 ✅  复用收益 0.32ms
复用别名（两次调用同一 buffer）= True（复用生效）
```

**收益**：ViT ~0.39ms + LLM ~0.32ms ≈ **~0.7ms**（TRT 链路 ~17ms → ~16.3ms）。

**保护机制**：
- **shape/dtype 变化自动重新分配**（复用前检查 `shape==` 和 `dtype==`）
- **别名风险**：复用返回同一块 buffer——跨调用保存输出引用会被覆盖。VLA 链路 hidden_states 是"forward 内即消费"（喂 _get_image_hidden_state），不跨帧保存，安全。若未来出现跨帧缓存，用 `TRT_REUSE_OUTPUT=0` 关闭。

### 4.6.4 get_image_hs 预计算 index 优化（2026-09-01）

> 目的：`vlm.get_image_hs`（13.2ms）是 **boolean mask expand + fancy indexing 后处理**（非模型推理），
> 用**预计算 image token 位置 index** 替代每帧 expand，预计 13.2ms → ~1-2ms。
> 代码：`get_image_hidden_state/`（hook 方式，仿 prepare_inputs），经 `trt_infer_shim.py` 安装。

**发现（实测验证）**：image token 位置 index **不依赖图像内容，只依赖图像数量**（配置固定 2 图 → 288 个 image_pad 占位符）。

| 文本 | seq_len | image token 位置数 | 位置范围 |
|------|---------|-------------------|---------|
| 文本0 | 316 | 288 | [4,5,...,291,292,293] |
| 文本1 | 309 | 288 | [4,5,...,291,292,293] |
| 文本2 | 305 | 288 | [4,5,...,291,292,293] |

**位置 0 vs 1 相同: True / 位置 0 vs 2 相同: True** → 即使文本内容/长度变化，image token 位置固定（始终是固定前缀 [4..293] 连续段）。

**为什么**：`image_hs_mask = input_ids == image_token_id` 判断占位符位置，由**图像数量/顺序**决定，不是图像内容。图像占位符在 input_ids 固定前缀，文本变化只影响后半段文本 token。

**优化方案**（hook 替换 `_get_image_hidden_state`）：
```python
# 当前：每帧 expand + boolean gather（13.2ms）
image_hs_mask_expanded = mask.view(*mask.shape, 1).expand(-1, -1, hidden_size)
image_hidden_state = final_hidden_state[image_hs_mask_expanded].view(bs, -1, hidden_size)

# 优化：预计算 index + index_select（~1-2ms）
image_idx = (input_ids_template == image_token_id).nonzero()[:, 1]  # [288] 一次算好
image_hidden_state = final_hidden_state[:, image_idx, :]            # 连续 gather
```

**前提**：图像数量固定（任务内 2 图）。若图像数量变化，index 需重算（按 image 数量缓存多个模板）。

**实现**（已集成，经 `trt_infer_shim.py` 安装）：
- 代码：`get_image_hidden_state/gpu_get_image_hs.py` → `install_get_image_hs_hook` 替换 `_get_image_hidden_state`
- index 首次从 input_ids 计算后缓存，每次调用**校验 image token 数量**（一致复用，变化重算）——保证正确性
- 无 image token 时回退原始实现；shim 静默失败不影响 ViT/LLM

**单元验证**（verify_gihs.py）：
```
输出 shape: orig (1,288,2048) vs fast (1,288,2048)
max_diff: 0.000000（一致=✅）
mask 一致: True
原始(boolean mask expand): 0.052 ms
优化(index_select)       : 0.037 ms（1.4x）
```

**端到端实测**（真实链路 infer_trt.sh，100 iters avg）：
```
vlm.get_image_hs : 13.20 ms → 3.71 ms（省 ~9.5ms，72%）
vlm.forward      : ~17 ms   → 13.07 ms（TRT 链路）
```

**⚠️ 根因分析（重要）**：TRT 链路下 get_image_hs 耗时降低**主要是测量/等待机制，不全是 hook 功劳**：

1. `vlm.get_image_hs` 用 CPU `time()` 计时，但 GPU kernel 异步
2. `_get_image_hidden_state` 第一个 GPU 操作会**隐式等待前面的 vlm.forward kernel 完成**，等待被计入计时
3. **torch 链路**：vlm.forward 是 50ms 慢推理，GPU 队列堆满 kernel，get_image_hs 开始时还在等 → 13.2ms 大部分是"等 GPU"
4. **TRT 链路**：vlm.forward 13ms 且 **engine `execute_v2` 自带 `torch.cuda.synchronize()`**（trt_hooks L74），get_image_hs 开始时 GPU 已同步 → 3.71ms 等待少

**证据**：独立测 `_get_image_hidden_state` 算法（torch/TRT 输入都连续 FP32）两者都是 **0.040ms**（几乎一样）——真实算法开销极小，13.2ms 主要是不该计入的异步等待。预计算 index 优化真实收益是算法 0.052→0.037ms（1.4x），非 9.5ms。

### 4.7 延迟隐藏（规划）

GPU 化后，prepare_inputs 的图像处理（~3ms GPU）可与上一帧的 LLM 推理（~10ms GPU）**overlap**（用 CUDA stream 并发）：
- 当前串行：prepare_inputs(11ms) → VLM forward(50ms) → 总计 61ms
- 延迟隐藏：prepare_inputs 提前到上一帧 forward 期间执行 → 隐藏 ~3-8ms
- 前提：图像处理在独立 stream，不阻塞 LLM kernel

---

## 五、ViT ONNX 导出（checkpoint-46000，4 输出）

### 5.1 导出脚本

脚本：`hpc_opt/trtllm/vit/export_vit_384_fp16.py`（FP16 版）/ `export_vit_384_fp32.py`（FP32 版）

- **FP16 版**：`torch_dtype=torch.float16` 加载，pixel_values `.half()`——产物 `vit_qwen3vl_384_fp16.onnx`（与 FP16 engine 同精度；实际部署的 engine 选择见 5.11 4-engine 对比）
- **FP32 版**：`torch_dtype=torch.float32` 加载，pixel_values 不降精度——与 VLA 原始链路（mixed_precision="no"）同精度，产物 `vit_qwen3vl_384_fp32.onnx`（研究 FP32 链路用）
- 两者共用 `vit_wrapper.py` 的 `ViTWithMergerWrapper` + `load_vit_from_checkpoint`

基于 `trtllm-qwen3-vl/export_vit_onnx.py`（1920x1080 版本）修改：
- 输入改为 2 张 384x384 图像（VLA 双相机）
- 模型权重用 checkpoint-46000（ViT layer 12-23 微调）
- 输出 4 个 tensor：pooler_output + 3× deepstack_features

> **为什么需要 4 个输出**：TRT-LLM 的 Qwen3-VL model 实现里有完整的 DeepStack 支持。官方做法是把 pooler_output 和 3 个 deepstack_features 在 dim=1 拼接成 [288, 8192]，通过 prompt_embedding_table 传给 LLM engine，LLM 内部 split 后在 layer 0/1/2 分别注入。serve.py 只输出 1 个是因为它简化了（只做文本生成），不代表架构不需要。

### 5.2 Wrapper 设计

```python
class ViTWithMergerWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values, grid_thw):
        out = self.visual(pixel_values, grid_thw=grid_thw)
        if isinstance(out, tuple):
            pooler = out[0]
            deepstack = out[1]
        else:
            pooler = out.pooler_output
            deepstack = out.deepstack_features
        ds_list = []
        for ds in deepstack:
            if isinstance(ds, (list, tuple)):
                ds = torch.cat(ds, dim=0)
            ds_list.append(ds)
        return pooler, ds_list[0], ds_list[1], ds_list[2]
```

> **坑：wrapper 初始化不能加 `.to("cuda").half()`**。`visual` 对象已经在 cuda 且 fp16 了，重复 `.to("cuda").half()` 会触发 PyTorch 隐式行为导致数值变化（pooler_output max_diff=0.098，deepstack_2 max_diff=0.75）。去掉后与原始 forward 逐 bit 一致。正确写法：`wrapper = ViTWithMergerWrapper(visual)`，不要链式调用 `.to().half()`。

### 5.3 Monkey-patch 绕过 GQA

PyTorch ONNX exporter 不支持 `scaled_dot_product_attention` 的 `enable_gqa` 参数：

```python
_orig_sdpa = F.scaled_dot_product_attention
F.scaled_dot_product_attention = lambda *a, **kw: _orig_sdpa(
    *a, **{k: v for k, v in kw.items() if k != "enable_gqa"}
)
```

### 5.4 checkpoint-46000 权重加载

从 `model.safetensors` 中提取 ViT 权重（key 含 `.visual.`），通过直接映射加载到 `model.model.visual`。315 个参数全部匹配（missing=0, unexpected=0）。

### 5.5 grid_thw 被烘焙进 engine

ONNX trace 时 ViT 内部调用 `grid_thw.tolist()`，导致 grid_thw 被固化为常量。因此：
- engine 只有 1 个输入 `pixel_values`，没有 `grid_thw` 输入
- 更改图像数量或分辨率需要重新导出 ONNX + 构建 engine

### 5.6 输入输出规格

| 项 | 值 |
|----|-----|
| 输入 pixel_values | [1152, 1536] float16（576 patches × 2 images = 1152） |
| grid_thw（常量） | [[1, 24, 24], [1, 24, 24]]（T, H, W = 1帧, 24×24 patches，384/16=24） |
| 输出 pooler_output | [288, 2048] float16（含 merger） |
| 输出 deepstack_0 | [288, 2048] float16（ViT layer 5） |
| 输出 deepstack_1 | [288, 2048] float16（ViT layer 11） |
| 输出 deepstack_2 | [288, 2048] float16（ViT layer 17） |

### 5.7 导出结果

| 项 | 值 |
|----|-----|
| ONNX 原始 | 776.7 MB |
| ONNX 精简后 | 774.9 MB |
| ONNX 节点数 | 1624 |
| ONNX 校验 | ✅ 通过 |

### 5.8 TRT FP16 Engine 构建

脚本：`hpc_opt/trtllm/vit/build_vit_384_engine.py`（支持 `--fp32` / `--rotary-inner-fp32`）

```bash
# FP16（默认）
python3 build_vit_384_engine.py

# 或用 trtexec
/usr/local/tensorrt/bin/trtexec \
    --onnx=vit_engine_384/vit_qwen3vl_384_simplified.onnx \
    --saveEngine=vit_engine_384/vit_qwen3vl_384.plan \
    --fp16

# FP32
python3 build_vit_384_engine.py --fp32

# FP16 + rotary 内部 FP32（推荐，精度更好，见 5.12 第5节实验）
python3 build_vit_384_engine.py --rotary-inner-fp32
```

| 项 | FP16 | FP32 | FP16+rotary_fp32 |
|----|------|------|------------------|
| Engine 大小 | 783.0 MB | 1554.4 MB | 825 MB |
| GPU Compute Time | 3.75 ms | 9.92 ms | 3.75 ms |
| 构建结果 | ✅ PASSED | ✅ PASSED | ✅ PASSED |

> build_vit_384_engine.py 修复了 TRT 10 API：`trt.Flag.EXPLICIT_BATCH` → `1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)`

> `--rotary-inner-fp32` 用 `--layerPrecisions` 强制 rotary 内部 336 个 FP32 算子（rotary_inner_fp32_layers.txt）以 FP32 计算——实测精度提升 22-35%、无性能损失（见 5.12 第5节）。注意 `--layerPrecisions` 需用 **ONNX 原始层名**（如 `/visual/blocks.0/attn/Mul`）而非 TRT 融合层名 `__myl_*`（约束在融合前应用）。

### 5.9 产物文件

> 注：旧导出脚本（export_vit_384.py）已拆分为 fp16/fp32 两版，旧产物名 `vit_qwen3vl_384.onnx` 为 FP16 版早期产物（等价于 `vit_qwen3vl_384_fp16.onnx`）。当前以带精度标识的新命名为准。

**ONNX 产物**：

| 文件 | 说明 |
|------|------|
| vit/vit_engine_384/vit_qwen3vl_384_fp16.onnx | FP16 版原始 ONNX（776.7 MB） |
| vit/vit_engine_384/vit_qwen3vl_384_fp16_simplified.onnx | FP16 版精简 ONNX（774.9 MB，1624 节点） |
| vit/vit_engine_384/vit_qwen3vl_384_fp32.onnx | FP32 版原始 ONNX（1552.9 MB） |
| vit/vit_engine_384/vit_qwen3vl_384_fp32_simplified.onnx | FP32 版精简 ONNX（1548.9 MB，1528 节点） |

**Engine 产物（5 组合，对比见 5.11）**：

| 目录 | onnx | engine | rotary | plan 文件 |
|------|------|--------|--------|-----------|
| fp16onnx_fp16engine/ | fp16 | fp16 | fp16 | vit_qwen3vl_384_fp16.plan（824 MB） |
| fp16onnx_fp16engine_rotaryinner_fp32/ | fp16 | fp16 | fp32 | vit_qwen3vl_384_fp16_rotaryinner_fp32.plan（825 MB） |
| fp32onnx_fp16engine/ | fp32 | fp16 | fp16 | vit_qwen3vl_384_fp16.plan（824 MB） |
| fp32onnx_fp16engine_rotaryinner_fp32/ | fp32 | fp16 | fp32 | vit_qwen3vl_384_fp16_rotaryinner_fp32.plan（825 MB） |
| fp32onnx_fp32engine/ | fp32 | fp32 | fp32 | vit_qwen3vl_384_fp32.plan（1.6 GB） |

### 5.10 ViT wrapper vs PyTorch 验证

验证脚本：`hpc_opt/trtllm/vit/verify_vit_wrapper.py`
共享模块：`hpc_opt/trtllm/vit/vit_wrapper.py`（`ViTWithMergerWrapper` 类 + `load_vit_from_checkpoint` 函数，export_vit_384.py / verify_vit_wrapper.py 共用，避免代码漂移）

验证方式：用 checkpoint-46000 的 ViT 权重加载到 HF 模型，对比 `model.visual(pixel_values, grid_thw)` 原始输出与 `ViTWithMergerWrapper` 包装后输出。

> **注意**：这里的"PyTorch"是 **FP16 加载的 HF 参考模型**（`torch_dtype=torch.float16`，与 FP16 ViT engine 同精度），**不是 VLA 原始链路**（原始链路为 FP32，mixed_precision="no"）。本验证只证明 wrapper 壳的透明性（数据流正确），不涉及 FP16 vs FP32 的推理精度对比。

#### 对比 1：wrapper 壳透明性（FP16，同精度）

| 输出 | max_diff | 结果 |
|------|---------|------|
| pooler_output | 0.0 | ✅ 逐 bit 一致 |
| deepstack_0 | 0.0 | ✅ |
| deepstack_1 | 0.0 | ✅ |
| deepstack_2 | 0.0 | ✅ |

Wrapper 与原始 PyTorch 完全对齐（max_diff=0.0）——**wrapper 壳透明，导出用的包装不改变数值**（5.10 原结论不变）。

#### 对比 2：FP32 wrapper vs 真实链路截取值（新增，真实数据）

用 `llm_pt_reference_with_ds.safetensors` 的真实数据（**FP32 pixel_values + FP32 模型**，复现 VLA 真实链路）：

- 真实 deepstack：截取的 `deepstack_0/1/2`（FP32）
- 真实 pooler：**重建**——`inputs_embeds[0, visual_mask 位置]`（masked_scatter 填入的 image_embeds = pooler 行，mask 288 个位置验证通过）

| 输出 | max_diff | mean_diff | 结果 |
|------|---------|-----------|------|
| pooler_output | 0.000000 | 0.000000 | ✅ 逐 bit 复现 |
| deepstack_0 | 0.000000 | 0.000000 | ✅ |
| deepstack_1 | 0.000000 | 0.000000 | ✅ |
| deepstack_2 | 0.000000 | 0.000000 | ✅ |

**wrapper 用真实 FP32 输入能逐位复现真实链路的 ViT 输出**——与 LLM wrapper 的 FP32 实验（0.0000）互相印证，整个 wrapper 体系在真实链路上逐位对齐。

#### 对比 3：FP16 wrapper vs 真实链路截取值（新增，量化 FP16/FP32 gap）

FP16 模型 + FP16 输入 vs 真实 FP32 链路，量化 **FP16 推理的固有精度 gap**：

| 输出 | max_diff | mean_diff | 结果 |
|------|---------|-----------|------|
| pooler_output | 0.0354 | 0.0014 | ✅ |
| deepstack_0 | 0.0059 | 0.0003 | ✅ |
| deepstack_1 | 0.4693 | 0.0009 | ✅ |
| deepstack_2 | **1.3222** | 0.0024 | ✅（<2.0 阈值） |

- mean 全部 <0.003（平均差异极小）
- max 集中在 deepstack_2（ViT layer 17 最深，FP16 累积误差最大）——**规律与 5.11 engine 验证一致**（FP16 engine 里 ds2 也是最大列，如 A=1.525、B=1.834），证明是 FP16 固有行为非 bug
- 对比 3 的 1.32（FP16 wrapper）< 5.11 的 engine 差值（A=1.525~C=4.504）：纯 FP16 torch vs FP32 torch 无 TRT kernel 差异，engine 还叠加了 TRT kernel 误差
- **结论**：FP16 ViT vs 真实 FP32 链路差异 ≈ 0.04~1.3（max）/ <0.003（mean），全部在 FP16 合理范围——这也是 FP16 engine 部署后端到端误差 <1% 的 ViT 层面依据

### 5.11 ViT TRT engine 数值验证 + 性能 benchmark

验证脚本：`hpc_opt/trtllm/vit/verify_vit_engine.py`（复用 vit_wrapper.py）

验证环境：whh_vla_trt2（TRT 10.11.0.33），checkpoint-46000 权重，真实输入（`llm_pt_reference_with_ds.safetensors`，pixel_values 原生 FP32 直接喂，2026-08-31 起 engine IO 统一 fp32:chw），真实输出重建 pooler + 截取 3×deepstack（FP32），engine 输出转 FP32 后对比；耗时 100 iter 平均。

#### 5 engine 对比：onnx 精度 × engine 精度 × rotary 精度（2026-08-31 更新，全部 fp32 IO）

> 2026-08-31 起：ViT engine 的 inputIOFormats/outputIOFormats 统一改为 **fp32:chw**（对齐真实 VLA 链路输入输出 FP32）。以下 5 个 engine 均以 fp32 IO 重新构建（build_vit_384_engine.py 改 IO 后），用 verify_vit_engine.py 对同一真实链路 target 实测。

**5 个 engine 的构建组合**（`export_vit_384_fp16/fp32.py` 导出 ONNX → `build_vit_384_engine.py` 构建）：

| tag | onnx 精度 | engine 精度 | rotary 精度 | 目录 |
|-----|-----------|------------|------------|------|
| A0 | fp16 | fp16 | fp16 | fp16onnx_fp16engine/ |
| A | fp16 | fp16 | fp32 | fp16onnx_fp16engine_rotaryinner_fp32/ |
| B | fp32 | fp16 | fp16 | fp32onnx_fp16engine/ |
| C | fp32 | fp16 | fp32 | fp32onnx_fp16engine_rotaryinner_fp32/ |
| D | fp32 | fp32 | fp32 | fp32onnx_fp32engine/ |

**耗时表**（benchmark 100 iters + 10 warmup）：

| engine | onnx | engine | rotary | 耗时 ms |
|--------|------|--------|--------|---------|
| **A0** | fp16 | fp16 | fp16 | **3.920** |
| A | fp16 | fp16 | fp32 | 3.896 |
| B | fp32 | fp16 | fp16 | 3.930 |
| C | fp32 | fp16 | fp32 | 3.927 |
| D | fp32 | fp32 | fp32 | 10.260 |

（参考：PyTorch ViT FP16 10.02 ms/iter，FP16 engine 相对加速 ~2.6x）

**数值表（max_diff，输出转 FP32 vs 真实链路）**：

| 输出 | A0(fp16,fp16,fp16) | A(fp16,fp16,fp32) | B(fp32,fp16,fp16) | C(fp32,fp16,fp32) | D(fp32,fp32,fp32) |
|------|--------------------|-------------------|-------------------|-------------------|-------------------|
| pooler max | **0.054** | 0.195 | 0.074 | 0.065 | 0.009 |
| deepstack_0 max | **0.020** | 0.022 | 0.021 | 0.025 | 0.002 |
| deepstack_1 max | **0.072** | 0.091 | 0.091 | 0.103 | 0.014 |
| deepstack_2 max | **0.288** | 0.344 | 0.327 | 0.428 | 0.041 |

（mean_diff 全部 <0.01，此处省略；D 全 FP32 精度王）

**结论（fp32 IO 新数据）**：
1. **A0（fp16onnx + fp16engine + rotary fp16）成为最优 FP16 方案**——4 个输出 max_diff 全列最优（0.054/0.020/0.072/0.288），耗时 3.920ms 也最低（噪声级）
2. **fp32 IO 下 rotary fp16 反而比 rotary fp32 好**（A0 < A，B < C）——与旧 fp16 IO 时代结论相反（当时 rotary fp32 有优势）。fp32 边界 + rotary fp16 的组合精度最佳
3. **D（全 FP32）精度王**（max 低 1-2 个数量级），但 10.26ms 慢 2.6x
4. **部署选型更新**：FP16 方案从"B（fp32onnx）"改为 **A0（fp16onnx）**——精度最好且耗时最低；精度优先仍用 D
5. **耗时规律**：FP16 engine 全部 ~3.9ms，FP32 engine 10.3ms——rotary 精度不影响耗时，只有 engine 精度影响

> 注意：此处验证的是 ViT 层面的 4 个输出，不是 VLA 最终的 image_hidden_state。VLA 的 image_hidden_state 是 ViT 输出经过 LLM 28 层前向后再从 hidden_states[-2] 提取的，需要 LLM engine 做好后再端到端验证。

### 5.12 ViT 精度机理分析（PyTorch / ONNX / TRT 三层）

> 依据：PyTorch 源码检查 + ONNX 节点分析 + TRT engine 的 layerinfo.json / profile.json（`--profilingVerbosity=detailed` 导出）。以下结论全部有文件证据，无推测。

#### 1. PyTorch 源码层面：ViT 的精度（两种场景）

> ⚠️ 早期版本把 ViT 精度写成"FP16 + 局部 FP32 upcast"，混淆了场景——**只有 FP16 导出场景才有分层；VLA 真实链路（FP32）是全 FP32**（与 LLM 的 RMSNorm 同理，见 6.2）。

- **VLA 真实推理链路（部署目标）**：模型 FP32（config `mixed_precision: "no"`，bf16 权重经 `model.to()` 无损升 fp32——转换点详见 task1b 3.7）——**ViT 全部子模块 FP32 计算**，无 FP16 分层
- **FP16 导出场景（export_vit_384_fp16.py 用 torch_dtype=float16）**：权重 FP16，只有 rotary（`apply_rotary_pos_emb_vision`）内部 upcast 到 FP32 再降回，其余（LayerNorm/Linear/MLP/merger）FP16

**FP16 导出场景**下各子模块精度（`named_parameters()` dtype 集合 = {torch.float16}）：

| 组件 | 类型 | 精度 | 说明 |
|------|------|------|------|
| patch_embed (Conv) | Qwen3VLVisionPatchEmbed | FP16 | - |
| norm1 / norm2 | **nn.LayerNorm** | FP16（IO） | Python 层无显式 upcast（与 LLM 的 RMSNorm 不同，RMSNorm 源码强制 `.to(float32)`）；但 LayerNorm 的均值/方差归约对精度敏感，TRT 实现内部行为见第 3 节 |
| attn.qkv / attn.proj | Linear | FP16 | - |
| rotary（apply_rotary_pos_emb_vision） | 函数 | **FP32** | Q/K 计算 rotary 时 upcast 到 FP32，计算完降回 FP16 |
| mlp (linear_fc1/fc2 + act_fn) | Qwen3VLVisionMLP | FP16 | - |
| merger / deepstack_merger_list | Qwen3VLVisionPatchMerger | FP16 | - |

**结论**：ViT 不是纯 FP16——rotary embedding 部分有 FP32 upcast，其余是 FP16。**VLA 真实链路则是全 FP32**。

#### 2. ONNX 节点层面：Cast 节点统计

对 `vit_qwen3vl_384_fp16_simplified.onnx` 的全节点分析：

| Cast 类型 | 数量 | 对应结构 |
|-----------|------|---------|
| FP16→FP32 | 48 | 24 blocks × 2（Q、K 各一个，rotary 前 upcast） |
| FP32→FP16 | 48 | 24 blocks × 2（Q、K 各一个，rotary 后 downcast） |
| **总计** | **96** | **全部在 attention 的 rotary embedding 路径上** |

按节点名精确分类（脚本统计）：

| 节点名模式 | 转换 | 数量 |
|------------|------|------|
| `attn/Cast`, `attn/Cast_1` | FP16→FP32 | 48（rotary up） |
| `attn/Cast_2`, `attn/Cast_3` | FP32→FP16 | 46（rotary down） |
| `attn/Cast_4`, `attn/Cast_5`（仅 blocks.0） | FP32→FP16 | 2（rotary down 等价物） |

**ONNX 里的 LayerNorm 无 Cast**。`/visual/blocks.0/norm1/LayerNormalization` 节点的输入输出：

```
in:  /visual/Add_3_output_0 (FP16), weight (FP16), bias (FP16)
out: LayerNormalization_output_0 (FP16)
```

前后也没有任何 Cast 节点——ONNX 中精度转换只存在于 rotary 路径。

以 blocks.4/attn 为例的 rotary Cast 链（ONNX 节点）：
```
qkv/Gemm (FP16) → Squeeze [HALF] → Cast [FP16→FP32]
  → Mul (scaling) → Slice/Neg/Concat (rotary 分解) → Mul (sin) → Add (合成) [全部 FP32]
  → Cast [FP32→FP16] → MatMul/Softmax/MatMul [FP16]
```

#### 3. TRT engine 中 Cast 的存在形式

> 依据：`vit_qwen3vl_384_fp16_layerinfo.json`（256 层）和 `vit_qwen3vl_384_fp32_layerinfo.json`（432 层）的 Metadata 字段（记录每个融合层融合了哪些原始 ONNX 层）。

**ONNX 的 96 个 Cast 在最终 engine 中被完全消除，没有融合进任何层，也没有独立存在。**

证据：FP16/FP32 engine 所有层的 Metadata 共引用 1503 个 ONNX 层，其中 **0 个 Cast**。

- **FP16 engine**：全部 254 个计算层 IO 均为 Half，0 个 Half↔Float 转换层。rotary 融合层 `__myl_TranSlic...MulMulAdd...` 的 Metadata 列出 31 个融合的 ONNX 层（Reshape/Transpose/Split/Slice/Neg/Concat/Mul/Add），无 Cast——**rotary 在 FP16 下计算**（ONNX 的 FP16→FP32 Cast 被消除）
- **FP32 engine**：全 engine 仅 5 个精度转换层（下表）。Cast 被消除的原因是输入已升为 FP32，upcast 为恒等操作
- LayerNorm 融合层 `__myl_AddCastMeanSubMul...CastMulAdd` 的 Metadata 只含 `Add`（residual）+ `LayerNormalization` 两个 ONNX 层——层名中的 Cast 是 **TRT 展开 LayerNormalization 实现时自己插入的**（归约部分 FP32 计算），非 ONNX Cast。FP32 engine 对应层名无 Cast（`AddMeanSubMul...`）可佐证

FP32 engine 中仅存的 5 个精度转换层：

| 层 | 位置 | 转换 | 作用 |
|----|------|------|------|
| `[1] __myl_ReshCastConc` | 输入边界 | Half → Float | pixel_values (FP16 输入) 升到 FP32 进入网络 |
| `[415] __myl_Cast` | 输出边界 | Float → Half | pooler_output 降回 FP16 |
| `[420] __myl_Cast` | 输出边界 | Float → Half | deepstack_2 降回 FP16 |
| `[425] __myl_Cast` | 输出边界 | Float → Half | deepstack_1 降回 FP16 |
| `[430] __myl_Cast` | 输出边界 | Float → Half | deepstack_0 降回 FP16 |

#### 4. FP16/FP32 engine 中各层的实际计算精度

> 依据：layerinfo.json 中每层 Inputs/Outputs 的 `Format/Datatype` 字段。

**FP16 engine（256 层）**：

| 层 | 位置 | 输入 dtype | 输出 dtype |
|----|------|-----------|-----------|
| `/visual/patch_embed/proj/Conv` | patch embed | Half | Half |
| `__myl_AddCastMeanSub...CastMulAdd` | LayerNorm+residual（融合，Cast 为 TRT 插入） | Half | Half |
| `/visual/blocks.N/attn/qkv/Gemm` | qkv 投影 | Half | Half |
| `__myl_TranSlic...MulMulAdd...` | rotary（融合） | Half | Half |
| `_gemm_mha_v2` | attention (QK^T+softmax+V) | Half | Half |
| `__myl_FcMulMulMulAddMulTanhAddMulMul` | MLP 激活（融合） | Half | Half |
| `/visual/blocks.N/mlp/linear_fc2/Gemm` | MLP 投影 | Half | Half |

全 engine dtype 统计：**Half 254 层**（另 2 层为辅助层）。

**FP32 engine（432 层）**：

| 层 | 位置 | 输入 dtype | 输出 dtype |
|----|------|-----------|-----------|
| `[1] __myl_ReshCastConc` | 输入边界 | **Half** | **Float** |
| `/visual/patch_embed/proj/Conv` | patch embed | Float | Float |
| `__myl_AddMeanSubMul...` | LayerNorm+residual（融合，无 Cast） | Float | Float |
| `/visual/blocks.N/attn/qkv/Gemm` | qkv 投影 | Float | Float |
| `__myl_TranSlic...MulMulAdd...` | rotary（融合，无 Cast） | Float | Float |
| `/visual/blocks.N/attn/MatMul` | QK^T | Float | Float |
| `__myl_MaxrSubExpSumDivMul` | softmax（独立融合层） | Float | Float |
| `[415-430] __myl_Cast` ×4 | 输出边界 | **Float** | **Half** |

全 engine dtype 统计：**Float 426 层，Half 4 层**（Half 的 4 层即上表的输入边界 1 层 + 输出边界 4 层，其中第 1 层 IO 两端各计一次）。

#### 5. layerPrecisions 数值验证实验（rotary 强制 FP32）

> 用 `--layerPrecisions`（ONNX 原始层名 + `--precisionConstraints=obey`）验证 rotary 的实际计算精度，并为精度优化提供方案。

提取 ONNX rotary 路径中被 Cast 包裹的全部 FP32 算子（336 个：Mul×96 + Slice×96 + Neg×48 + Concat×48 + Add×48），写入 `vit_engine_384/rotary_inner_fp32_layers.txt`，构建强制 FP32 的 engine C 与默认 engine A 对比：

| 输出 | A（默认）max | C（rotary 内部 FP32）max | 改善 |
|------|-------------|------------------------|------|
| pooler_output | 0.695 | **0.453** | -35% |
| deepstack_0 | 0.027 | 0.023 | -14% |
| deepstack_1 | 0.666 | 0.660 | -1% |
| deepstack_2 | 3.813 | **2.969** | -22% |

耗时：A 3.770 ms vs C 3.751 ms（-0.5%，噪声级）——rotary 内部 FP32 **无性能代价**。

实验结论：
1. **数值验证了"默认 FP16 engine 的 rotary 在 FP16 下计算"**——强制 FP32 后输出改变（A vs C 互差 max=1.02），若原本就是 FP32 则不会有变化
2. `--layerPrecisions` 需使用 **ONNX 原始层名**（网络定义阶段的层）才生效；对融合后的 `__myl_*` 层名无效（约束在融合前应用）
3. **实用优化**：用 `rotary_inner_fp32_layers.txt` 构建 engine 可白拿 22-35% 的精度提升，无性能损失

构建命令见 5.8（`python3 build_vit_384_engine.py --rotary-inner-fp32`）。

**小结**：
- **PyTorch**：ViT 权重 FP16（FP16 导出场景），rotary embedding 内部 FP32 upcast；Norm 用 LayerNorm，Python 层无显式 upcast。VLA 真实链路则是全 FP32
- **ONNX**：96 个 Cast（24 层 × Q/K 各一对 upcast/downcast），**全部在 rotary 路径；LayerNorm 在 ONNX 无 Cast**
- **FP16 engine**：全层 IO Half。**ONNX 的 96 个 Cast 被 TRT 完全消除（Metadata + layerPrecisions 数值双重证据）——rotary 在 FP16 下计算**，与 PyTorch 的 FP32 rotary 存在精度差异。LayerNorm 融合层名中的 Cast 是 TRT 展开 LayerNormalization 时插入的（归约 FP32），非 ONNX Cast
- **FP32 engine**：入口一次 Half->Float（`__myl_ReshCastConc`），全程 FP32 计算（426 层 Float），出口 4 次 Float->Half（pooler + 3×deepstack）。ONNX 的 Cast 因恒等被消除
- **两者数值均能与 PyTorch 对齐**（FP16 engine vs 真实链路 max_diff 见 5.11 4-engine 对比：A=1.525、D=1.678 量级，均 FP16/FP32 合理范围）
- **精度优化**：rotary 内部算子强制 FP32（第 5 节实验）可将 FP16 engine 的 max_diff 再降 22-35%，无性能损失

---

## 六、LLM 结构分析

### 6.1 基本信息

| 参数 | 值 |
|----|-----|
| type | Qwen3VLTextModel |
| num layers | 28 |
| hidden_size | 2048 |
| num_heads | 16 |
| num_kv_heads | 8（GQA，groups=2） |
| head_dim | 128 |
| intermediate_size | 6144 |
| vocab_size | 151936 |
| M-RoPE | 3D position_ids [3, 1, seq_len] |
| DeepStack | layer 0/1/2 后注入 deepstack_visual_embeds |

### 6.2 Decoder Layer 模块结构

根据 `Qwen3VLTextDecoderLayer.__init__` 和 `forward` 源码（`transformers/models/qwen3_vl/modeling_qwen3_vl.py`），每层由 4 个子模块组成：

```python
# decoder_layer.forward 执行顺序
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)          # 模块1: RMSNorm
hidden_states, _ = self.self_attn(...)                        # 模块2: Attention
hidden_states = residual + hidden_states                      # residual add

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)  # 模块3: RMSNorm
hidden_states = self.mlp(hidden_states)                       # 模块4: MLP
hidden_states = residual + hidden_states                      # residual add
```

| 子模块 | 类型 | 内部组件 |
|--------|------|---------|
| input_layernorm | Qwen3VLTextRMSNorm | weight [2048]，**内部 FP32 计算**（pow/rsqrt 用 float32） |
| self_attn | Qwen3VLTextAttention | q_proj/k_proj/v_proj/o_proj（Linear）、q_norm/k_norm（RMSNorm）、rotary_emb（**内部 FP32**）、SDPA（GQA） |
| post_attention_layernorm | Qwen3VLTextRMSNorm | weight [2048]，**内部 FP32 计算** |
| mlp | Qwen3VLTextMLP | gate_proj/up_proj/down_proj（Linear）、act_fn=SiLU |

> 精度列：上述 Linear/SDPA 在 **FP16 导出场景**是 FP16 计算，在 **VLA 真实链路（FP32）**是 FP32 计算——见下方"各子模块的精度分析"两种场景对照。

跨层共享的组件：
- **rotary_emb**（Qwen3VLTextRotaryEmbedding）——所有层共享同一个实例，预计算 cos/sin 后传入每层。**内部强制 FP32 计算**（inv_freq.float() + position_ids.float() + maybe_autocast(enabled=False)）
- **deepstack 注入**（官方 `_deepstack_process`：index 赋值 `h[mask, :] += ds`）——只在 layer 0/1/2 有（旧 wrapper 的 torch.where 写法已废弃，见 task1b 5.7）
- **residual add**——每层 2 次

**每一层的子模块类型完全相同**（28 层全部一致），仅参数值不同。通过 named_children 实际实例验证 layer 0/13/27：

| layer | self_attn | mlp | input_layernorm | post_attention_layernorm |
|-------|-----------|-----|-----------------|--------------------------|
| 0 | Qwen3VLTextAttention | Qwen3VLTextMLP | Qwen3VLTextRMSNorm | Qwen3VLTextRMSNorm |
| 13 | Qwen3VLTextAttention | Qwen3VLTextMLP | Qwen3VLTextRMSNorm | Qwen3VLTextRMSNorm |
| 27 | Qwen3VLTextAttention | Qwen3VLTextMLP | Qwen3VLTextRMSNorm | Qwen3VLTextRMSNorm |

#### 各子模块的精度分析

> ⚠️ **分两种场景**（早期版本混淆，已修正）：
> - **VLA 真实推理链路（部署目标）**：模型 FP32（config `mixed_precision: "no"`，bf16 权重经 `model.to()` 无损升 fp32）——**全部子模块 FP32 计算**（RMSNorm/rotary 的 `.to(float32)` 是恒等，SDPA/Linear 也是 FP32），无 FP16 分层。**export_llm_onnx.py 现按此场景导出纯 FP32 ONNX**（FP32 权重 + FP32 输入输出，sdpa attention，2026-08-28 起）
> - **FP16 导出场景（历史版本，已废弃）**：早期 export_llm_onnx.py 用 torch_dtype=float16 导出（对应旧 FP16 ONNX / FP16 TRT engine）。权重 FP16，SDPA/Linear 在 FP16 计算，RMSNorm/rotary 内部 upcast 到 FP32 再降回——**只有这个场景下才有"FP16 计算 + 局部 FP32 upcast"的分层**。以下分析针对此场景（历史记录，保留供 FP16 engine 排查参考）。

**FP16 导出场景**（历史）下，权重全部 FP16（`named_parameters()` dtype 集合 = {torch.float16}）。但以下组件内部有 FP32 upcast：

| 组件 | 源码中的 FP32 upcast | 计算精度 | IO 精度 |
|------|---------------------|---------|---------|
| RMSNorm | `hidden_states.to(torch.float32)` → pow(2).mean → rsqrt → `.to(input_dtype)` | FP32 | FP16 |
| rotary_emb | `inv_freq.float()` + `position_ids.float()` + `maybe_autocast(enabled=False)` | FP32 | FP16 |
| SDPA attention | 无 upcast | FP16 | FP16 |
| Linear (proj) | 无 upcast | FP16 | FP16 |
| MLP (SiLU) | 无 upcast | FP16 | FP16 |

**VLA 真实链路（FP32）下同一张表**：

| 组件 | 计算精度 | IO 精度 | 说明 |
|------|---------|---------|------|
| RMSNorm | FP32 | FP32 | `.to(float32)` 恒等 |
| rotary_emb | FP32 | FP32 | 同上 |
| SDPA attention | FP32 | FP32 | 无 upcast，本来就是 FP32 |
| Linear (proj) | FP32 | FP32 | 同上 |
| MLP (SiLU) | FP32 | FP32 | 同上 |

> **对 TRT 的意义**：VLA 真实链路是纯 FP32，而 TRT engine 是 FP16——engine 的 FP16 计算（含 SDPA/Linear）与真实链路的 FP32 存在固有精度差（1-2 ULP），这正是 task1b 里"engine vs 真实链路 2 ULP"的来源之一。RMSNorm/rotary 的 FP32 强制（layerPrecisions）是在 FP16 engine 内部尽量复刻真实链路的 FP32 计算，但 SDPA/Linear 无法强制（会 OOM），只能接受 FP16 精度。

RMSNorm 的 forward 源码：
```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype              # FP16
    hidden_states = hidden_states.to(torch.float32)  # upcast to FP32
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)  # back to FP16
```

rotary_emb 的 forward 源码：
```python
def forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, None, :, None].float()          # FP32
    position_ids_expanded = position_ids[:, :, None, :].float()              # FP32
    with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float())  # FP32 matmul
        # ...
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # back to FP16
```

> **重要**：dynamo ONNX 导出时如果把这些 `.to(torch.float32)` → `.to(input_dtype)` 的 Cast 操作正确保留，onnxruntime 应该能跟 PyTorch 对齐。如果 dynamo 把 Cast 操作 trace 掉了，onnxruntime 全用 FP16 计算就会导致差异。

#### LLM ONNX 的 Cast 节点实测（llm_add_ds.onnx）

对 `llm_engine/llm_add_ds.onnx`（2528 节点）的全节点 Cast 分析：

| 项 | 数量 | 说明 |
|----|------|------|
| 总 Cast 节点 | **216** | 108 对 |
| F16→F32 | **108** | 每个 RMSNorm 入口（FP16 输入升 FP32） |
| F32→F16 | **108** | 每个 RMSNorm 出口（FP32 结果降回 FP16） |
| 完整 RMSNorm FP32 链 | **108** | `Cast→Pow→ReduceMean→Add→Sqrt/Reciprocal→Mul→Cast`（= 27 层 × 4 个 RMSNorm/层：input_layernorm + post_attention_layernorm + q_norm + k_norm） |

**关键结论**：
- **LLM ONNX 里所有 216 个 Cast 全部属于 RMSNorm 链**——没有其他算子（SDPA/Linear/MLP）有 Cast
- **rotary_emb 在 LLM ONNX 里没有 Cast**：因为 LLM 的 cos/sin 是**外部输入**（打桩截取的 `llm_pt_reference_with_ds.safetensors` 原生 FP32 存储；TRT engine 输入为 FP16，推理时 cast），`inv_freq @ position_ids` 的 FP32 计算发生在图外（推理时算好喂给 engine），不在 ONNX 图内——与 ViT 不同（ViT 的 rotary 在 `apply_rotary_pos_emb_vision` 内部，有 96 个 Cast，见 5.12）
- 这 216 个 Cast 正是 task1b 里 `--layerPrecisions` 强制 FP32 的 648 个节点的来源（108 链 × 6 算子 = 648）

### 6.3 hidden_states[-2] 的真实含义

- `capture_outputs` 装饰器的 `tie_last_hidden_states=True` 会把 `hidden_states[-1]` 替换为 `last_hidden_state`（经过 final norm 后的值）
- 所以 `hidden_states[-2]` 实际是 **layer 26 的输出**（norm 前），不是 layer 27

## 七、方案调研记录

### 7.1 方案A：LLM ONNX 导出 + trtexec 转 engine（已完成）

> 完整排查过程见 `task1b_llm_trt_precision_debug.md`（时间线：二分定位 → mask 假设排除 → RMSNorm FP16 累积根因 → 数据重截取 → TRT 兼容修复）。

**路线**：wrapper 导出 ONNX（dynamo=True）→ trtexec 构建 engine → PyTorch/真实链路验证。

**最终 wrapper 设计（export_llm_onnx.py，四 wrapper）**：复现 `Qwen3VLTextModel.forward`（line 856-940），只跑前 27 层（VLA 取 `hidden_states[-2]` = layer26 输出）：

| wrapper | deepstack | causal mask | 用途 |
|---------|-----------|-------------|------|
| manual_no_ds | 无 | None（SDPA 内部 is_causal） | 对照 |
| official_no_ds | 无 | create_causal_mask 显式 4D | 对照 |
| official_with_ds | 官方 index 赋值 | create_causal_mask 显式 4D | 参考（TRT 不兼容） |
| **add_ds** | **图外展开 ds_full + 图内纯 Add** | create_causal_mask 显式 4D | **最终方案** |

add_ds 说明：官方 index 赋值 trace 出 NonZero+GatherND+ScatterND，TRT 10.11 对数据依赖 shape 推导崩（288≠163）。改为图外把 ds [288,2048] 按 visual_pos_masks 展开成 [1,seq,2048] dense（非 mask 位置为 0），图内 `h = h + ds_full` 纯 Add——**与官方 index 赋值 bit 级等价（max_diff=0.0000）**。

**输入数据**：VLA 真实链路打桩截取（`CAPTURE_TAG` 触发，`DISABLE_DEEPSTACK=1` 出 no_ds 版），17 个 tensor 含正确 M-RoPE 的 cos/sin。

**导出命令**（纯 FP32 真实链路，2026-08-28 起）：
```bash
# 在 whh_vla_trt2 容器中（TRT 10.11）
cd /root/workspace/embodichain/hpc_opt/trtllm
# 纯 FP32：模型 FP32（sdpa attention）+ 输入输出 FP32（与 VLA 真实推理一致）
python3 export_llm_onnx.py --wrapper add_ds --export      # 导出 llm_add_ds.onnx（FP32 权重 + FP32 IO）
# 构建 engine（fp32 IO + fp16 内部 + RMSNorm FP32，2026-08-28 起 IO 统一 fp32 对齐真实链路）
python3 build_llm_engine.py --onnx llm_add_ds.onnx --fp32-layers
# → 产物 llm_engine/fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan
python3 verify_engine.py --engine llm_engine/fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan
# ── 以下为旧 FP16 IO engine 流程（FP16 ONNX 时代，已废弃；仅历史参考）──
# python3 gen_fp32_layers.py --onnx llm_engine/llm_add_ds.onnx   # 生成 RMSNorm FP32 约束（旧 FP16 ONNX 用）
# python3 build_llm_engine.py --onnx llm_add_ds.onnx --fp32-layers  # 旧版（IO 是 fp16）
# python3 verify_engine.py --engine llm_engine/fp16_fp32_layers/llm_add_ds_fp16_fp32_layers.plan
```

**关键坑（详见 task1b 4.x）**：
- ⚠ 旧 7 输入版（含 BOOL visual_pos_masks）不能用 `--inputIOFormats=fp16:chw` 广播——BOOL 被强制 FP16 报 "condition tensor must have boolean type"。add_ds 版 6 输入全 FP16 无此问题
- ⚠ layerPrecisions 必须合并成逗号分隔单参数；名字必须用 TRT 层名 `node_{op}_{idx}`（ONNX tensor 名匹配率 0，静默失效）
- ⚠ RMSNorm 的 648 个节点（Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul 各 108）必须强制 FP32，否则 27 层 FP16 归约累积 106 ULP

**最终精度**（vs 真实链路 target，abs_max≈14750，2026-08-28 新数据，verify_engine.py 三 engine 实测 + benchmark）：
- **FP32 wrapper vs 真实链路 = 0.0000**（cos/sin/target 原生 FP32 + sdpa attention 后逐位一致）
- **TRT engine（fp32 IO + fp16 内部 + fp32 RMSNorm）vs 真实链路 = 13.6（1.7 ULP，0.09%），耗时 5.03 ms/iter**——当前部署方案
- 对照：无 RMSNorm 约束的 fp16engine = **104.4 ULP / 4.98 ms**（RMSNorm FP16 归约累积根因仍存在，但约束零性能代价）；全 FP32 engine = **0.1 ULP / 10.06 ms**（5.4GB + 2.02x 慢）
- FP16 wrapper vs 真实链路 = 5.62（纯 FP16 推理误差参考值）
- 纯 FP16 为 860（107 ULP），改善 98.4%（107 ULP → 1.7 ULP）

**统计口径说明**（为什么 TRT 输出 mean 与 target 几乎一致，但 max_diff 有 13.6）：

| 统计量 | 值 | 含义 |
|--------|-----|------|
| 均值差（output.mean - target.mean） | +0.0003 | 净偏移 ≈ 0 → **无系统性偏置**（引擎没有整体漂移） |
| mean_diff（\|output - target\| 逐元素平均） | 0.2546 | 平均绝对误差，相对峰值 14750 = **0.0017%** |
| max_diff（最大单点绝对差） | 13.6 | 1.7 ULP（FP16 在峰值处的 ULP=8） |

原因：hidden_states 是 316×2048 = 64.7 万元素，分布在 [-14750, +14750]，误差**正负对称分布**——求均值时正负抵消（净偏移≈0），但每个元素的绝对误差都贡献 ~0.25。**均值差反映系统性偏置（引擎是否漂移），mean_diff/max_diff 反映真实误差幅度（是否对齐）**。类比：一屋子人平均身高差 1mm，不代表每个人身高接近——可能有人高 30cm、有人矮 29cm，均值抵消了。引擎与真实链路的"对齐"要看 mean_diff/max_diff，不是看输出均值。

### 7.2 方案B：TRT-LLM LLM() API / GenerationSession（已排除）

| 检查点 | 结果 | 说明 |
|--------|------|------|
| LLM() API `generate()` | ❌ 无 output_hidden_states | 只输出 token ids |
| GenerationSession `hidden_states_output` | ⚠️ 仅 PP 场景 | pipeline parallelism 用，输出最后一层不是 -2 层 |
| `gather_context_logits=True` | ⚠️ 输出 logits 不是 hidden_states | logits = hidden_states[-1] → norm → LM_head，不可逆 |
| engine_inspector | ⚠️ 只读 | 能查看 engine 内部信息但不能修改输出 |

**结论**：GenerationSession 不支持提取 hidden_states[-2]。

**但可通过修改 build + runtime 两端实现**（即方案C）：在 trtllm-build 时于 layer 26 输出处 `network.mark_output()`，在 GenerationSession 读取该 output tensor。

### 7.3 方案C：修改 TRT-LLM build + runtime 两端（待尝试）

核心思路：
1. **build 端**：在 `trtllm-build` 构建过程中，于 layer 26 的输出处 `network.mark_output()`，标记为 engine output
2. **runtime 端**：修改 GenerationSession 读取这个新 output tensor 并返回

好处：能走完整的 TRT-LLM build 线路，engine 里 28 层都有 `gpt_attention_plugin` 优化（fused attention kernel、KV cache 等），比 trtexec 裸导出可能快不少。

需要改的代码路径：
```
trtllm-build
  → tensorrt_llm/models/qwen/ (或 qwen3_vl 相关)
  → builder 里循环创建 28 层 decoder layer
  → 在第 26 层（index 26，即倒数第二层）的输出处加：
     network.mark_output(layer_output, name="hidden_states_neg2")
  → 重新 trtllm-build 生成 engine

GenerationSession
  → decode_regular 里加：
     if hasattr(self, 'hidden_states_neg2'):
         outputs['hidden_states_neg2'] = self.hidden_states_neg2
  → setup 时分配这个 output tensor 的地址
```

---

## 八、文件清单

### 已有文件（容器内，当前状态）

| 文件 | 路径 | 说明 |
|------|------|------|
| export_vit_384.py | hpc_opt/trtllm/vit/ | ViT 384x384 ONNX 导出（4 输出，checkpoint-46000，monkey-patch） |
| build_vit_384_engine.py | hpc_opt/trtllm/vit/ | ONNX → TRT engine（支持 --fp32，TRT 10 API 修复） |
| verify_vit_wrapper.py | hpc_opt/trtllm/vit/ | ViT wrapper vs PyTorch 验证（逐 bit 对齐） |
| verify_vit_engine.py | hpc_opt/trtllm/vit/ | ViT TRT engine vs PyTorch 验证（FP16+FP32） |
| llm_wrapper.py | hpc_opt/trtllm/ | LLM 共享模块（4 wrapper + WRAPPERS + NUM_LAYERS + load_model(torch_dtype, attn_implementation)） |
| export_llm_onnx.py | hpc_opt/trtllm/ | LLM ONNX 导出（四 wrapper，**纯 FP32 真实链路**：模型 FP32 + sdpa + 输入输出 FP32，复用 llm_wrapper） |
| build_llm_engine.py | hpc_opt/trtllm/ | LLM engine 构建（FP16 / FP32 / FP16+fp32_layers，自动读 ONNX 输入） |
| gen_fp32_layers.py | hpc_opt/trtllm/ | 从 ONNX 生成 RMSNorm 的 layerPrecisions（TRT 层名格式） |
| verify_engine.py | hpc_opt/trtllm/ | 通用 TRT engine vs 真实链路 target 验证 |
| verify_where_ds.py | hpc_opt/trtllm/ | where/add 版 deepstack vs 官方 index 赋值等价验证（--dtype fp16/fp32/both） |
| verify_reference_full.py | hpc_opt/trtllm/ | 截取数据内部一致性（A1-A5）+ 三方一致性（wrapper/engine/target，FP32） |
| gqa/verify_gqa*.py / verify_sdpa_onnx*.py | hpc_opt/trtllm/ | GQA 路径对比（GPU/CPU，dynamo vs legacy） |
| gqa/sdpa_gqa_dynamo*.onnx | hpc_opt/trtllm/ | dynamo 导出的 GQA SDPA ONNX（enable_gqa / repeat_kv） |
| vit/vit_engine_384/vit_qwen3vl_384_simplified.onnx | hpc_opt/trtllm/ | 775MB 精简 ONNX |
| vit/vit_engine_384/vit_qwen3vl_384.plan | hpc_opt/trtllm/ | 783MB FP16 ViT engine |
| vit/vit_engine_384/vit_qwen3vl_384_fp32.plan | hpc_opt/trtllm/ | FP32 ViT engine |
| vit/vit_engine_384/rotary_inner_fp32_layers.txt | hpc_opt/trtllm/ | rotary 内部 336 个 FP32 算子约束表 |
| vit/vit_engine_384/vit_qwen3vl_384_fp16_rotaryinner_fp32.plan | hpc_opt/trtllm/ | FP16 + rotary 内部 FP32 的 ViT engine |
| llm_engine/llm_add_ds.onnx (+ .data) | hpc_opt/trtllm/ | **LLM 最终 ONNX**（FP32 权重 + FP32 IO，6 输入全 FP32，2026-08-28 起） |
| llm_engine/llm_add_ds_fp32_layers.txt | hpc_opt/trtllm/ | LLM RMSNorm layerPrecisions（TRT 层名格式，648 节点） |
| llm_engine/fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan | hpc_opt/trtllm/ | **LLM 最终 engine**（2.7GB，fp32 IO + fp16 内部 + RMSNorm FP32，vs 真实链路 1.7 ULP） |
| llm_engine/fp32onnx_fp32engine/llm_add_ds_fp32.plan | hpc_opt/trtllm/ | 全 FP32 engine（5.4GB，0.1 ULP，慢 2.6x，备选） |
| llm_engine/llm_pt_reference_{no_ds,with_ds}.safetensors | hpc_opt/trtllm/ | VLA 真实链路截取参考数据（17 tensor，正确 M-RoPE） |
| llm_engine/keep/ | hpc_opt/trtllm/ | 排查证据（layerinfo json ×2 + hs.pt ×2） |

> 已清理：debug_engine/（62GB 排查产物）、旧 llm_qwen3vl_dynamo*.onnx、旧 fp16/fp32 engine、verify_llm_wrapper.py、verify_llm_engine.py、debug_llm_engine.py。需要复现排查过程见 task1b。

### TRT-LLM 参考文件

| 文件 | 路径 | 说明 |
|------|------|------|
| export_vit_onnx.py | trtllm-qwen3-vl/ | 1920x1080 ViT 导出参考（monkey-patch + 1 输出） |
| build_trt_engine.py | trtllm-qwen3-vl/ | ViT engine 构建参考 |
| serve.py | trtllm-qwen3-vl/ | TRT-LLM 双 engine 推理 pipeline 参考 |
| extract_qwen3_llm.py | trtllm-qwen3-vl/ | LLM 权重提取 |
| convert_and_build.py | trtllm-qwen3-vl/ | LLM convert_checkpoint + trtllm-build |

---

## 九、关键技术问题

### 9.1 GQA 不支持 ONNX 导出（legacy）→ dynamo 解决

PyTorch ONNX exporter 有两种模式：

- **legacy（dynamo=False, jit trace）**：不支持 `scaled_dot_product_attention` 的 `enable_gqa` 参数，Q/KV head 数不匹配时直接报错"conversion of SDPA not implemented if enable_gqa is True"
- **dynamo（dynamo=True）**：能正确导出 GQA，不需要 monkey-patch，不需要 repeat_kv。dynamo 内部把 K/V Expand 到 16 heads 再做标准 attention，导出的 ONNX 算子与 repeat_kv 版本完全一致（见 9.2）

之前通过 monkey-patch + repeat_kv 绕过 GQA 问题的方式已废弃，改用 dynamo=True 直接导出。

### 9.2 GQA 两种路径分析

Qwen3-VL-2B LLM 使用 GQA（Q=16 heads, K/V=8 heads, groups=2）。SDPA 内部有两条处理路径：

**路径1: enable_gqa=True（GQA kernel）**
- `use_gqa_in_sdpa` 返回 True（条件：torch>=2.5 且 attention_mask=None）
- K/V 保持 8 heads，kernel 内部按 group 映射，每 2 个 Q head 共享 1 个 K/V head
- 不做数据搬运，省显存带宽

**路径2: repeat_kv 后标准 attention**
- `use_gqa_in_sdpa` 返回 False
- 先把 K/V 从 8 heads repeat_interleave 到 16 heads
- 走标准 16×16 attention（MatMul + Softmax + MatMul）

验证脚本：`hpc_opt/trtllm/gqa/verify_gqa.py`（GPU）+ `hpc_opt/trtllm/gqa/verify_gqa_cpu.py`（CPU）

#### SDPA backend 分析

问题：PyTorch 底层是显式调用 FlashAttention 吗？

回答：不是显式调用。PyTorch SDPA 是一个 dispatch 层，运行时根据条件自动选择 backend：

```
F.scaled_dot_product_attention(q, k, v, ...)
    → C++ 层 sdp_utils.cpp 检查条件：
        - dtype 是 FP16/BF16？
        - attention_mask 是 None？
        - enable_gqa=True 且 Q/KV head 数不同？
        - 序列长度？
        → 选择最优 backend（FlashAttention 优先）
    → 调用对应的 CUDA kernel
```

Python 层只调了 `F.scaled_dot_product_attention`，不直接调 FlashAttention。是 C++ runtime 在运行时自动路由的。

SDPA 有 4 个 backend，优先级：FlashAttention > cuDNN > memory_efficient > math

| backend | enable_gqa 支持？ | 说明 |
|---------|------------------|------|
| FlashAttention | ✅ | 支持 GQA，内部按 group 映射，不 repeat |
| cuDNN | ✅ | 支持 GQA |
| math | ✅ | 支持 GQA（手动 softmax） |
| memory_efficient | ❌ | 不支持，Q/KV head 数不匹配时报错 |

本环境（RTX 5090, PyTorch 2.7，whh_vla_trt2）默认走 FlashAttention backend。

`enable_gqa=True` 时，FlashAttention kernel 内部按 group 关系处理——每 2 个 Q head 共享 1 个 K/V head，通过 index 映射访问 K/V 数据，没有额外的数据拷贝。

不管 `enable_gqa` 是 True 还是 False，FP16 下底层走的都是同一个 FlashAttention kernel。区别只在输入数据的 shape（8 heads vs 16 heads），计算完全等价。

#### enable_gqa vs repeat_kv 数值对比

问题：enable_gqa 和 repeat_kv 两种路径是否存在误差？

##### GPU 对比（verify_gqa.py，torch 2.7 + RTX 5090）

| 精度 | backend | path1 vs path2 max_diff | 说明 |
|------|---------|--------------------------|------|
| FP16 | 默认（自动选择） | **0.0** | 两条路径都走 FlashAttention |
| FP16 | FlashAttention | **0.0** | 完全一致 |
| FP16 | cuDNN | **0.0** | 完全一致 |
| FP16 | math | **0.0** | 完全一致 |
| FP32 | 默认（自动选择） | **0.443** | path1 走 math，path2 走 mem_efficient |
| FP32 | math（强制） | **0.0** | 完全一致 |

##### FP32 下 path1 vs path2 差异根因

**根因是 torch 2.7 对两条路径分配了不同的 SDPA backend，不是算法有差异。**

torch 2.7 在 RTX 5090 上的 backend 选择逻辑：

| 路径 | FP16 | FP32 |
|------|------|------|
| path1（enable_gqa=True, Q=16h, KV=8h） | FlashAttention（原生支持 GQA） | **math**（flash 不支持 FP32 + GQA → fallback） |
| path2（repeat_kv, Q=16h, KV=16h） | FlashAttention（标准模式） | **mem_efficient**（不需要 GQA，支持 FP32） |

- `path1 [flash]` FP32 下失败（flash 只支持 FP16/BFloat16）
- `path1 [math] vs 默认: max_diff=0.0`（path1 默认实际走的就是 math）
- `path2 [mem_efficient] vs 默认: max_diff=0.119`（path2 默认走 mem_efficient，不是 math）
- **`[math] path1 vs path2: max_diff=0.0`**（强制都走 math，结果逐 bit 一致）

结论：path1 和 path2 的算法完全等价。FP32 下的 max_diff=0.443 纯粹来自 math backend 和 mem_efficient backend 的 FP32 累加顺序不同（math 逐行 softmax，mem_efficient tiled 分块），不是算法差异。

##### CPU 对比（verify_gqa_cpu.py，torch 2.7）

| 精度 | max_diff | 说明 |
|------|---------|------|
| FP16 | **0.001953** | path1（group 映射）和 path2（展开后标准 attention）走不同代码路径，FP16 softmax 累加顺序不同导致舍入差异 |
| FP32 | **0.000002** | 几乎一致（ULP 级别），算法完全等价 |

CPU SDPA 没有 FlashAttention，两条路径走的是不同代码路径：path1 内部按 group 映射访问 K/V，path2 先复制到 16 heads 再走标准 attention。FP16 下累加精度不够导致 0.002 的差异，FP32 下精度够高所以只有 ULP 级别差异。

> 注：whh_vla_trt2（torch 2.7）CPU FP16 下有 0.002 的差异，whh_trtllm_r130r20（torch 2.11）CPU FP16 下完全一致（max_diff=0.0）。两个 torch 版本的 CPU SDPA 实现不同。

##### GPU vs CPU 对比

| 环境 | FP16 | FP32 |
|------|------|------|
| GPU（FlashAttention kernel） | max_diff=0.0 ✅ | max_diff=0.443 ❌（backend 不同） |
| CPU（math kernel） | max_diff=0.002 ❌ | max_diff=0.000002 ✅ |

GPU FP16 比 CPU FP16 更一致——因为 GPU 两条路径都走同一个 FlashAttention kernel，而 CPU 走的是不同代码路径。GPU FP32 差异更大（0.443），原因是两条路径被分配了不同 backend（math vs mem_efficient），而非算法差异。

#### 性能对比（FlashAttention, 1000 iter, seq_len=301）

> 注：seq_len=301 是早期导出脚本的 add_generation_prompt=False 数据；VLA 真实链路打桩数据为 seq_len=316（含 288 visual token + 文本）。该对比是 GQA 路径的相对性能，不受 seq_len 具体值影响。

| 方式 | 耗时(ms/iter) | 比值 |
|------|-------------|------|
| enable_gqa=True（8 heads K/V） | 0.0109 | 1.0x |
| repeat_kv（16 heads K/V） | 0.0210 | 1.94x |

repeat_kv 慢约 2x——多了一步 expand + reshape 的数据搬运开销（16 heads K/V 的显存读写是 8 heads 的两倍）。多次测量比值稳定在 1.9-2.6x。

#### ONNX 导出时怎么找对应算子

问题：底层是 C++ 实现，torch 导出时是怎么找对应算子的？

回答：`torch.onnx.export`（jit trace）不是"找对应算子"，而是实际跑一遍模型，把所有 PyTorch 操作记录成 ONNX 算子。但 SDPA 是 C++ builtin，trace 不进内部实现——PyTorch 注册了一个 ONNX fallback 把 SDPA 拆成 Sub → Mul → MatMul → Add → Softmax → MatMul。这个 fallback 不认识 enable_gqa，Q/KV head 数不匹配时报错。

之前用的是 `torch.onnx.export`（legacy jit trace, `dynamo=False`），不是 dynamo_export。

#### ONNX dynamo vs legacy 导出对比

问题：用 dynamo=True 能否正确导出 GQA？

验证脚本：
- GPU：`hpc_opt/trtllm/gqa/verify_sdpa_onnx.py`（PyTorch CUDA + onnxruntime CPU）
- CPU：`hpc_opt/trtllm/gqa/verify_sdpa_onnx_cpu.py`（PyTorch CPU + onnxruntime CPU，排除 GPU kernel 影响）

只对比 SDPA 这个算子（不涉及完整模型），参数与 Qwen3-VL-2B LLM 一致。

##### GPU 测试结果（verify_sdpa_onnx.py）

| 方式 | 导出 | onnxruntime 推理 | vs PyTorch enable_gqa |
|------|------|-----------------|---------------------|
| PyTorch enable_gqa=True | — | — | 基准 |
| PyTorch repeat_kv | — | — | max_diff=0.0 ✅ |
| ONNX dynamo=True (enable_gqa) | ✅ 成功 | ✅ 成功 | max_diff=0.001 ✅ |
| ONNX dynamo=False (legacy) | ❌ 失败 | — | "conversion of SDPA not implemented if enable_gqa is True" |
| ONNX dynamo=True + repeat_kv | ✅ 成功 | ✅ 成功 | max_diff=0.001 ✅ |

##### CPU 测试结果（verify_sdpa_onnx_cpu.py）

问题：PyTorch 跑在 CPU 上，结果是什么？

| 对比项 | max_diff | mean_diff | 说明 |
|--------|---------|-----------|------|
| [2] PT repeat_kv vs [1] PT enable_gqa (FP16) | 0.0 | 0.0 | 完全一致 |
| [3] ONNX enable_gqa vs [1] PT enable_gqa (FP16) | 0.001953 | 0.000059 | onnxruntime vs PyTorch FP16 算术精度差异 |
| [4] ONNX repeat_kv vs [1] PT enable_gqa (FP16) | 0.001953 | 0.000059 | 同上 |
| [3] ONNX enable_gqa vs [4] ONNX repeat_kv (FP16) | 0.0 | 0.0 | 两个 ONNX 完全一致 |
| [2] PT repeat_kv vs [1] PT enable_gqa (FP32) | 0.0 | 0.0 | 完全一致 |

关键结论：
1. PyTorch CPU 上 enable_gqa 和 repeat_kv 完全一致（FP16 和 FP32 都是 max_diff=0.0）
2. ONNX dynamo 导出的两个版本也完全一致（max_diff=0.0）
3. ONNX vs PyTorch 有 max_diff=0.001953 的差异——这是 onnxruntime 和 PyTorch 对 FP16 Softmax/MatMul 的实现差异（不同库的 FP16 算术精度），不是 GQA 问题
4. **dynamo export 能正确导出 GQA**（enable_gqa=True），旧版 legacy export 直接报错

##### FP32 对比说明

ONNX IO 被 dynamo 固定成了 FP16，无法喂 FP32 输入做纯 FP32 对比。FP32 对比时输入被降精度到 FP16（`q32.half().numpy()`），导致 max_diff=5.13——这是 FP32 PyTorch vs FP16 onnxruntime 的对比，不是同等精度对比，没有参考价值。如需纯 FP32 对比，需重新导出 FP32 版本的 ONNX（模型权重和 IO 都用 FP32）。

#### 两个 ONNX 的算子对比

问题：dynamo 导出的 enable_gqa=True 和 repeat_kv 两个 ONNX 从算子上看有什么区别？

两个 ONNX 的算子类型和数量**完全相同**（都是 18 节点）：

| 算子 | enable_gqa=True | repeat_kv |
|------|----------------|-----------|
| Add | 1 | 1 |
| Cast | 1 | 1 |
| Equal | 1 | 1 |
| Expand | 2 | 2 |
| MatMul | 2 | 2 |
| Mul | 2 | 2 |
| Reshape | 3 | 3 |
| Softmax | 1 | 1 |
| Transpose | 1 | 1 |
| Trilu | 1 | 1 |
| Unsqueeze | 2 | 2 |
| Where | 1 | 1 |
| **节点总数** | **18** | **18** |

两个版本都有 Expand 操作（K/V 从 8 heads 扩展到 16 heads）。说明 dynamo 对 `enable_gqa=True` 的处理方式跟 repeat_kv 一样——都是把 K/V Expand 到 16 heads 再做标准 attention。

| 项 | dynamo enable_gqa=True | dynamo repeat_kv |
|----|----------------------|-----------------|
| MD5 | a5e72c... | 4562d1... |
| 大小 | 372642 bytes | 372678 bytes |

差异只在文件大小（差 36 bytes，来自 tensor name 长度不同，如 `val_35` vs `unsqueeze`）和节点排列顺序。但连线关系（哪个 tensor 连到哪个节点）完全一致，数据流图拓扑等价。

**结论：两个 ONNX 算子完全相同，连线完全一致，拓扑等价。** 推荐用 `dynamo=True` + `enable_gqa=True`——不需要 monkey-patch，不需要 repeat_kv，直接导出就行。

#### 对 ONNX 导出的影响（总结）

**GQA 导出问题已解决：**
- 旧版 `torch.onnx.export`（dynamo=False, legacy jit trace）不支持 enable_gqa，Q/KV head 数不匹配时报错
- 旧版通过 monkey-patch + repeat_kv 绕过 GQA 问题，但有两种失败情况：
  - patch `enable_gqa` 去掉 + `attention_mask=None`：SDPA 报维度不匹配（16 vs 8），导出失败
  - patch `use_gqa_in_sdpa=False` 走 repeat_kv：ONNX 导出"成功"，但 onnxruntime 加载失败（`/Add_1 Incompatible dimensions`）
- 更早之前用显式 causal_mask（未 patch，自然走 repeat_kv）导出的旧 ONNX：TRT 能加载但数值全错（max_diff=14928）
- **新版 `torch.onnx.export`（dynamo=True）能正确导出 GQA**，不需要 monkey-patch，不需要 repeat_kv
- dynamo 导出的 ONNX 校验通过，onnxruntime CPU 推理成功，vs PyTorch CPU max_diff=0.001953（FP16 算术精度差异）

### 9.3 grid_thw 被烘焙进 engine

grid_thw 在 ONNX trace 时被 `grid_thw.tolist()` 转为常量。更换图像数量/分辨率需重新导出。

### 9.4 TRT 10 API 适配

- `trt.Flag.EXPLICIT_BATCH` 不存在 → 用 `1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)`
- `trt.DataType.FLOAT16` → `trt.DataType.HALF`
- `context.set_tensor_address(name, ptr)` 设置输入输出
- `context.execute_async_v3(stream)` 异步执行
- `create_causal_mask` 返回 None 时表示用 SDPA 内置 is_causal，不要手动构造显式 mask
- trtexec 路径：whh_trtllm_r130r20 为 `/usr/local/tensorrt/bin/trtexec`；whh_vla_trt2 为 `/opt/lib/tensorrt/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/bin/trtexec`
- TRT engine 跨容器不兼容：TRT 10.15（whh_trtllm_r130r20）构建的 engine 不能在 TRT 10.11（whh_vla_trt2）加载——**engine 必须用目标容器同版本 TRT 构建**

### 9.5 VLA 推理精度的实现机制（2026-08-29 追查确认）

**现象**：config 里 encoder 写 `dtype: bf16`、checkpoint 权重也是 bf16，但打桩实测 VLM 推理是 FP32（625 参数全 `torch.float32`）。转换发生在哪？

**bf16→fp32 的精确链路**：
1. **读取**：`safetensors.torch.load_model` → `load_state_dict`（checkpoint bf16 读入覆盖模型参数——此时 VLM 仍是 bf16，实测 `{torch.bfloat16: 625}`）
2. **转换**：真实链路 `model = model.to(device)` 触发 `DexForceVLA.to()` 重写（dexforcevla_runner.py L485-493）——`dtype = self.get_dtype()`（=fp32，来自构造时 `from_pretrained(dtype=fp32)` 传入），对 `self.encoders.values()` 逐个 `module.to(device, dtype=fp32)` → VLM bf16 → fp32
3. **无损**：bf16 尾数 7 bit ⊂ fp32 23 bit，实测 10000 个随机 bf16 值 `bf16→fp32→bf16` 往返 bit 级一致；且是升精度不可能丢信息

**对导出的影响**：export_llm_onnx.py 按此原则导出纯 FP32 ONNX（FP32 权重 + FP32 输入输出 + sdpa attention），与真实链路逐位一致（0.0000）。TRT engine 用 fp32 IO + fp16 内部 + RMSNorm FP32，边界对齐真实链路。

**关键教训**：config 的 encoder `dtype: bf16` 是**初始加载 dtype**，真正决定推理 dtype 的是 `model.to()` 传导——分析精度时不能只看 config，要看运行时实测（打桩打印 VLM 参数 dtype 是最可靠的）。

---

## 十、环境

**whh_trtllm_r130r20**（ViT engine 构建/验证）：

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA RTX 5090 D v2 (24 GB) |
| Python | 3.12.3 |
| PyTorch | 2.11.0 (NVIDIA build) |
| TensorRT | 10.15.1.29 |
| TensorRT-LLM | 1.3.0rc20 |
| Transformers | 5.5.4 |
| ONNX | 1.22.0 |
| onnxsim | 0.6.5 |
| numpy | 2.1.0（与 onnxruntime 1.27.0 兼容） |

**whh_vla_trt2**（LLM engine 构建/部署/TRT hook）：

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| TensorRT | 10.11.0.33 |
| Transformers | 5.5.4 |
| ONNX | 1.22.0 |
| onnxruntime | 1.19.2 (CPU-only) |
| numpy | 1.26.4 |

⚠️ **VLA 推理精度：FP32**（config `mixed_precision: "no"`，非全流程 FP16——早期文档表述有误已修正）。**存储 dtype ≠ 推理 dtype**：checkpoint VLM 权重存 bf16（加载后仍 bf16），真实链路 `model.to()` 触发 DexForceVLA.to() 重写对 encoders 调 `.to(fp32)` 无损升精度（转换点详见 task1b 3.7），推理为"bf16 存储精度 + FP32 计算精度"。TRT engine 为 FP16，hook 输入自动 cast。

LD_LIBRARY_PATH 设置：
```bash
# whh_trtllm_r130r20
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# whh_vla_trt2
export LD_LIBRARY_PATH=/opt/lib/tensorrt/TensorRT-10.11.0.33/lib:$LD_LIBRARY_PATH
```

---

## 十一、与 1920x1080 版本的对比

| 属性 | 384x384 (本方案) | 1920x1080 (参考) |
|------|------------------|-------------------|
| 图像数 | 2 | 1 |
| pixel_values | [1152, 1536] | [2040, 1536] |
| ViT 输出 | 4 (pooler + 3× deepstack) | 1 (pooler only) |
| ONNX 大小 | 775 MB | 679 MB |
| Engine 大小 | 783 MB | 730 MB |
| 用途 | VLA 双相机 | 单图理解 |
