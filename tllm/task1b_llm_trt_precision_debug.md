# LLM TRT Engine 精度对齐问题排查报告

> 记录方案A（LLM ONNX 导出 + trtexec 转 engine）的完整排查时间线。
> 按时间顺序：每个阶段 = 现象 → 实验 → 结论（含被排除的假设）。
> 最终结论见第 5 节，踩坑合集见第 4 节。

---

## 1. 问题概述

### 1.1 现象

LLM（Qwen3-VL-2B 的 language_model，27 层 decoder + deepstack 注入）导出 ONNX 后构建 TRT engine，**TRT 推理输出与 PyTorch 参考数据严重偏离（max_diff≈14744）**，而 **onnxruntime 推理同一 ONNX 正常（max_diff=8）**。

### 1.2 环境

| 项 | 值 |
|---|---|
| 容器 | whh_vla_trt2 |
| torch | 2.7.0+cu128 |
| TensorRT | 10.11.0.33 |
| transformers | 5.5.4 |
| onnxruntime | 1.19.2 (CPU-only) |
| GPU | NVIDIA GeForce RTX 5090 D v2 |
| 模型 | checkpoint-46000（ViT 315 + LLM 310 = 625 个权重） |
| 数据 | VLA 真实链路截取（hdf5 真实相机帧 + 真实指令），seq_len=316 |

### 1.3 初始参考数据（已作废，仅作为排查起点记录）

- 旧 safetensors（llm_pt_reference_dynamo）：`orig_hidden_neg2` [1,316,2048] FP16，mean=0.268，abs_max=14744；position_ids 是 arange（非 M-RoPE），2026-08-26 已重新截取（见 3.7）
- TRT FP16 / FP32 / FP16+fp32_layers 三个 engine 输出全错（vs PT max_diff≈14744），ort 正常（8.0）——关键观察：onnxruntime 正确 → ONNX graph 计算逻辑没问题，问题在 TRT 构建/推理环节

---

## 2. 排查思路

1. **二分逐层定位**：分段导出（1/2/3/.../27 层）→ 找第一个出错层
2. **子模块分解**：对出错层拆成 norm/qkv/rotary/attn 单独测
3. **手动 forward vs 官方 forward**：定位 dynamo trace 的 graph 差异
4. **假设验证**：每次只改一个变量
5. **基准教训**（重要）：早期用 onnxruntime 做基准，ort 始终 8~16（= 1-2 ULP）无法区分"graph 对"和"TRT 错"；后来改为 **PyTorch forward（GPU FP16）为唯一基准**（见 3.5）

> ⚠️ 澄清：这里的"FP16 基准"是**排查策略**——TRT engine 是 FP16，用同精度 FP16 PyTorch 对照才能隔离"TRT 实现差异"（避免混入 FP16 vs FP32 精度差）。**它不是 VLA 真实链路**（真实链路是 FP32，见 9.5：model.to() 传导 + 打桩实测 `{torch.float32: 625}`）。真实链路用 FP32 target 对比（3.7/3.8.3）。

---

## 3. 排查时间线

### 3.1 分段二分：误差从第 1 层开始

将 wrapper 分段导出前 N 层 ONNX（dynamo=True，动态 seq_len），构建 FP16 engine，对比 TRT / ort / PyTorch。

| 层数 | ort vs PyTorch max_diff | TRT vs PyTorch max_diff | 状态 |
|------|--------------------------|--------------------------|------|
| 1 | 0.015625 | 90.41 | ❌ |
| 2 | 0.031250 | 99.46 | ❌ |
| 3 | 16.0 | 14760 | ❌ |
| 4 | 16.0 | 14760 | ❌ |
| 6 | 16.0 | 14760 | ❌ |
| 8 | 16.0 | 14760 | ❌ |
| 13 | 8.0 | 14784 | ❌ |
| 20 | 16.0 | 14832 | ❌ |
| 27 | 8.0 | 14744 | ❌ |

**结论**：**第 1 层（layer 0）就出错**（TRT max_diff=90）。ort 全对（≤16）→ ONNX graph 逻辑没问题。误差从第 1 层开始逐层累积放大（第 3 层 deepstack 注入后跳变到 14760）。

### 3.2 layer 0 子模块分解：单独测都对，组合错

将 layer 0 拆成 4 个子模块独立导出 + 构建：

| 子模块 | 包含 | ort vs PT | TRT vs PT |
|--------|------|-----------|-----------|
| norm_only | input_layernorm | 0.002 | 0.004 ✅ |
| norm_qkv | + qkv proj + per-head q/k norm | 0.5 | 1.0（值小） |
| norm_qkv_rotary | + rotary 应用 | 0.5 | 1.75（值小） |
| norm_attn | 完整 attention（rotary+SDPA+o_proj） | 0.004 | 0.016 ✅ |

**结论**：各子模块单独测 TRT 基本正常（<2，norm_qkv 的 0.5 来自 q/k 未归一化值域大，非错误）。**完整 attention 单独测对（0.016），但完整 layer 0 错（90）——合起来就错，单独测都对。**

### 3.3 手动分解 vs 官方 forward：同样的计算，不同的 graph

手写 layer 0 完整 forward（RMSNorm+QKV+rotary+SDPA+o_proj+residual+post_norm+MLP+residual）导出多输出 ONNX，对照官方 `self.layers[0]()` 的 wrapper。

手动分解版（6 输出）：

| 输出 | ort vs PT | TRT vs PT |
|------|-----------|-----------|
| out（最终输出） | 0.016 | **0.031 ✅** |
| h2 / h3 / mlp_out / o / attn_flat | 0.002~0.008 | 0.008~0.031 ✅ |

官方 decoder_layer wrapper 版：

| 版本 | ort vs PT | TRT vs PT |
|------|-----------|-----------|
| layer1_no_ds（无 deepstack） | 0.016 | **38.80 ❌** |
| layer1_with_ds（有 deepstack） | 0.016 | **97.16 ❌** |

**结论**：**同样的计算逻辑，手写 forward 导出 TRT 正常，官方 `Qwen3VLDecoderLayer.forward()` 导出 TRT 错。** 去 deepstack 也错（38.8 vs 97.2，deepstack 只是放大误差不是根因）。问题缩小到：**dynamo trace 官方 forward 生成的 graph 与手动不同**。

### 3.4 假设 H1：causal mask 数据类型是根因（后被排除）

节点级对比（手动版 A：103 节点，官方版 B：109 节点）发现关键差异在 causal mask 路径：

| | A（手动，TRT 正常） | B（官方，TRT 错） |
|---|---|---|
| Trilu 输入/输出 dtype | **BOOL** | **FP32** |
| Where 输入 | (BOOL, FP16, FP16) | (BOOL, FP32, FP32) |
| Where 输出 | FP16 | FP32 → Cast(FP32→FP16) |
| IsNaN | 有 | 无 |
| Equal | 无 | 有 |

A 路径：`Expand(BOOL) → Trilu(BOOL) → Where(BOOL, FP16, FP16→FP16)`
B 路径：`Expand(FP32) → Trilu(FP32) → Equal(FP32→BOOL) → Where(BOOL, FP32, FP32→FP32) → Cast(FP32→FP16)`

**当时推测**：FP32 的 Where/Cast 路径在 TRT FP16 构建中处理错误，曾判定为"根因确定"。但后续验证实验全部推翻了它：

| 假设 | 验证实验 | 结果 |
|------|---------|------|
| FP32 mask dtype 是根因 | FP32 mask + IsNaN → TRT 正常 | ❌ 排除 |
| 缺少 IsNaN 是根因 | BOOL mask + 无 IsNaN → TRT 正常 | ❌ 排除 |
| FP32 mask + 无 IsNaN 组合 | 同上 → TRT 正常 | ❌ 排除 |
| 动态 shape 推导是根因 | mask_dynamic_shape → TRT 正常 | ❌ 排除 |
| arange+索引比较构造 mask | mask_arange_compare → TRT 正常 | ❌ 排除 |
| is_causal=True 的 SDPA | sdpa_is_causal_true → TRT 正常 | ❌ 排除 |
| causal mask 实现方式 | 所有 mask 变体 TRT 都正常 | ❌ 排除 |
| RMSNorm 本身有问题 | 子图单独构建 → TRT 正常 | ❌ 排除 |
| graph 拓扑差异 | 手动 3 层（318 节点）和官方 3 层（324 节点）TRT 都错 | ❌ 排除 |
| deepstack 的 torch.where 触发 | 3 层无 deepstack TRT 正常，27 层无 deepstack 也错 | ❌ 排除 |
| 未使用输入影响优化 | 完整 LLM 7 输入全被使用也出错 | ❌ 排除 |
| FP16 溢出 | abs_max 始终 < 15000，远低于 65504，无 inf/nan | ❌ 排除 |

**结论：causal mask 不是根因（H1 排除）。** 所有 mask 变体 TRT 都正常，mask 相关假设全部排除。

### 3.5 以 PyTorch forward 为基准：RMSNorm FP16 累积（根因 1）

改用 **PyTorch forward（GPU FP16）为唯一基准**，手动 N 层 wrapper（3 输入，无 deepstack，真实 checkpoint-46000 权重 + 真实输入）重新测：

| 层数 | ONNX 节点数 | PT abs_max | ort vs PT | TRT vs PT | TRT/ort | 状态 |
|------|------------|-----------|-----------|-----------|---------|------|
| 1 | 111 | 21.25 | 0.016 | 0.047 | 3.0x | ✅ |
| 2 | 211 | 47.41 | 0.063 | 0.109 | 1.75x | ✅ |
| 3 | 311 | 14760 | 16.0 | 8.0 | 0.50x | ✅ |
| 4 | 411 | 14760 | 16.0 | 8.97 | 0.56x | ✅ |
| 6 | 611 | 14760 | 16.0 | 8.84 | 0.55x | ✅ |
| 8 | 811 | 14760 | 16.0 | 16.0 | 1.00x | ✅ |
| 12 | 1211 | 14776 | 8.0 | **56.0** | **7.0x** | ❌ |
| 16 | 1611 | 14792 | 8.0 | 88.0 | 11.0x | ❌ |
| 20 | 2011 | 14832 | 16.0 | 257.3 | 16.1x | ❌ |
| 27 | 2711 | 14744 | 8.0 | 850.1 | 106.3x | ❌ |

补充 9/10/11 层确认转折点形态：

| 层数 | PT abs_max | ort vs PT | TRT vs PT | ULP (TRT) |
|------|-----------|-----------|-----------|-----------|
| 8 | 14760 | 16.0 | 16.0 | 2 |
| 9 | 14760 | 16.0 | 24.0 | 3 |
| 10 | 14768 | 8.0 | 40.0 | 5 |
| 11 | 14768 | 8.0 | 48.0 | 6 |
| 12 | 14776 | 8.0 | 56.0 | 7 |

关键观察：
1. **无 FP16 溢出**——abs_max 始终 14744~14832，远低于 65504，无 inf/nan
2. **ort vs PT 始终稳定**（8~16）——正是 FP16 在 abs_max≈14744 处的 1-2 ULP（14744 落在 [8192,16384) 区间，ULP = 2^(13-10) = 8）
3. **TRT vs PT 从 12 层开始放大**——8 层 2 ULP → 12 层 7 ULP → 27 层 106 ULP，**逐层平滑累积，无突变转折点**

**结论：误差源 1 = RMSNorm 的 ReduceMean（归约）在 FP16 下精度不足，TRT GPU 归约顺序与 PyTorch 不同，每层多引入 ~1 ULP，27 层累积 106 ULP。**

### 3.6 修复验证（旧数据，已作废）：RMSNorm FP32 方向确认

> ⚠️ 本节数字（850/8.0）基于**已作废的旧参考数据**（arange M-RoPE），仅证明"RMSNorm 强制 FP32"是正确修复方向。**最终精度以 3.8.3 为准**（新数据 + 当前 engine）。

修复方向：
1. 纯 FP32 engine（慢 2-3 倍）
2. **混合精度（采用）**：`--layerPrecisions` 强制 RMSNorm 用 FP32
3. 分段构建（每段 ≤ 8 层，段间 FP32 传递）
4. FP8/INT8 量化

用 `--layerPrecisions` 强制 648 个 RMSNorm 节点（Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul 各 108）FP32 + `--precisionConstraints=obey` 后，误差从 106 ULP 降到 1 ULP（旧数据）——**根因 1 确认：误差累积源头就是 RMSNorm 的归约**。强制 FP32 后归约在 FP32 下计算，消除累积。新数据下（3.8.3）修复效果：107 ULP → 1.7 ULP。

### 3.7 完整 LLM（有 deepstack）仍错 → 数据重截取（2026-08-26）

完整 7 输入 LLM ONNX 加 RMSNorm FP32 后仍错（14744）。此时曾**错误猜想"deepstack 的 torch.where 是独立第二误差源"**（3.4 排除表已排除 torch.where 写法本身，此猜想后被彻底证伪）。

同时发现 wrapper vs VLM 真实链路输出差 373~500，排查出两个真根因：

**根因 A：cos/sin 的 position_ids 用错（贡献 373）**
- 旧 wrapper：简单 arange 位置计算
- 真实链路：M-RoPE（`compute_3d_position_ids`，text/temporal/height/width 拆成 [3,1,s] 三维位置）
- 两者 position_ids 差 275，27 层后输出差 373

**根因 B：deepstack 注入方式 + 打桩 bug（贡献 357）**
- 旧 wrapper：`torch.where(mask, h + ds_full, h)`，且 ds_full 打桩有 bug（`_ds_full[0][mask] = _ds[0]` 把第一个 token 特征广播到全部 288 个位置）
- 官方 `_deepstack_process`：index 赋值 `h[mask, :] = h[mask, :] + ds`，ds 为原始 [288, 2048] 特征

**修复：真实链路直接打桩重截取**：
- 位置：`qwen2_5_vl.py` 的 `forward_for_dexforcevla`，在 `outputs = self.vlm(...)` 前后保存
- 触发：`CAPTURE_TAG` 环境变量（无变量零影响）；`DISABLE_DEEPSTACK=1` 出 no_ds 版
- 产出：`llm_pt_reference_no_ds.safetensors` / `llm_pt_reference_with_ds.safetensors`（各 17 tensor，含正确 M-RoPE cos/sin）
- 注意：`ds0/ds1/ds2_full` 三个 tensor 因打桩 bug 是错的（已弃用）

**⚠️ VLA 推理模型是 FP32（重要事实，verify_reference_full.py 全链路验证确认）**：
- config `grasp_book_qwen3vl2b_0601_bce_master.yaml` 的 `mixed_precision: "no"` → model_dtype=fp32
- **存储 dtype ≠ 推理 dtype**：checkpoint-46000/model.safetensors 的 VLM 625 个权重全是 **bf16**（实测 dtype 分布 {float32: 186, bfloat16: 625}，186 个 fp32 是 adaptor/cerebellum）。推理时 VLM 是 **fp32**（真实链路打桩实测 `vlm params={torch.float32: 625}`），所以推理是"bf16 存储精度 + FP32 计算精度"。存储用 bf16 省空间，推理用 fp32 保证精度
- **bf16→fp32 的精确转换点（2026-08-29 追查确认）**：
  1. **读取**：`safetensors.torch.load_model` → `load_state_dict`（checkpoint bf16 读入，覆盖到模型参数——此时 VLM 仍是 bf16，实测 `{torch.bfloat16: 625}`）
  2. **转换**：真实链路 `model = model.to(device)` 触发 `DexForceVLA.to()` 重写（dexforcevla_runner.py L485-493）——`dtype = self.get_dtype()`（=fp32，来自构造时 `from_pretrained(dtype=fp32)` 传入），对 `self.encoders.values()` 逐个 `module.to(device, dtype=fp32)` → VLM bf16 → fp32
  3. **无损**：bf16 尾数 7 bit ⊂ fp32 23 bit，实测 10000 个随机 bf16 值 `bf16→fp32→bf16` 往返 bit 级一致；且是升精度不可能丢信息
- 因此截取的 inputs_embeds/deepstack/pixel_values 均为 FP32（真实链路特性，非 bug）
- **cos/sin/orig_hidden_neg2 存储精度变更（2026-08-28）**：早期打桩对这三个 tensor 显式 `.half()` 保存（为了喂 FP16 TRT engine 同精度对比），**已改为原生 FP32 保存**（打桩代码移除 3 处 `.half()`，两个 safetensors 重新截取）。原因：FP32 链路验证时用 FP16 cos/sin 会引入 0.25 误差（cos/sin FP16 舍入被 27 层放大），改为 FP32 后 FP32 wrapper 与真实链路 **max_diff=0.0000 逐位一致**
- 验证结果（新数据）：数据内部一致（mask 数==deepstack 行数、cos/sin/position_ids 重建 0 差异）；**FP32 wrapper vs 真实链路 0.0000**；FP16 wrapper vs 真实链路 **5.62**（纯 FP16 推理误差，不再被 target 的 FP16 存储掩盖）；wrapper↔engine 1 ULP
- wrapper/engine 用 FP16 推理，与 FP32 原始链路有固有差异（FP16 精度范围内）
- **FP32 wrapper 实验佐证**（verify_where_ds.py --dtype fp32，已取代 verify_fp32_wrapper.py）：wrapper 换 FP32（fp32 权重 + fp32 输入）跑同一数据 = 与真实链路 target **max_diff 0.0000 逐位一致**——证明 wrapper 逻辑与官方实现完全一致，FP16/FP32 差异纯粹是精度档

三个 wrapper（PyTorch 层面）vs 真实链路：

| wrapper | deepstack | causal mask | vs 真实链路（旧 target FP16） | vs 真实链路（新 target FP32） |
|---------|-----------|-------------|------------------------------|------------------------------|
| manual_no_ds | 无 | None（SDPA 内部 is_causal） | 8.0（1 ULP） | 5.62 |
| official_no_ds | 无 | create_causal_mask 显式 4D | 8.0（1 ULP） | 5.62 |
| official_with_ds | 官方 _deepstack_process（index 赋值） | create_causal_mask 显式 4D | 8.0（1 ULP） | 5.62 |

> 说明：旧列（8.0）是 target 为 FP16 时的数据（1 ULP，两边 FP16 舍入部分抵消）；新列（5.62）是 cos/sin/target 改为原生 FP32 后，FP16 wrapper vs 真实 FP32 链路的纯推理误差（更真实）。三个 wrapper 互相 bit 级一致（0.0000）。"torch.where 是第二误差源"的猜想证伪——之前 373~500 的差异全是 position_ids 错误 + 打桩 bug 造成的数据问题。

wrapper 复现的都是 `Qwen3VLTextModel.forward`（modeling_qwen3_vl.py line 856-940），只跑前 27 层（VLA 取 `hidden_states[-2]` = layer26 输出），不跑第 28 层和 final norm。

### 3.8 ONNX/TRT 最终验证（新数据）

**3.8.1 官方 index 赋值 ONNX → TRT profile run 崩**

`llm_official_with_ds.onnx`（index 赋值 trace 出 NonZero+GatherND+ScatterND）构建通过但推理崩：

```
node_Add_389: Broadcast has incompatible dimensions: 288 != 163
```

根因：TRT 10.11 对 NonZero/GatherND 这类**数据依赖 shape**的组合 shape 推导错（GatherND 输出的视觉 token 数与 mask 的 seq 绑定丢失，288 被推成 163）。

修复：**add_ds wrapper**——图外把 ds 按 visual_pos_masks 展开成 [1, seq, 2048] 的 dense（非 mask 位置为 0），图内 `h = h + ds_full` 纯 Add，无任何数据依赖 shape 算子。

验证（verify_where_ds.py）：**add 版 vs 官方 index 赋值版 bit 级一致（max_diff=0.0000，全 27 层逐层 0）**，vs 真实链路 1.7 ULP（当前 engine）。ONNX 2528 节点，6 输入全 FP16 [1, s0, ...]。

**3.8.2 layerPrecisions 静默失效（TRT 层名坑）**

用 ONNX tensor 名（pow_1/mean/add_42/...）生成 648 个 RMSNorm 的 layerPrecisions 后**完全无效**，TRT vs target 仍 860（107 ULP）：

```
check_parser_names.py: layerPrecisions 目标 648，精确匹配 ILayer.name: 0
```

根因：TRT ONNX parser 生成的 ILayer 名是 **node_{op}_{idx}**（如 node_Pow_58，= ONNX 节点的 name），而 fp32_layers.txt 用的是 ONNX 输出 tensor 名（pow_1 = node_Pow_58 的输出）。名字对不上 → 约束静默忽略（obey 模式也不报错）。

修复：Sqrt 回溯法找到 108 个 RMSNorm 子图，取其 6 个节点的 **name** 生成 `llm_add_ds_fp32_layers.txt`（脚本 gen_fp32_layers.py）。构建 log 出现 `Set layer node_Pow_58 to precision fp32` 即生效。

**3.8.3 最终精度（新参考数据 = 正确 M-RoPE，2026-08-28 更新，verify_engine.py 实测）**

| 配置 | engine 路径 | TRT vs 真实链路 | ULP (abs_max 14750, ULP=8) | 耗时 ms/iter | 状态 |
|------|------------|----------------|---------------------------|-------------|------|
| fp16engine（无 RMSNorm 约束） | `fp32onnx_fp16engine/llm_add_ds_fp16.plan` | 835.3 | 104.4 | 4.98 | ❌ |
| **fp16engine + RMSNorm FP32（当前部署）** | `fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan` | **13.6** | **1.7** | **5.03** | ✅ |
| fp32engine（全 FP32） | `fp32onnx_fp32engine/llm_add_ds_fp32.plan` | 0.888 | **0.1** | 10.06 | ✅ |

> 三个 engine 均为 fp32 IO（2026-08-28 build_llm_engine.py 改 IO 统一 fp32 后构建），用 verify_engine.py 对同一真实链路 target 实测（精度 + benchmark：100 iters + 10 warmup，CUDA events，seq=316）。全 FP32 engine 5.4GB（fp16 版 2.7GB），耗时慢 2.02x（10.06 vs 5.03 ms/iter）。

- **FP32 wrapper vs 真实链路 = 0.0000**（cos/sin/target 原生 FP32 + sdpa attention 后逐位一致）
- 三种 deepstack 注入方式（index 赋值 `OfficialWithDSWrapper` / torch.where `WhereDSWrapper` / add `AddDSWrapper`）互相比 **bit 级一致（0.0000）**（verify_where_ds.py 实测，FP16/FP32 双链路）——误差来自 TRT FP16 kernel（attention/GEMM 归约顺序）vs 真实 FP32 链路的固有实现差异，与 deepstack 注入方式或 causal mask 路径无关
- **RMSNorm 约束价值**：104.4 ULP → 1.7 ULP（改善 98.4%），**耗时仅 +0.9%**（4.98 → 5.03 ms，噪声级）——精度白拿、零性能代价
- 13.6/14750 ≈ 0.09% 相对误差，对 VLA 下游（第 28 层 + final norm + lm_head）影响可忽略

---

## 4. 踩坑记录

### 4.1 跨容器 LLM 权重未加载

**现象**：wrapper vs 参考数据 max_diff=1422。
**根因**：export 脚本只加载了 checkpoint-46000 的 ViT 权重（315 个），LLM 用的是 HF 原始权重；VLA 容器加载了全部 625 个权重。embed_tokens 有 241 个权重差异（max_diff=3e-8）。
**修复**：所有脚本改为加载 checkpoint 全部权重（`encoders.vision_language.vlm.` 前缀映射，lm_head 与 embed_tokens tied）。

### 4.2 跨容器 torch 版本差异

**现象**：加载全部权重后 max_diff=8.0 不为 0。
**根因**：whh_vla_trt2（torch 2.7）和 whh_trtllm_r130r20（torch 2.11）的 SDPA 实现不同（CPU FP16 0.002 vs 0.0；GPU FP32 0.443 vs 0.001，torch 2.7 对 enable_gqa/repeat_kv 分配了不同 backend）。8.0 = FP16 累积差异，非错误。

### 4.3 onnxruntime 版本问题

**现象**：`import onnxruntime` 段错误或 `Unsupported IR version: 10`。
**根因**：GPU 版与容器 CUDA 12.8 / cuDNN 9.7.1 冲突（cuDNN 8 编译的旧版、protobuf C++ ABI 冲突、numpy 2.x 不兼容等）。
**解决**：`onnxruntime==1.19.2`（CPU-only），CPUExecutionProvider 推理。

### 4.4 ONNX simplify 的 protobuf 2GB 限制

**现象**：`onnxsim.simplify()` 报 `exceeded maximum protobuf size of 2GB`。
**解决**：cd 到 ONNX 目录（external data 相对路径），`load_external_data=True` 加载后 simplify，保存用 `save_as_external_data=True`。2820 → 2739 节点，验证不变。
> 注：2.7GB 模型仍有被 kill 的风险，新流程（add_ds）已不需要 simplify。

### 4.5 trtexec layerPrecisions 必须合并成单参数

**现象**：`Unknown option: --layerPrecisions=...`
**根因**：trtexec 不接受多个独立 `--layerPrecisions` 参数（会被当成新选项）。
**解决**：逗号分隔合并成一个 `--layerPrecisions=layer1:fp32,layer2:fp32,...`。

### 4.6 layerPrecisions 对计算索引的算子报错

**现象**：`cannot use precision Float for layer that computes indices`
**根因**：Slice/Shape/Equal 等输出 INT64/BOOL，不能强制 FP32。
**解决**：按输出 dtype 过滤，只保留输出全 FP32 的节点（最终 648 个 RMSNorm 算子）。

### 4.7 ViT 与 LLM 的 rotary 路径差异

**现象**：ViT 的 rotary_inner_fp32_layers.txt（336 节点）能构建，LLM 的不能。
**根因**：ViT 用 legacy 导出（Slice 输出全 FP32），LLM 用 dynamo 导出（Slice 输出 FP16 + INT64，Cast 被消除）。
**结论**：两者 rotary 不可用同一策略，LLM 只保留 RMSNorm 的 648 个 FP32 节点。

### 4.8 layerPrecisions 名字必须用 TRT 层名（node_{op}_{idx}）

**现象**：用 ONNX tensor 名生成的 648 个约束静默失效（0 命中），TRT vs target 仍 860。
**根因**：TRT ILayer 名 = ONNX 节点 name（node_Pow_58），不是输出 tensor 名（pow_1）。匹配不上时 obey 模式也不报错。
**解决**：gen_fp32_layers.py 用 Sqrt 回溯取节点 name（见 3.8.2）。

---

## 5. 最终结论

| 误差源 | 根因 | 修复 | 效果 |
|--------|------|------|------|
| 1 | RMSNorm ReduceMean 在 FP16 下归约精度不足，每层多引入 ~1 ULP，27 层累积 106 ULP | layerPrecisions 强制 648 个 RMSNorm 节点 FP32 | 107 → 2 ULP |
| 2 | 官方 deepstack index 赋值 trace 出 NonZero/GatherND，TRT 对数据依赖 shape 推导崩（288≠163） | 图外展开 ds_full + 图内纯 Add（bit 级等价） | 构建推理通过 |

- **torch 层面**：三个 wrapper 与真实链路完全对齐（8.0 = 1 ULP）
- **TRT 层面**：16 / 14752 = 2 ULP ≈ 0.11% 相对误差，可接受
- **全 FP32 engine 因构建 OOM（exit=-9）不可行**，混合精度方案即最终方案
- 过程中被排除的假设：FP32 causal mask、IsNaN 缺失、动态 shape、mask 构造方式、graph 拓扑、deepstack torch.where 写法、未使用输入、FP16 溢出——见 3.4 排除表

---

## 6. 文件索引（当前状态）

### 6.1 参考数据（llm_engine/）

| 文件 | 说明 |
|------|------|
| llm_pt_reference_no_ds.safetensors | 真实链路截取，无 deepstack（17 tensor，正确 M-RoPE cos/sin） |
| llm_pt_reference_with_ds.safetensors | 真实链路截取，有 deepstack（17 tensor） |
| llm_pt_reference_with_ds.json | with_ds 版元信息 |
| ~~llm_pt_reference_dynamo.safetensors~~ | 已作废（position_ids 为 arange，非 M-RoPE） |

### 6.2 ONNX 与 engine（llm_engine/）

| 文件 | 说明 |
|------|------|
| llm_add_ds.onnx (+ .data) | **最终 ONNX**（FP32 权重 + FP32 IO，add_ds wrapper，2026-08-28 起纯 FP32 导出） |
| llm_add_ds_fp32_layers.txt | **layerPrecisions 配置**（648 RMSNorm 节点，TRT 层名格式 node_Pow_58:fp32） |
| fp32onnx_fp16engine/llm_add_ds_fp16.plan | fp16engine（无 RMSNorm 约束，104.4 ULP，对照用） |
| fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan | **最终 engine**（2.7GB，fp32 IO + fp16 内部 + RMSNorm FP32，1.7 ULP） |
| fp32onnx_fp32engine/llm_add_ds_fp32.plan | 全 FP32 engine（5.4GB，0.1 ULP，慢 2.6x，备选） |
| fp32onnx_*/llm_add_ds_fp16_fp32_layers.log / layerinfo.json / profile.json | 构建日志与层信息证据 |
| keep/ | layerinfo json ×2 + hs.pt ×2（排查证据） |
| simplify_llm_onnx.py | 已不适用（2.7GB 模型 protobuf 超限） |

### 6.3 脚本（trtllm/）

| 文件 | 说明 |
|------|------|
| export_llm_onnx.py | 四 wrapper 导出（manual_no_ds / official_no_ds / official_with_ds / add_ds），--wrapper 选择，--export 导出。**2026-08-28 起改为纯 FP32 真实链路**（模型 FP32 + sdpa attention + 输入输出 FP32，复用 llm_wrapper.load_model） |
| build_llm_engine.py | TRT engine 构建（FP16/FP32/FP16+fp32_layers），自动读 ONNX 输入信息 |
| gen_fp32_layers.py | 从 ONNX 生成 TRT 层名格式 fp32_layers.txt（Sqrt 回溯 108 个子图） |
| verify_where_ds.py | where/add 版 vs 官方 index 赋值 bit 级等价验证 |
| verify_engine.py | 通用 TRT engine vs 真实链路 target 验证 |
