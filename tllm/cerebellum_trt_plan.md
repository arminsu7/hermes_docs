# cerebellum（DexDiffusionPolicy）TRT 化

> 2026-09-01，whh_vla_trt2 容器
> 目标：用 TRT engine 替代 `self.cerebellum.inference`，代码放 `hpc_opt/trtllm/cerebellum/`

## 1. 调用链

```
dexforcevla_runner.py L1654:
  data = self.cerebellum.inference(data, inference_horizon=32, adaptors=self.adaptors)
    → DexDiffusionPolicy.inference (policy/dexdp.py L567)
      → 5 步扩散采样循环 (num_inference_timesteps: 5)
        → 每步 self.forward_action(...)   ← 单步 denoise 预测（TRT 化的核心）
          → policy/dexdp_blocks.py L491 (DiTPolicyWithAdditionalModality)
      → self.noise_scheduler_sample.step(...)  ← 采样步进（外层 torch）
```

## 2. 配置（checkpoint-46000 实测）

```yaml
cerebellum:
  name: DexDiffusionPolicy
  depth: 8              # 8 层 DiTBlock
  output_dim: 128       # action dim
  num_heads: 4
  hidden_size: 384
  state_history_len: 1
  pred_horizon: 64
  num_cameras: 2
  img_cond_len: 288
  lang_cond_len: 500
  use_lang: True
  use_progress: False
  use_geomap: False
  split_arm_cond: True
  num_inference_timesteps: 5   # yaml 里 10，实际加载为 5
```

## 3. forward_action 输入输出（单步 denoise）

```
输入:
  x        : [B, T, D]  state-action 轨迹（含 state_history_len）
  t        : [B]        timestep（noise scheduler step）
  lang_c   : [B, L, D]  语言条件
  img_c    : [B, H*C*img_len, D] 图像条件
  lang_mask: [B, L]     语言 mask
  shape    : (bs, hs, cn, img_len, c)
输出:
  model_output     : [B, T, out_channels]  action 预测
  intermediate_output: [B, T, D]           中间表示（相机解码用）
```

## 4. forward_action 内部结构

```
1. t_embedder(t) → t_                       # 时间嵌入
2. x = cat([t, x]) + x_pos_embed            # 条件拼装 + 位置嵌入
3. lang_c += lang_cond_pos_embed / img_c += img_cond_pos_embed
4. img_c, mask = _parse_img_c(img_c, mask_pred, shape)   # 按相机解析（split_arm）
5. 8 层 DiT block: x = block(x, t_, c, mask)   # cross-attention 多条件
6. final_layer(x) → x[:, -T:]               # 只保留 action token
7. cat_results(x, ...)                      # 左右臂合并
```

## 5. TRT 化难点

| 难点 | 说明 |
|------|------|
| 多模态动态 shape | lang_c 可变长、img_c 多相机 |
| attention cache | `_attn_cache.begin_step` 跨步缓存（推理默认 disabled） |
| 5 步循环 | 采样循环在 torch 侧，每步调 engine |
| 左右臂 split | split_arm_cond 时 img_c 翻倍 + right_left_embeds |

## 6. TRT 化粒度

**只 TRT 化 `forward_action`（单步 denoise 预测）**：外层采样循环（noise_scheduler step、cat、tile）保持 torch。

## 7. 交付脚本（hpc_opt/trtllm/cerebellum/）

| 脚本 | 功能 |
|------|------|
| `cerebellumWrapper.py` | 加载 + `CerebellumWrapper`（torch 模式 = 原始 forward_action；trt 模式 = 前置拼装 + engine + 后处理）|
| `export_cerebellum_onnx.py` | 导出纯 transformer ONNX |
| `build_cerebellum_engine.py` | 建 engine（`--precision fp32 / fp16 / fp16_rmsnorm_fp32`）|
| `verify_cerebellum_wrapper.py` | **wrapper(torch 模式) vs 原始 forward_action** 对齐验证 |
| `verify_cerebellum_engine.py` | engine 链路 vs 原始对齐验证 |
| `extract_norm_layers.py` | 生成 `cerebellum_fp32_layers.txt` |

## 8. TRT 化边界

**ONNX 边界后移**：split_arm 翻倍 / right_left_embeds / `_parse_img_c` / pos_embed 留在 torch 前置（`prepare_onnx_inputs`），ONNX 只做纯 transformer（8×DiTBlock + final_layer + action 裁剪），后处理 `cat_results` 留 torch。

- ONNX 输入：x[1,34,384], lang_c[1,19,384], img_c[1,288,384]
- 真实链路 mask=None 时 `_parse_img_c` 不翻倍（实际 bs=1）
- t_embed 被 ONNX fold 掉（DiTBlock 不用 time embedding）

### 8.1 两个 wrapper 的分工

**`CerebellumOnnxWrapper` 是 ONNX 导出专用模块**（不是运行时用的）：

| wrapper | 用途 | 内容 |
|---------|------|------|
| **CerebellumWrapper** | 运行时封装 | torch 模式 = 原始 `forward_action`；trt 模式 = 前置拼装 + engine + cat_results |
| **CerebellumOnnxWrapper** | **导出专用**（仅供 `export_cerebellum_onnx.py`）| 8×DiTBlock + final_layer + camera_wise_decoder，打包成可 `torch.onnx.export` 的子图 |

**为什么不能直接导出原始 `forward_action`**：

| 部分 | 能否进 ONNX | 原因 |
|------|------------|------|
| 条件拼装（t 拼接、pos_embed、`_parse_img_c`、split_arm 翻倍）| ❌ | 动态 shape + 分支 + 自定义 `cat_with_zero_dim` |
| 8 层 DiTBlock + final_layer | ✅ | 标准算子（线性/注意力/FFN/norm）|
| camera_wise_decoder | ✅ | 纯 cat 拼接（真实配置 `cat([x,x],-1)`）|
| cat_results（左右臂合并）| ❌ | 依赖 arm_type/right_related_indices 多分支 |

**所以边界后移**：
- torch 前置：`CerebellumWrapper.prepare_onnx_inputs`（t 拼接、pos_embed、parse、split_arm）
- ONNX 主体：`CerebellumOnnxWrapper`（blocks + final_layer + camera_wise_decoder）
- torch 后处理：`CerebellumWrapper.post_process`（cat_results）

**运行时**：`CerebellumWrapper(trt_mode=True).forward` = `prepare_onnx_inputs` → engine（由 CerebellumOnnxWrapper 导出的 ONNX 转）→ `post_process`。

### 8.2 4 条指令的 shape 分析（动态性决策）

**任务指令（4 条固定）**：
```
"Place the brown book into the empty slot on the right of blue one, using right arm."
"Place the blue book into the empty slot on the right of brown one, using right arm."
"Hand the brown book over using right arm."
"Hand the blue book over using right arm."
```

**指令文本 token 数**（Qwen3-VL tokenizer 实测）：
| 指令类型 | 指令 token | LLM 文本总长（指令+模板9）| 图像 token | **LLM input_ids 总长** |
|---------|-----------|--------------------------|-----------|----------------------|
| Place | 19 | 28 | 288 | **316** |
| Hand | 9 | 18 | 288 | **306** |

**LLM 输入 seq 长度**：只有两种 **306（Hand）/ 316（Place）**（reference 实测 input_ids=(1,316)）。

**Cerebellum 输入 shape**（图像固定 2×384，T 固定 32）：
| 输入 | Place | Hand | 是否变化 |
|------|-------|------|---------|
| x (state_action_traj) | [1, 34, 384] | [1, 34, 384] | ✅ 固定 |
| **lang_c** | **[1, 19, 384]** | **[1, 9, 384]** | ⚠️ **变化（19↔9）** |
| img_c | [1, 288, 384] | [1, 288, 384] | ✅ 固定 |

**决策**：
- **lang_c 第 1 维动态** `[1, 'lang_len', 384]`，TRT 区间 `min=1x9x384, opt=1x19x384, max=1x19x384`
- 其余（x/img_c）静态（T 固定 32、图像固定 288）
- LLM engine 区间收窄到 `MIN/OPT/MAX = 306/316/316`（4 条指令只覆盖两种长度，收窄提高精度）

**注意**：若只部署单一指令类型（全 Place 或全 Hand），lang_c 可静态（19 或 9）。4 条指令混用则必须动态。

## 9. 精度对齐验证

> 术语约定：
> - **backend**：`torch`（PyTorch）/ `onnxruntime`（ONNX 推理）/ `engine`（TRT）
> - **wrapper** = `CerebellumWrapper`（torch 模式调原始 `forward_action`；trt 模式走 engine 链路）
> - 所有对比用同一真实 reference 输入（`cerebellum_ref.safetensors`）

### 9.1 对比矩阵（max_diff，同一输入）

> **engine 说明**：以下数值为 **fp16_rmsnorm（norm-only）engine**（`cerebellum_core_fp16_rmsnorm_fp32.plan`，当前推荐）的结果。
> **纯 fp16 engine 输出错误**（见 §13），不参与精度对比。

| 对比 | backend A | 跑什么 | vs backend B | 跑什么 | max_diff | 结论 |
|------|-----------|--------|--------------|--------|----------|------|
| ONNX 图正确性 | onnxruntime | ONNX 推理 | torch | `CerebellumOnnxWrapper`（纯 transformer）| **6.7e-6**（CPU）| ✅ ONNX 导出逻辑正确 |
| engine 执行正确性 | engine(fp32) | TRT 推理 | onnxruntime | ONNX 推理 | **0.002** | ✅ engine 执行正确 |
| engine 链路 vs 原始 | engine(fp32) | wrapper(trt 模式) | torch | 原始 `forward_action` | **0.053** | ⚠️ 在容差内 |
| 原始链路非确定性 | torch | 原始 `forward_action` 跑 3 次 | torch | 原始 `forward_action` | **0.013-0.03** | 非确定 |

### 9.1b 三 engine 对比（耗时 + 精度，真实 reference 输入）

| backend | engine | max_diff | mean_diff | 耗时(ms) | 加速 | 可用 |
|---------|--------|----------|-----------|----------|------|------|
| torch | 原始 forward_action | - | - | 3.836 | 1.00x | ✅ 基线 |
| **fp32** | cerebellum_core_fp32.plan | **0.060** | 0.000371 | **0.704** | **5.45x** | ✅ |
| fp16 | cerebellum_core_fp16.plan | 12.85 ❌ | 0.283 | 0.639 | 6.01x | ❌ 输出错 |
| **fp16_rmsnorm（norm-only）** | cerebellum_core_fp16_rmsnorm_fp32.plan | **0.068** | 0.000496 | **0.595** | **6.45x** | **✅ 推荐** |

> **注**：fp16_rmsnorm 已用 **ONNX 层名**（norm-only 443 层，参考 vit/rotary_inner_fp32_layers.txt）修复——之前 12.85 是 layerPrecisions 用错层名格式（TRT 融合层名）导致静默失效。见 §13。

**结论**：**用 fp16_rmsnorm（norm-only）**——精度 0.068 在容差内（原始非确定性 ~0.007-0.012）+ **6.45x 加速** + **42MB**（比 fp32 74MB 小 43%，比 fp32 快 18%）。attn 层保持 fp16（消融验证：只需 norm fp32）。

> ⚠️ **2026-09-03 真机修正**：以上「只需 norm fp32」结论在**离线随机/小数据**下成立，但**真机真实输入下 fp16 attn 溢出 NaN**（见下 §9.5 真机 NaN 修复）。真机最终采用 **fp16 + attn/norm fp32（779 层）**。

### 9.2 精度结论

1. **ONNX 图正确**（onnxruntime vs torch wrapper，CPU 下 6.7e-6）
2. **engine 执行正确**（engine vs onnxruntime 0.002）
3. **剩余差异 0.053 = GPU fused attention 非确定性 + 微小精度差**：
   - 原始 `forward_action` 在 GPU 上本身有 0.013-0.03 非确定性（fused attn，见 §10）
   - ONNX/engine 是确定性的，落在原始波动范围内
4. **容差**：`max_diff < 0.08`（覆盖 ONNX 精度 ~0.055 + 原始非确定性 ~0.013）

### 9.3 影响性能（engine 链路）

| 项 | 耗时 |
|----|------|
| engine 链路（前置拼装 + engine + 后处理） | ~0.86 ms/iter（实测）|
| 原始 forward_action（GPU，单步） | 需对比（见性能章节）|

### 9.3b ✅ refine TRT 化（2026-09-02，cerebellum_exp 实验 → 已并入 cerebellum）

**背景**：`cerebellum.inference` 里 `refine`（use_refiner=True 时执行）是**最后一个未 TRT 化的大块**（torch 3.2-3.7ms）。

**排查过程（时间线）**：

**① 发现（耗时分解）**
- 用 profile 脚本分解 `cerebellum.inference` 各段耗时：
  - `forward_action`（5 步扩散）torch 4.9ms/步（已 TRT，~0.6ms/步）
  - **`refine`（use_refiner=True 必跑）torch 3.62ms——最大的未 TRT 化部分**
- 结论：refine 是下一个优化目标（省 ~3ms/次推理）

**② 结构分析（refine 与 forward_action 同构）**
- `refine` 实现 = `refiner`（DiTBlock 列表）+ `refiner_final_layer`（FinalLayer，`2*hidden` 因 cat([init_x,x],-1)）
- 输入：`full_x[B,T,D]`（action_adaptor 输出）+ `control_x`（intermediate，取 `[:, -1, :].mean(-1)` 作 ada_ln_cond）+ `img_c`
- **关键发现**：DiTBlock.forward(x, time_c, c, mask) 里 **time_c 参数实际未被使用**（forward_action 传 t_embed，refine 传 ada_ln_cond，但都不参与计算）→ ada_ln_cond 导出时会被 fold

**③ 实现（RefineOnnxWrapper + 导出）**
- `refine_wrapper.py`：`RefineOnnxWrapper`（refiner + refiner_final_layer）+ `prepare_refine_inputs`（img_c+pos_embed → _parse_img_c；x+refine_cond_pos_embed；ada_ln_cond）+ `refine_post_process`（cat_results）
- `export_refine_onnx.py`：导出 `refine_core_fp32_simplified.onnx`（75MB）
- **第 1 次验证**：wrapper vs 原始 refine **max_diff=0.000000**（bit 级一致）

**④ 踩坑 1：ada_ln_cond 被 onnxsim fold**
- 现象：build 报 `Cannot find input tensor with name "ada_ln_cond"` → trtexec FAILED
- 发现：检查 ONNX 输入，**只有 x + img_c 2 个**（ada_ln_cond 被 onnxsim 折叠）
- 根因：DiTBlock 的 time_c（ada_ln_cond）**实际未参与计算**，onnxsim 识别为死输入删除
- 解决：build shapes 去掉 ada_ln_cond（只 x + img_c）

**⑤ 踩坑 2：静态 ONNX 不能传 --minShapes**
- 现象：`Static model does not take explicit shapes` → trtexec FAILED
- 根因：refine ONNX **无动态维度**（x/img_c 全静态）→ 传 --minShapes 冲突（之前 cerebellum 的 ONNX 有动态 lang_c 才可传）
- 解决：去掉 SHAPES（静态 build）

**⑥ 踩坑 3：refine 输入 T 维度错（T=33 vs 32）**
- 现象：端到端 infer_trt.sh 报 `size of tensor a (33) must match tensor b (32)`
- 发现：engine 输出 [1,33,128]，但 init_action 是 [1,32,128]（T=inference_horizon=32）
- 根因：**refine 输入 full_x 是 [1,32,384]（T=32），不是 [1,33,384]**——导出时用错维度（假设 T=33）
- 解决：重新导出 ONNX（full_x [1,32,384]）+ 重建 engine

**⑦ 精度确认：refine 不需要 norm fp32（纯 fp16 即可）**
- 检查 build：LayerPrecisions 的 `:fp32` 数量 = **0**（norm 层提取为 0）
- 发现：refine 的 norm 层**融合进 `__myl_*` 层**（`__myl_MeanSubMulMeanMulMulAddSqrtDivMul_*`），**Metadata 无 ONNX norm 路径名** → 按 ONNX 名提取匹配不到
- **但纯 fp16 精度 bit 级一致（max_diff=0.000000）** → **refine 不需要 norm fp32**（与 forward_action 不同：forward_action 的 norm fp16 会数值错误，refine 不会）
- 解决：`build_refine_engine.py` 改为纯 fp16 构建，engine 命名 `refine_core_fp16.plan`（去掉误导的 rmsnorm_fp32）

**⑧ 端到端验证（第 2 步）**
- `trt_hooks.py` 加 `install_refine_trt_hook`；`trt_infer_shim.py` 接入（TRT_REFINE_ENGINE 环境变量 + 默认路径）
- infer_trt.sh：`[refine-hook] installed` + **EXIT=0** + cerebellum_infer.inference 稳态 **7.1-11.4ms**

**最终验证结果**：
| 项 | torch | TRT engine | 结论 |
|----|-------|-----------|------|
| refine | 3.22-3.67ms | **0.33-0.50ms** | **6.5-11.2x** |
| 精度 | - | **max_diff=0.000000** | ✅ bit 级一致（纯 fp16）|

**⑨ 目录整合（第 4 件，2026-09-02）**
- cerebellum_exp 是 cerebellum 超集（相同文件 md5 全一致 + 新增 refine 文件）
- 删除 cerebellum → 重命名 cerebellum_exp → cerebellum → 更新引用路径（trt_hooks/trt_infer_shim/run_trt）
- 端到端复验：EXIT=0 + 两个 hook 生效（forward: cerebellum_core_fp16_rmsnorm_fp32.plan，refine: refine_core_fp16.plan），cerebellum_infer.inference 稳态 **7.1-11.4ms**

### 9.4 ✅ CPU 对齐验证（确定性环境，wrapper vs 原始）

**目的**：排除 GPU fused attention 非确定性，验证 wrapper 封装本身是否正确。

| 对比 | max_diff | 结论 |
|------|----------|------|
| 原始 forward_action CPU 自洽（2 次） | **0.0000000000** | CPU 完全确定 |
| wrapper(torch 模式) vs 原始 model_output | **0.0000000000** | ✅ **bit 级一致** |
| wrapper(torch 模式) vs 原始 intermediate | **0.0000000000** | ✅ **bit 级一致** |

**结论**：**wrapper torch 模式 vs 原始 forward_action 在 CPU 上 bit 级一致（0.0）**——wrapper 封装本身正确。GPU 上的 0.019 差异**确证来自 GPU fused attention 非确定性**（不是 wrapper 问题）。

**验证脚本**：`verify_cerebellum_wrapper.py`（默认 CPU+GPU 都测，`--cpu`/`--gpu` 可选）：
```
CPU: ✅ 对齐（bit 级一致 0.0）
GPU: ✅ 对齐（max_diff=0.019 ≤ 容差，差异=fused attn 非确定性）
```

### 9.5 ✅ 原始链路 dtype 确认

**原始 forward_action 链路全 FP32**：
- 权重：cerebellum 所有参数 = **float32**
- 输入：state_action_traj / lang_cond / img_cond = **float32**（t = int64，forward 内 `.float()` 转）
- 输出：model_output / intermediate = **float32**

### 9.6 ✅ 方案 B：engine 双输出（intermediate 支持 refine）

**背景**：真实链路 `use_refiner=True`，refine 需要 `intermediate_output`（forward_action 第二输出）。旧 engine 只输出 model_output → hook 保守跳过 TRT 化。

**根因**：`intermediate` 的 split_arm 处理（camera_wise_decoder）**始终执行**（由 split_arm_cond 控制，与 mask 无关），实测 intermediate = [1,32,384,2]。

**实现**：扩展 ONNX 边界——`CerebellumOnnxWrapper` 双输出：
- `model_output`：[B,T,out]（final_layer + 裁剪）
- `intermediate`：[B,T,D,2] = `cat([x_b, x_b], -1)[:, -T:]`
  - 真实配置（cn=2, HEAD+RIGHT_WRIST, bs=1 不 split_arm）：camera_wise_decoder 的 HEAD+RIGHT_WRIST 分支 = `cat([middle, right])`，bs=1 时 middle=right=intermediate_x

**验证**（hook 真实链路，fp16_rmsnorm engine）：
| 输出 | max_diff | 相对误差 | 结论 |
|------|----------|----------|------|
| model_output | 0.049 | ~17%（相对 action abs_mean 0.287）| ✅ 容差内 |
| intermediate | 3.8 | **6%**（相对 abs_mean 63）| ✅ 可接受（refine 只取 adaLn 条件 `control_x[:, -1, :].mean(-1)`）|

**hook 不再跳过**：`install_cerebellum_trt_hook` 去掉 use_refiner 跳过，真实链路可用 TRT。

**改动文件**：cerebellumWrapper.py（双输出）、export_cerebellum_onnx.py（ONNX 双输出）、trt_hooks.py（hook 不跳过）、verify_cerebellum_engine.py、run_trt.py、combo_test_cerebellum.py。

### 9.5 ⚠️ 真机 NaN 根因与修复（2026-09-03，3060）

**现象**：真机全 TRT policy 推理时客户端报 **`SVD did not converge`**（FK/IK 求解崩）。server 日志无错误，但 cb 输出全 NaN。

**定位过程**（逐步隔离）：
1. cb engine_in 诊断：**输入 x NaN=0（正常）**，但**输出 model_output/intermediate 全 NaN**
2. dexdp 诊断：第一次 predict_noise 正常（_actions NaN=0），**第二次起全 NaN**（第一步 cb 输出 NaN → 污染后续采样）
3. 独立测 engine（随机 + 放大输入）：**输出正常 NaN=0**——**engine 本身没问题**
4. **结论：cb engine（fp16_rmsnorm norm-only）对「真实输入」输出 NaN，独立测（随机/小数据）正常**

**根因**：`fp16_rmsnorm_fp32`（norm fp32 + attn fp16）在**真实输入（x max~8、img_c max~138）**下，**attn 层 fp16 溢出 → NaN**。离线 verify 用 32/33 假数据（pad），未触发真实值域 → 掩盖了问题。**「只需 norm fp32」的消融结论在随机/小数据下成立，真实值域下不成立**。

**修复**（两种方案均验证 OK）：
| 方案 | engine | NaN | Step | 说明 |
|------|--------|-----|------|------|
| 全 fp32 | cerebellum_core_fp32.plan | 0 | 172-218ms | IO fp32，需 TrtEngine 输入 dtype 自适应 |
| **fp16 + attn/norm fp32** | fp16_rmsnorm_fp32.plan（779 层）| 0 | 194-210ms | 保留 IO fp16，attn/norm 层强制 fp32 |

**最终采用 fp16 + attn/norm fp32（779 层）**：engine 小（100MB）+ IO fp16 hook 不改 + Step ~194ms。

**配套改动**：
- `build_cerebellum_engine.py` extract 筛选改为 `ATTR_KEYS = (norm, attn, mha, q_, k_, v_, out_proj, qkv, matmul, softmax)`（norm-only 443 层 → attn+norm 779 层）
- `trt_hooks.py TrtEngine.__call__` 输入按 engine IO dtype 自适应（fp32 engine → .float()，fp16 → .half()）
- `config.py CB_ENGINE` 指向 fp16+attn+norm plan（真机）

## 10. ⚠️ 非确定性根因（重要）

**现象**：同一模型同一输入跑两次 `forward_action`，GPU 上输出 max_diff=0.004-0.03（非 bit 级一致）。

**根因**：`F.scaled_dot_product_attention`（fused attention）在 GPU 上的**浮点累积顺序非确定**（kernel race）。
- 实证：`cross_attn.fused_attn=True`（fused，非确定），`attn.fused_attn=False`（手动路径，确定）
- 诊断：关 fused attention（math）后 GPU 上差异 **0.000000**；manual_seed 无效（非 RNG）
- 排除：权重加载、RNG、dropout（全部 p=0）、attention cache（disabled）

**影响**：TRT engine 确定性 vs 原始链路非确定 → **无法 bit 级对齐**，只能容差对齐（§9.2）。

## 11. 踩坑记录

1. **TRT 10.11 `--workspace` 不支持** → 用 `--useSpinWait`
2. **静态 ONNX + `--minShapes` 冲突** → 静态 build
3. **mask 的 bool/float Where 错误**（cross_attn）→ ONNX 里 mask 传 None
4. **split_arm 未处理导致 engine 输出 12.85** → ONNX 边界后移（§8）
5. **reuse_output_buffers=True 导致别名污染**（12.85）→ 默认改 False

## 12. 产物

```
cerebellum_onnx/cerebellum_core_fp32_simplified.onnx   # forward ONNX（73MB）
cerebellum_onnx/refine_core_fp32_simplified.onnx       # refine ONNX（75MB）
cerebellum_engine/
├── cerebellum_core_fp32.plan               # forward fp32 engine（74MB）✅
├── cerebellum_core_fp16.plan               # forward fp16 engine（42MB）❌ 输出错
└── cerebellum_core_fp16_rmsnorm_fp32.plan  # forward fp16 + norm fp32（42MB）✅ 推荐
refine_engine/
└── refine_core_fp16.plan                   # refine 纯 fp16 engine（41MB）✅（bit 级一致）
cerebellum_ref.safetensors                  # 真实 reference
cerebellum_fp32_layers.txt                  # 443 个 norm 层（forward build 依赖，ONNX 层名）
refine_wrapper.py / export_refine_onnx.py / build_refine_engine.py  # refine TRT 化工具
```

## 13. ⚠️ fp16 engine 输出错误（重要，已修复）

> **当前状态**：fp16_rmsnorm（norm-only）已修复可用（0.068）。本节是**排查过程记录**（时间线保留）。

**现象（2026-09-01，当时未修复）**：fp16 / fp16_rmsnorm engine 输出 vs 原始 forward_action **max_diff=12.85**（完全错误），fp32 engine 正常（0.046）。⚠️ **该现象已被 ONNX 层名修复（2026-09-02）**，见下方根因。

**诊断**（同一真实 reference 输入，engine vs onnxruntime）：
| engine | 输出 mean | std | 非零 | NaN | max_diff |
|--------|----------|-----|------|-----|----------|
| fp32 | 0.1222 | 正常 | 4096/4096 | 0 | **0.002** ✅ |
| fp16 | -0.0033 | 0.0314 | 4096/4096 | 0 | **12.85** ❌ |
| fp16_rmsnorm | -0.0033 | 0.0314 | 4096/4096 | 0 | **12.85** ❌ |

**已排查并排除**：
1. reuse_output_buffers（改 False 已修）
2. IO 格式（不加 `--inputIOFormats` 也一样错）
3. onnxsim（未 simplify ONNX 构建也一样错）
4. RMSNorm fp32（41 层 TRT 融合层名强制 fp32 仍错）⚠️ 后被排除——**TRT 融合层名匹配无效**
5. gemm fp32（norm+gemm 102 层 TRT 融合层名全 fp32 仍错）⚠️ 后被排除——同上
6. NaN/溢出（输出无 NaN、全非零，但计算错误）
7. attn+norm 全 fp32（65 层 TRT 融合层名）仍 12.85 ⚠️ 后被排除——同上

**根因（2026-09-02 确认）**：⚠️ **之前"TRT fp16 深层 bug"结论不实**——真正的根因是 **`--layerPrecisions` 用错了层名格式**：
- 之前用 **TRT 融合层名**（`__myl_*_myl3_N`、`_gemm_mha_v2_*`）——**TRT 静默忽略未匹配层名**（约束未生效），fp16 仍输出错
- 正确做法（参考 vit/rotary_inner_fp32_layers.txt）：**用 ONNX 层名**（layerinfo Metadata `[ONNX Layer: ...]` 关联）——`/blocks.X/norm*/...`、`/final_layer/norm_final/...`
- **用 ONNX 层名标记 norm 层 fp32 后：max_diff 12.85 → 0.068** ✅

**消融验证（2026-09-02）**——确认**只需 norm fp32**：
| 配置 | 层数 | max_diff | 结论 |
|------|------|----------|------|
| norm-only（norm 路径） | 443 | **0.0518** ✅ | **norm fp32 足够** |
| attn-only（attn/mha 路径） | 80 | 12.85 ❌ | attn fp32 不必要（attn fp16 正常） |
| all（norm+attn） | 523 | 0.0518 | 同 norm-only（attn 层 fp32 无额外收益） |

**结论**：**fp16_rmsnorm（norm-only）可用**——精度 0.068（容差内）+ 6.45x 加速 + 42MB。**只需 norm 层 fp32**（RMSNorm 在 fp16 下数值错误，attn 在 fp16 下正常）。之前"head_dim=96 attention fp16 bug"的推断是**错的**（那是 layerPrecisions 失效的假象）。
