# 任务1C：TRT 代理 hook 接入真实推理链路（ViT + LLM）

> 容器：whh_vla_trt2（conda py310）
> 代码目录：/root/workspace/embodichain/hpc_opt/trtllm/vitllm_infer/
> **不改动任何原始推理代码**（dexechain/ + scripts/ 零修改）
> engine：ViT **V1**（fp16onnx_fp16engine，fp32 IO）+ LLM **L2**（fp32onnx_fp16engine_fp32RMSNorm，fp32 IO）——2026-08-31 从 15 组合测试选定（见 5.1）

---

## 一、目标与方案

### 1.1 目标

把已验证对齐的 TRT engine（ViT + LLM）接入 VLA 真实推理链路（`infer_realdata.sh`），
验证端到端输出与 torch 基线的一致性——**不改动原始推理代码**。

### 1.2 方案：hook 最小替换

通过 monkey-patch transformers 的 `Qwen3VLModel` 方法，把 ViT/LLM 的计算替换为 TRT engine，
其余（embed、masked_scatter、M-RoPE、deepstack 展开、第 28 层、final norm）仍走 torch：

| hook 位置 | 原始实现 | 替换为 |
|-----------|---------|--------|
| `Qwen3VLModel.get_image_features` | visual (ViT) 算 pooler + 3×deepstack | TRT ViT engine |
| `Qwen3VLTextModel.forward` | 28 层 decoder 循环 + deepstack 注入 | TRT LLM engine（前 27 层）+ torch（第 28 层 + norm） |

hook 触发方式：patch `Qwen25VLEncoder.forward_for_dexforcevla`，首次推理时给 `self.vlm` 装 hooks，
后续推理复用 engine。通过 `sitecustomize.py` 在 python 启动时按环境变量自动加载。

---

## 二、文件清单（vitllm_infer/）

| 文件 | 作用 |
|------|------|
| `trt_hooks.py` | TRT engine 封装（TrtEngine）+ ViT/LLM 两个 hook（核心） |
| `trt_infer_shim.py` | patch `Qwen25VLEncoder.forward_for_dexforcevla`，首次推理自动装 hooks |
| `sitecustomize.py` | python 启动时读 `USE_TRT_HOOKS=1` 自动 import shim（完全静默） |
| `infer_trt.sh` | 启动脚本：设环境变量 + PYTHONPATH，调原始 `scripts/infer_realdata.sh` |
| `run_torch.py` | torch 参考链路验证脚本（完整 VLM forward vs 真实链路截取） |
| `run_trt.py` | TRT 代理链路验证脚本（--no-vit-trt / --no-llm-trt 可单独替换；--bench 测耗时） |
| `combo_test_vit_llm.py` | 15 组合（ViT 5×LLM 3）全排列精度+耗时测试，汇总表格（V1/L1 等注释见脚本） |
| `combo_results.json` | 15 组合测试结果（max_diff / mean_diff / 耗时） |

### 2.1 TrtEngine 封装要点

```python
class TrtEngine:
    def __call__(self, inputs):
        # 输入按 engine 期望 dtype 自动适配（读取 engine.get_tensor_dtype）
        # 关键：VLA 真实模型是 FP32，engine IO 2026-08-31 起统一 fp32:chw（输入 FP32 直接喂）
        # 输出按 engine 实际 dtype 分配（HALF→fp16, FLOAT→fp32）
        ...
        ok = self.ctx.execute_v2([buffers[n].data_ptr() for n, _ in self.io])
```

### 2.2 ViT hook

```python
def hook(pixel_values, image_grid_thw=None, **kwargs):
    pv = pixel_values.float().contiguous()   # engine IO 是 fp32:chw
    outs = engine({"pixel_values": pv})
    pooler = outs[engine.output_names[0]]            # [288, 2048] fp32
    deepstack = [outs[n] for n in engine.output_names[1:]]  # 3×[288, 2048] fp32
    # 与原始 get_image_features 一致：按 grid_thw split pooler
    split_sizes = (image_grid_thw.prod(-1) // visual.spatial_merge_size ** 2).tolist()
    pooler_out = torch.split(pooler, split_sizes)
    return BaseModelOutputWithDeepstackFeatures(
        pooler_output=pooler_out, deepstack_features=list(deepstack), ...)
```

### 2.3 LLM hook

```python
def hook(input_ids, attention_mask, position_ids, inputs_embeds, visual_pos_masks,
         deepstack_visual_embeds, **kwargs):
    # 1. M-RoPE position_ids 处理（复制官方 forward）
    # 2. attention_mask = create_causal_mask(...)
    # 3. position_embeddings = rotary_emb(inputs_embeds, position_ids)  # fp32
    # 4. deepstack 按 visual_pos_masks 展开成 ds_full [1, seq, 2048]
    # 5. TRT engine 跑前 27 层 → layer26 输出（= hidden_states[-2]，VLA 需要的值）
    # 6. torch 补跑第 28 层 + final norm（保证 last_hidden_state/logits 路径完整）
    hidden_neg2 = engine({...})["hidden_states"]     # fp32（engine IO FP32，无需 cast）
    h27 = text_model.layers[27](hidden_neg2, ...)
    last_hs = text_model.norm(h27)
    return BaseModelOutputWithPast(last_hidden_state=last_hs, ...)
```

**hidden_states[-2] 的语义**（关键）：`Qwen3VLTextModel.forward` 用 `@capture_outputs`
装饰器通过 hooks 收集每层输出（[0]=embed，[1..28]=各层，[-1] 被 tie 成 norm 后），
所以 **[-2] = layer26 输出（norm 前）**——hook 里 TRT engine 输出直接作为 hs[-2]。

---

## 三、接入方式（不改原始代码）

```bash
# TRT 代理（默认 engine，最简命令——不依赖任何 BRAIN_DATA_PATH）
cd /root/workspace/embodichain
bash hpc_opt/trtllm/vitllm_infer/infer_trt.sh

# 关掉 TRT（torch 基线）
USE_TRT_HOOKS=0 bash hpc_opt/trtllm/vitllm_infer/infer_trt.sh

# 换 engine
TRT_VIT_ENGINE=... TRT_LLM_ENGINE=... bash hpc_opt/trtllm/vitllm_infer/infer_trt.sh
```

> **BRAIN_DATA_PATH 说明**：该参数是**原始 infer_realdata.sh 自带**的（`dexforcevla_runner.py`
> 里在 brain_infer 后把 brain 输出存成 .pt，供 cerebellum 导出用），**与 TRT 代理链路无关**。
> TRT 代理不需要它，不设则默认写 `/root/workspace/embodichain/brain_data.pt`（一次运行只写一次，
> 3MB，无开销）。仅在验证 TRT vs torch 输出对比时用它指定两个不同的保存路径：
> ```bash
> # 验证对比（非必需）：torch 基线 + TRT 代理各存一份，再 diff
> bash scripts/infer_realdata.sh                                  # → brain_data.pt
> BRAIN_DATA_PATH=/root/workspace/embodichain/brain_data_trt.pt \
>     bash hpc_opt/trtllm/vitllm_infer/infer_trt.sh               # → brain_data_trt.pt
> ```

`infer_trt.sh` 内部：
```bash
export USE_TRT_HOOKS=1
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"   # sitecustomize 可见
bash "${SCRIPT_DIR}/../../../scripts/infer_realdata.sh" "$@"
```

加载链：`python 启动 → sitecustomize → (USE_TRT_HOOKS=1) → trt_infer_shim.install() →
patch forward_for_dexforcevla → 首次推理装 hooks → TRT engine 代理`

---

## 四、踩坑记录

### 4.1 sitecustomize 的 print 污染第三方库（严重）

**现象**：第一版 sitecustomize 安装时 print 中文提示，导致 glfw 库崩溃：
```
File ".../glfw/library.py", line 153, in _glfw_get_version
    return eval(out)
SyntaxError: invalid character '（' (U+FF08)
```

**根因**：`sitecustomize` 在 python 启动**极早期**执行，任何 stdout 输出都可能被
第三方库的 `eval()`/subprocess 解析截获——本案例是 glfw 的版本探测用 `eval(out)`
读子进程输出，全角括号 `（` 触发了 SyntaxError。

**修复**：sitecustomize 和 shim 全部**静默化**——print 全删、异常吞掉（不能崩解释器）。
测试命令：
```bash
USE_TRT_HOOKS=1 PYTHONPATH=.../vitllm_infer python3 -c "import glfw; print('glfw OK')"
```

### 4.2 模型/engine dtype 不匹配（历史坑 + 现状）

**历史坑**：早期 engine IO 是 fp16，直接塞 fp32 tensor 给 fp16 engine 输入地址会崩（指针指向 fp32 数据但 engine 按 fp16 读）。当时修复：TrtEngine 按 `engine.get_tensor_dtype(name)` 自动 cast 输入；LLM hook 第 28 层补跑前把 engine 输出 cast 回模型 dtype。

**2026-08-31 现状**：ViT/LLM engine 的 IO 统一改为 **fp32:chw**（对齐真实链路），输入直接喂 FP32（`pixel_values.float()`），输出也是 FP32，无需 cast。但 TrtEngine 仍按 engine 实际 dtype 自动分配输出（HALF→fp16, FLOAT→fp32），保持通用性。

### 4.3 engine 路径（2026-08-31 更新为 V1+L2 选定组合）

- ViT **V1**：`vit/vit_engine_384/fp16onnx_fp16engine/vit_qwen3vl_384_fp16.plan`（fp16onnx→fp16engine，rotary fp16，fp32 IO）
- LLM **L2**：`llm_engine/fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan`（fp32onnx→fp16engine，RMSNorm fp32，fp32 IO）
- 旧路径已废弃：ViT `fp16_rotaryinner_fp32/`、LLM `fp16_fp32_layers/`

---

## 五、验证结果

### 5.1 组件级（run_torch.py / run_trt.py，vs 真实链路截取 orig_hidden_neg2，abs_max≈14750，2026-08-31 V1+L2 重测）

| 链路 | vs 真实链路 | 判定 |
|------|-----------|------|
| torch（fp32 模型完整 forward） | 0.0 ULP | ✅ |
| **TRT 全替换**（V1 ViT + L2 LLM，均 fp32 IO） | **2.9 ULP (0.16%)**，耗时 17.1 ms/iter | ✅ 可用 |
| TRT vs torch | 2.9 ULP | ✅ |

**15 组合全排列测试**（combo_test_vit_llm.py，ViT 5×LLM 3）：

- **L1（LLM RMSNorm fp16）全灭**：任何 ViT 组合都是 835（104 ULP）——RMSNorm 强制 FP32 是 LLM 硬性要求
- **精度最优**：V5+L3（双全 FP32）= 5.0（0.6 ULP，但 27ms 最慢）；V5+L2 = 13.6（1.7 ULP，22ms）
- **FP16 档最优**：**V1+L2 = 23.1（2.9 ULP，17.6ms）**——性能优先首选
- **选定 V1+L2**：FP16 速度档 + RMSNorm FP32 精度保障，端到端误差可接受（见 5.2）

### 5.2 端到端（真实 hdf5 数据，torch 基线 vs TRT 代理 brain_data.pt，2026-08-31 V1+L2 重测）

> 对比方法：两次运行各用 BRAIN_DATA_PATH 指定不同保存路径（torch 基线 → brain_data_torch_v1l2.pt，
> TRT 代理 → brain_data_trt_v1l2b.pt），同一份真实数据下 diff 输出（diff_brain_data.py）。

| 输出 | torch abs_max | TRT abs_max | max_diff | 相对误差 |
|------|--------------|------------|----------|---------|
| images（VLM 隐藏状态） | 616.63 | 616.00 | 23.07 | 3.74% |
| images_cond（adaptor 后） | 207.28 | 207.17 | 8.42 | 4.06% |
| lang | 537.89 | 537.50 | 2.85 | 0.53% |
| lang_cond | 114.19 | 114.21 | 0.67 | 0.59% |
| states / 指示器 / states_cond | — | — | 0.0000 | 0% |
| geomap / affordance | — | — | 0 | — |

**下游动作指标**（sync_full_metrics，170 步，V1+L2）：

| 指标 | torch 基线 | TRT 代理 | 变化 |
|------|-----------|---------|------|
| right_armqpos mae | 0.038726 | 0.038812 | +0.2% |
| right_eefgripper mae | 0.019515 | 0.019690 | +0.9% |
| headqpos mae | 0.019168 | 0.019223 | +0.3% |
| waistqpos mae | 0.026437 | 0.026514 | +0.3% |

**结论**：TRT 代理端到端可用（V1+L2）——states/指示器 bit 级 0，images 3.7% 扰动经 adaptor 后对最终动作影响 <1%。

### 5.3 性能（2026-08-31 V1+L2）

| 链路 | VLM forward | 备注 |
|------|------------|------|
| torch 基线（真实链路） | 0.05s | 日志实测 |
| **TRT 代理（V1+L2）** | **0.01s** | **5 倍加速** |
| TRT 链路组件级（run_trt.py） | 17.1 ms/iter | 完整 VLM forward（含 ViT+LLM engine） |

---

## 六、遗留问题

1. **MAX_SEQ=332 限制**：build_llm_engine.py 已把 LLM engine 的 seq 上限收紧到 332
   （文本+图像总长），超过会 shape 越界（当前真实 316，余量 16）。
2. **部署集成**：当前是环境变量开关（USE_TRT_HOOKS + TRT_HOOK_ENTRY），正式部署可改为配置文件/启动参数。
3. **精度与速度权衡**：当前选 V1+L2（FP16 档最优，23 ULP / 17.6ms）；如需更高精度可选 V5+L2
   （13.6 ULP / 22ms）或 V5+L3（5 ULP / 27ms），需按实际任务精度要求权衡。
4. **15 组合测试脚本**：`combo_test_vit_llm.py` 可复用——新增 engine 时扩展 VIT_ENGINES / LLM_ENGINES 列表即可。
