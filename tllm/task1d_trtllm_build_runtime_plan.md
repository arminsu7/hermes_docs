# 任务1D：修改 TRT-LLM build + runtime 两端跑 VLA

> 容器：whh_vla_trt2（conda py310）/ whh_trtllm_r130r20（TRT-LLM 环境）
> 代码目录：/root/workspace/embodichain/hpc_opt/trtllm/
> 不修改原始 dexechain/ 代码
> 所有 engine 导出统一使用 checkpoint-46000 权重

---

## 一、环境信息

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA RTX 5090 D v2 (24 GB) |
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| TensorRT | 10.15.1.29 |
| TensorRT-LLM | 1.3.0rc20 |
| Transformers | 5.5.4 |
| ONNX | 1.22.0 |
| onnxruntime | 1.17.1 |
| numpy | 1.26.4 |

容器 whh_vla_trt2 的 LD_LIBRARY_PATH 设置：
```bash
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 二、前置问题修复

### 2.1 问题：torch.compile (Dynamo) crash

**现象**：执行 `bash scripts/infer_realdata.sh` 直接报错退出。

**报错**：
```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <built-in method arange>(*(Max(u1, u2, u4, u5),), ...)
got ValueError('too many values to unpack (expected 2)')
```

**根因**：

1. 脚本 `infer_realdata.sh` 硬编码 `TORCH_COMPILE=1`
2. `compile_model()` 把 VLM（Qwen3-VL）用 `torch.compile(mode="default")` 包起来
3. Dynamo 追踪 ViT forward → `modeling_qwen3_vl.py:103` 的 `torch.arange(seqlen)`
4. `seqlen` 是 symbolic shape 表达式 `Max(u1, u2, u4, u5)`（4 个操作数，来自 grid_thw 维度计算）
5. PyTorch 2.7.0 的 `symbolic_shapes.py:5830` 的 `simplify()` 执行 `a, b = atom.args`，硬编码只处理 2 操作数的 Max
6. 4 个操作数 → `ValueError: too many values to unpack (expected 2)`

这是 PyTorch 2.7.0 的已知 bug：`simplify()` 没考虑 Max 有超过 2 个操作数的情况。

**修复**：

`scripts/infer_realdata.sh` 里 `TORCH_COMPILE=1` → `TORCH_COMPILE=0`

```diff
- TORCH_COMPILE=1    # 1=enable torch.compile for cerebellum+adaptors (first run ~30s overhead)
+ TORCH_COMPILE=0    # 1=enable torch.compile for cerebellum+adaptors (first run ~30s overhead)
```

**验证**：关掉 compile 后推理正常跑完，100 iter，VLM forward ~0.05s/iter，sync_full_metrics 正常输出。

---

## 三、方案C：修改 TRT-LLM build + runtime 两端（待实施）

### 3.1 核心思路

1. **build 端**：在 `trtllm-build` 构建过程中，于 layer 26 的输出处 `network.mark_output()`，标记为 engine output
2. **runtime 端**：修改 GenerationSession 读取这个新 output tensor 并返回

### 3.2 好处

能走完整的 TRT-LLM build 线路，engine 里 28 层都有 `gpt_attention_plugin` 优化（fused attention kernel、KV cache 等），比 trtexec 裸导出可能快不少。

### 3.3 需要改的代码路径

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

### 3.4 背景：为什么需要 hidden_states[-2]

VLA 推理不需要文本生成（token ids），而是需要 LLM 的中间层 hidden states 来驱动后续 cerebellum + adaptors。

Qwen3-VL 的 `capture_outputs` 装饰器有 `tie_last_hidden_states=True`，会把 `hidden_states[-1]` 替换为 `last_hidden_state`（经过 final norm 后的值）。所以 `hidden_states[-2]` 实际是 **layer 26 的输出**（norm 前），不是 layer 27。

### 3.5 背景：DeepStack 问题

Qwen3-VL 的 ViT 不只输出最终 pooler_output，还输出 3 个 deepstack_features（来自 ViT layer 5/11/17）。LLM 的 forward 需要：
- `inputs_embeds`：embed_tokens(input_ids) → masked_scatter(image_embeds)
- `visual_pos_masks`：image token 位置 mask
- `deepstack_visual_embeds`：3 个中间层特征，注入 LLM 对应层（layer 0/1/2 后）

TRT-LLM 的 Qwen3-VL model 实现里没有写 deepstack 逻辑。官方 serve.py 的流程是：
```
ViT engine → pooler_output [2040, 2048]
  → prompt_embedding_table（通过 fake token ID 映射到 embedding）
  → LLM engine (GenerationSession) → 文本生成
```
完全没有 deepstack。

如果走 TRT-LLM LLM engine 路线，DeepStack 被跳过，精度可能有损失，需要评估影响。

### 3.6 已有调研结论

| 方案 | 状态 | 说明 |
|------|------|------|
| 方案B：TRT-LLM LLM() API / GenerationSession | 已排除 | GenerationSession 不支持提取 hidden_states[-2] |
| 方案A：LLM ONNX 导出 + trtexec 转 engine | 已完成，数值未对齐 | FP16 逐层累积误差，16 层后超阈值 |
| 方案C：修改 TRT-LLM build + runtime 两端 | 待实施 | 本文档 |
| 方案D：GenerationSession context phase 提取 | 待尝试 | — |

---

## 四、进度记录

（待补充）
