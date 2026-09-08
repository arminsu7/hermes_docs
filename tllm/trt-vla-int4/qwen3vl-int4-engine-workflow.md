# Qwen3-VL LLM 模块 INT4 engine 生成工作流

> 环境：TensorRT 10.11（trtexec），已有 bf16 safetensors → fp16 engine 的成功经验
> 目标：用已有 int4_awq safetensors 生成 INT4 engine（无校准数据）

## 0. 核心结论（先看这个）

1. **`trtexec --int4` 不做量化**。读 v10.11 源码（`samples/common/sampleOptions.cpp`）确认：
   `--int4` 只设置 `BuilderFlag::kINT4`（允许 builder 用 INT4 kernel），
   不读 AWQ 权重、不算 scale、不做校准。INT4 权重必须已经以 **Q/DQ 节点**存在于 ONNX 中。
2. **没有校准数据 = 可以正常做**。INT4 AWQ 是 weight-only（W4A16，激活保持 FP16），
   scale 已固化在 safetensors 里。要做的是「复用已有 scale」，不是「重新校准」。
3. 完整链路：`int4_awq safetensors → (复用 scale) → QDQ ONNX → trtexec → INT4 engine`

```
int4_awq safetensors (scale 已固化)
        │  ① 判断格式（见 §2）
        ▼
ModelOpt 量化态模型 / TRT-LLM checkpoint
        │  ② 复用 scale，导出 QDQ ONNX（opset 20+）
        ▼
qwen3vl_llm_int4.onnx  (含 Q/DQ 节点，权重 INT4，激活 FP16)
        │  ③ trtexec 构建
        ▼
qwen3vl_llm_int4.engine
```

## 1. 为什么 trtexec --int4 行不通（证据）

v10.11 `sampleOptions.cpp` 关键代码：

```cpp
// line 1246
getAndDelOption(arguments, "--int4", int4);   // 只是读个 bool

// line 1256-1273：如果 --stronglyTyped，反而会把 --int4 禁用掉！
if (stronglyTyped) {
    disableAndLog(fp16, "fp16", "kFP16");
    ...
    disableAndLog(int4, "int4", "kINT4");     // ← --int4 + --stronglyTyped 会静默禁用
}
```

结论：
- `--int4` 是**隐式量化时代**（非 QDQ）的遗留开关
- 对含 QDQ 的 ONNX，TRT 自动进入 explicit quantization，精度由 QDQ 节点决定
- **正确的 trtexec 命令不需要 --int4**，直接 build QDQ ONNX 即可

## 2. 第一步：判断你的 int4_awq safetensors 格式

```bash
# 在能读到 safetensors 的机器上
python3 - <<'EOF'
from safetensors import safe_open
import sys
path = sys.argv[1]
with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
print(f"共 {len(keys)} 个 tensor")
for k in keys[:30]:
    print(k)
EOF
```

| 格式 | key 特征 | 后续路径 |
|------|---------|---------|
| **ModelOpt/TRT-LLM 原生** | `weight` + `weight_scale`（每层两个 key，可能带 `zeropoint`/`g_idx`） | §3-A：直接复用 scale 出 QDQ ONNX |
| **AutoAWQ（HF 常见）** | `qweight`（INT32 packed）+ `qzeros` + `scales` + `g_idx` | §3-B：需转换或换路径 |
| **GPTQ** | `qweight` + `qzeros` + `scales` + `g_idx`（打包不同） | §3-B |

## 3. 第二步：两条路径

### 路径 A：ModelOpt 复用 scale → QDQ ONNX → trtexec（贴合你现有 ONNX 链路）

前提：safetensors 是 ModelOpt 格式，或 ModelOpt 能直接加载。

```python
# export_int4_onnx.py —— 用你现有的 LLM wrapper（bf16→fp16 那条链路的 wrapper）
import torch
from transformers import AutoModelForCausalLM

# 1. 加载 int4_awq 模型（不重新量化）
model = AutoModelForCausalLM.from_pretrained(
    "<int4_awq模型目录>", torch_dtype=torch.float16, low_cpu_mem_usage=True
)

# 2. ModelOpt 初始化 INT4_AWQ 量化态 + 复用已有 scale（跳过校准！）
from modelopt.torch.quantization import initialize, load_state_dict_inplace
from modelopt.torch.quantization.config import QuantizeConfig

model = initialize(model, QuantizeConfig(quant_algo="INT4_AWQ"))
load_state_dict_inplace(model, "<int4_awq模型目录>")   # ← 关键：复用 checkpoint 里的 scale

# 3. 取 LLM 模块（你已有的 wrapper，参考技能里 LLM GQA 导出经验）
llm_module = model.model.language_model  # 或你自己的 LLM wrapper

# 4. 导出 QDQ ONNX（opset 20+，dynamo=True 解决 GQA）
torch.onnx.export(
    llm_module,
    (input_ids, attention_mask, position_ids),   # 你的真实输入
    "qwen3vl_llm_int4.onnx",
    opset_version=20,
    dynamo=True,        # 技能实测：dynamo=True 才能正确导出 GQA
    dynamic_axes={...}, # 保持你 fp16 导出时的 dynamic_axes
)
```

然后构建（**不需要 --int4**）：

```bash
trtexec \
    --onnx=qwen3vl_llm_int4.onnx \
    --saveEngine=qwen3vl_llm_int4.engine \
    --minShapes=... --optShapes=... --maxShapes=...   # 沿用你 fp16 的 shape 区间
```

### 路径 B：AutoAWQ/GPTQ 格式 → 换思路

AutoAWQ 的 INT4 是 **group-wise(128) 打包进 INT32**，与 TRT 的 INT4 block-quant WoQ 表达不同，
没有现成工具直接转 QDQ ONNX。两个选择：

1. **用 ModelOpt 重新量化**（需要校准数据，你没有 → 不推荐）
2. **如果目标是完整 LLM 服务**：用 TensorRT-LLM，`convert_checkpoint.py` 原生吃
   AutoAWQ 的 `int4_awq` 格式（`--quantization int4_awq`），不需要校准数据
   （注意：Qwen3-VL 在 TRT-LLM 里目前只有 PyTorch backend 支持，见 §5 风险）

## 4. 精度验证（必做！）

你的技能里有血的教训：**之前 dynamo 导出完整 LLM ONNX（2385 节点）→ trtexec FP16 engine 数值全错（max_diff=14928）**，onnxruntime 正常但 TRT 全错。INT4 量化误差只会叠加上去。

```python
# 用真实输入对比：PyTorch int4_awq 模型 vs TRT int4 engine
# 参考你 fp16 验证脚本，报告 max_diff + mean_diff
# 阈值建议：INT4 比 FP16 放宽（AWQ 的 INT4 通常 max_diff < 0.1 量级可接受）
```

## 5. 关键风险清单

| 风险 | 说明 |
|------|------|
| dynamo LLM ONNX 数值错 | 技能记录：TRT FP16 max_diff=14928，待排查 RMSNorm FP32 upcast Cast 节点。INT4 前先确认你 fp16 engine 数值是对的 |
| `--int4` + `--stronglyTyped` | 源码确认：stronglyTyped 会禁用 --int4。QDQ ONNX 不需要加 --int4 |
| Qwen3-VL TRT-LLM 支持 | 传统 convert_checkpoint+trtllm-build 对 Qwen3-VL 是 LEGACY；1.3.x 只有 PyTorch backend |
| opset | INT4 QDQ 需要 opset 20+（ModelOpt 默认 20；INT4 需 21+ 视算子） |
| 硬件 | INT4 WoQ 在 Ampere+ 可用（你的 RTX 3060 / SM86 没问题） |
| 权重打包格式 | AutoAWQ 的 INT32-packed 与 TRT block-quant 不同，直接塞不进 QDQ，必须先走 ModelOpt 或转换 |

## 6. 决策速查

```
你的 int4_awq safetensors 是什么格式？
├─ ModelOpt/TRT-LLM 原生 (weight+weight_scale)
│   → 路径 A：load_state_dict_inplace 复用 scale → QDQ ONNX → trtexec ✅ 最贴合你现有链路
├─ AutoAWQ (qweight/qzeros/scales/g_idx)
│   ├─ 只要 LLM 模块 engine → 无现成转换，需 ModelOpt 重新量化（要校准数据）⚠️
│   └─ 要完整 LLM 服务 → TensorRT-LLM convert_checkpoint（原生吃 int4_awq，无需校准）✅
└─ GPTQ
    → 同上 AutoAWQ，ModelOpt 有 GPTQ 支持但同样需要确认格式兼容
```
