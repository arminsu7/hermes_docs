# ModelOpt vs pytorch_quantization 框架选型调研

> 目标：为「AGX Orin / Orin NX / RTX3060/3050/4090/5050/5090」上的多类模型（qwen 系列、SAM3、机器人模型、YOLO、IGEV 视差模型）做量化框架选型，并解决「快速量化部署 + 手动微调 QDQ + Orin INT8 + 高精度模型」的混合需求。

---

## 0. 结论速览（TL;DR）

**结论：以 NVIDIA ModelOpt 为主选，pytorch_quantization 仅在维护老 TRT 8.6 项目时保留。**

- **ModelOpt 是 NVIDIA 官方现在唯一推荐的量化入口**，官方明说它 **replaces（取代）了已废弃的 PyTorch / TensorFlow Quantization Toolkit**，其中就包括 pytorch_quantization。
- **pytorch_quantization 已停止开发**，官方 README 明确「development has transitioned to the TensorRT Model Optimizer」，代码保留但不再有后续开发。
- **手动微调 QDQ 的能力，两个框架都具备，但 ModelOpt 更现代、更灵活**——通过 `quant_cfg` 精确控制每个量化器（禁用某层、改 per-tensor/per-channel、改算法），且支持导出 QDQ-ONNX 供 TensorRT 消费。
- **快速量化**：ModelOpt 的 `mtq.quantize(model, cfg, forward_loop=calib)` 一行完成量化+校准，官方推荐 128–512 个校准样本。
- **Orin INT8**：ModelOpt 有 aarch64（Jetson）wheel，配合 TensorRT-Edge-LLM 提供完整量化→ONNX→TRT 流水线。**但要注意**：INT8 在 Jetson 上并不总比 FP16 快（小模型/条件流可能反而更慢），且精度敏感模型需要精调。
- **高精度模型（IGEV 视差/SAM）**：两个框架都支持「跳过关键层量化」或「QAT」来保精度；ModelOpt 的 `quant_cfg` 做这件事比 pytorch_quantization 更顺手。

**一句话选型**：新项目、RTX 30/40/50 系、Orin 上量化部署、要快速量化又能手动微调 → **全部走 ModelOpt**；只有当你的旧 TRT 8.6 管线已经用 pytorch_quantization 写好且能跑，才继续保留它，同时规划迁移。

---

## 1. 两个框架的官方定位与现状

### 1.1 pytorch_quantization

| 项目 | 内容 |
|------|------|
| 官方位置 | NVIDIA/TensorRT 仓库的 `tools/pytorch-quantization/` 子目录 |
| 文档 | https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/ |
| 现状 | **已停止开发，官方弃用** |
| 最后版本 | 2.2.1（2023 年末，需从 pypi.nvidia.com / NGC 源安装；PyPI 上的 pytorch-quantization 2.2.1 是**假包/警告包**，提醒用户去装正确版本） |
| 替代者 | NVIDIA ModelOpt（TensorRT Model Optimizer） |

**官方弃用证据：**

1. pytorch_quantization 官方 README（NVIDIA/TensorRT 仓库 main 分支）原文：
   > "Pytorch Quantization development has transitioned to the TensorRT Model Optimizer. All developers are encouraged to use the TensorRT Model Optimizer to benefit from the latest advancements on quantization and compression. While the Pytorch Quantization code will remain available, it will no longer receive further development..."
   来源：https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/README.md

2. TensorRT 官方架构文档：
   > ModelOpt "Compresses models for TensorRT-LLM or TensorRT deployment. **Replaces the deprecated PyTorch and TensorFlow Quantization Toolkits.**"
   来源：https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html

3. NVIDIA TensorRT GitHub issue #3994（官方回复，2024-07-10）：
   > "TRT modelopt include pytorch-quantization. ModelOpt PyTorch quantization is **refactored based on pytorch_quantization**. Key advantages offered by ModelOpt's PyTorch quantization: 1) Support advanced quantization formats, e.g., Block-wise Int4 and FP8..."
   来源：https://github.com/NVIDIA/TensorRT/issues/3994
   （评论区：https://githubissues.com/NVIDIA/TensorRT/3994 ）

> 🔑 **一句话理解**：ModelOpt 的 PyTorch 量化 = pytorch_quantization 的**重构升级版**（同源、同思路），但补齐了 FP8/INT4 blockwise、AWQ/SmoothQuant/SVDQuant 等现代算法。ModelOpt 于 2024-06-13 公开发布，此后成为官方主推；其 `modelopt.torch.quantization` 命名空间与 pytorch_quantization 的 TensorQuantizer/QuantConv2d 等 API 一脉相承，**迁移成本低**。

**安装命令（避免假包陷阱）：**
```bash
# ❌ 错误：会装到 PyPI 的假包
pip install pytorch-quantization
# ✅ 正确：从 NGC 源装真包
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
# ModelOpt 正常装（无假包问题）
pip install nvidia-modelopt
```
证据：https://pypi.nvidia.com/pytorch-quantization/（真包，最后版本 2.2.1）

### 1.2 NVIDIA ModelOpt

| 项目 | 内容 |
|------|------|
| 官网 | https://nvidia.github.io/Model-Optimizer/ |
| GitHub | https://github.com/NVIDIA/Model-Optimizer |
| 包名 | `nvidia-modelopt`（旧名 TensorRT-Model-Optimizer） |
| 定位 | NVIDIA 统一的模型优化库：量化（PTQ/QAT）、蒸馏、剪枝、NAS、投机解码、稀疏化 |
| 输入 | HuggingFace / PyTorch / ONNX 模型 |
| 输出 | 量化 checkpoint、QDQ-ONNX、TensorRT-LLM 引擎、vLLM/SGLang 兼容格式 |
| 文档 URL | 官方文档存在 `Model-Optimizer/` 与旧 `TensorRT-Model-Optimizer/` 两套（内容一致） |

**证据：**
- 官网首页：https://nvidia.github.io/Model-Optimizer/
- GitHub README：https://github.com/NVIDIA/Model-Optimizer

---

## 2. 能力对比

### 2.1 PTQ（后训练量化）支持的格式

| 维度 | pytorch_quantization | ModelOpt |
|------|----------------------|----------|
| 经典 INT8（W8A8） | ✅ 支持（calibrator: max/histogram/percentile） | ✅ 支持 |
| INT8 SmoothQuant | ❌ | ✅ `INT8_SMOOTHQUANT_CFG` |
| FP8 | ❌（仅 INT8） | ✅ `FP8_DEFAULT_CFG` |
| INT4 blockwise | ❌ | ✅ |
| NVFP4（Blackwell） | ❌ | ✅ |
| W4A8 / W4A16 混合精度 | ❌ | ✅ |
| AWQ / SVDQuant | ❌ | ✅ |
| QAT（量化感知训练） | ✅（较弱，需手动接训练循环） | ✅（`examples/cnn_qat`，权重自适应、scale 冻结） |
| 校准算法 | max / histogram (percentile, mse, entropy) | max / AWQ / SmoothQuant / SVDQuant |

**证据：**
- ModelOpt 量化格式表：https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.config.html
- ModelOpt quant_cfg 文档：https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html
- ModelOpt CNN QAT 示例：https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat
- ModelOpt 预设配置目录：https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt_recipes/configs/ptq/presets

### 2.2 QDQ 手动微调能力（用户核心需求）

**关键问题：能不能手动微调 QDQ 节点？两个都能，ModelOpt 更现代。**

- **pytorch_quantization**：通过 `TensorQuantizer`、`QuantConv2d/QuantLinear` 等替换层，可手动控制每层量化；导出的 ONNX 带 Q/DQ 节点（`torch.onnx.export` 后得到 QDQ 格式，TensorRT 8.0+ 可导入）。
  证据：README "The quantized model can be exported to ONNX and imported by TensorRT 8.0 and later."

- **ModelOpt**：通过 **`quant_cfg`（quantization config）** 精确控制：
  - 哪个量化器激活、格式、per-tensor/per-channel、校准算法
  - **禁用指定层量化**（append disable entry，路径匹配，优先级最高）
  - 自定义 quantizer 配置
  这是它比 pytorch_quantization 强的地方——手动微调 QDQ 的粒度更细、语法更干净。
  证据：
  - quant_cfg 文档：https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html
  - PyTorch 量化指南：https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html

**导出 QDQ-ONNX 供 TensorRT 消费：**
- ModelOpt 支持 `mtq.export_onnx(...)` 导出 QDQ-ONNX，TensorRT 检测到 Q/DQ 后走显式量化路径构建引擎。
  证据：Torch-TensorRT 文档 "ModelOpt inserts quantize/dequantize (QDQ) nodes into the model graph; Torch-TensorRT then converts those nodes into TRT quantization layers" — https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html
- ModelOpt 还有独立的 **ONNX 量化路径**（`modelopt.onnx.quantization.int8`），直接对 ONNX 生成 QDQ 节点，遵循 TensorRT 规则。
  证据：https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html

### 2.3 快速量化 API

**ModelOpt 明显更"一行搞定"：**

```python
import modelopt.torch.quantization as mtq

# 一行：量化 + 校准（config 指定层/格式/校准算法，forward_loop 喂校准数据）
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calib_loop)
# 或 FP8
model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calib_loop)
# 导出 QDQ-ONNX：ModelOpt 没有独立 export_onnx，用标准 torch.onnx.export
import torch
torch.onnx.export(model, sample_input, "model_int8.onnx",
                  opset_version=20, dynamo=False)  # opset>=13 才支持 QDQ
```

官方原话：*"With the simple API below, you can very easily use Model Optimizer to quantize your model... using a small dataset (typically 128-512 samples) to calibrate the quantization scaling factors."*
来源：https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq/

pytorch_quantization 的 PTQ 需要手动：替换模块 → 设置 calibrator → 跑 forward 收集直方图 → compute_amax → 冻结 → 导出。步骤更多、更繁琐。

> ⚠️ **导出澄清**：ModelOpt 对 PyTorch 模型**没有独立的 `export_onnx` 函数**。官方文档明确 "After PTQ, the model can be exported to ONNX with the normal PyTorch ONNX export flow"——QDQ 就是普通 PyTorch 模块，用标准 `torch.onnx.export` + `opset>=13`（官方用 20）+ `dynamo=False` 即可。来源：https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html

### 2.4 硬件支持范围（用户硬件映射）

| 硬件 | pytorch_quantization | ModelOpt |
|------|----------------------|----------|
| RTX 30 系（Ampere, SM 8.6）：3060/3050 | ✅ INT8 | ✅ INT8（Ampere INT8 有效，官方称 A100 INT8 比 BF16 提升 1.4–1.6x） |
| RTX 40/50 系（Ada/Blackwell）：4090/5050/5090 | ✅ INT8（但无 FP8） | ✅ INT8 + Blackwell 的 NVFP4/FP8 |
| AGX Orin / Orin NX（Jetson, SM 8.7） | ✅（但需从 NGC 源装，aarch64） | ✅ 有 aarch64 wheel（`nvidia_modelopt-core` manylinux aarch64），配合 TensorRT-Edge-LLM |
| DLA | 需手动走 cuDLA QAT 流程（**仅 TRT 8.x 官方验证**） | 需手动走 cuDLA QAT 流程（**TRT 10.x 后配合困难**，且 TRT 11 不支持 DLA） |

**关键证据：**
- ModelOpt 有 aarch64 wheel：https://pypi.org/project/nvidia-modelopt-core/（`nvidia_modelopt_core-0.33.1-cp311-cp311-manylinux_2_28_aarch64.whl`）
- TensorRT-Edge-LLM 官方原话：*"provides a complete pipeline for quantizing LLMs and VLMs using NVIDIA ModelOpt and exporting them to optimized ONNX for deployment on edge platforms such as NVIDIA Jetson and DRIVE."* Orin 支持 FP16/INT8/INT4。
- Ampere INT8 收益：NVIDIA 官方博客（A100 INT8 1.4–1.6x vs BF16）
- cuDLA YOLOv5 官方博客（Orin DLA INT8 全流程标杆）：https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/

> ⚠️ **Orin INT8 的重要警示**：INT8 在 Jetson 上**并不总是比 FP16 快**。实测案例：
> - Orin Nano 4GB ViT-S+DPT 架构：ModelOpt INT8 反而导致 **2.7x 性能回退**（https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835）
> - DEIMv2 在 Orin NX：INT8 量化后性能反而更差（https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549）
> - 原因：INT8 底层 kernel 收益有限、条件流/小模型开销大、反量化(reformat)节点增加。
> **结论**：在 Orin 上不要默认 INT8 一定快，要逐个模型实测对比 FP16 vs INT8。

### 2.5 与 TensorRT / TensorRT-LLM 的关系

- **pytorch_quantization**：只服务于经典 TensorRT（导出 QDQ-ONNX 给 trtexec/TensorRT 建引擎）。与 TensorRT-LLM 无直接集成。
- **ModelOpt**：
  - 对经典 TensorRT：导出 QDQ-ONNX（`mtq.export_onnx` / ONNX 量化路径）
  - 对 TensorRT-LLM：`quantize.py` 产出校准好的 checkpoint → `trtllm-build` 建引擎
  - 对 vLLM/SGLang：导出兼容格式（`--quantization fp8` 或 modelopt）
  证据：https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_ptq/README.md

---

## 3. 针对用户场景的选型建议

### 3.1 按需求分类

| 你的需求 | 推荐 | 理由 |
|----------|------|------|
| **快速量化部署**（新项目） | **ModelOpt** | 一行 `mtq.quantize`，128-512 样本校准，支持 PTQ+QAT，多格式 |
| **手动微调 QDQ 节点** | **ModelOpt**（`quant_cfg`） | 细粒度控制每层量化、禁用敏感层，比 pytorch_quantization 干净 |
| **Orin 上 INT8** | **ModelOpt** | aarch64 wheel + TensorRT-Edge-LLM 完整流水线；但需实测 FP16 vs INT8 |
| **fp16 够用即可** | 直接用 TensorRT FP16，无需量化框架 | 最省事，TensorRT 对 FP16 有原生 Tensor Core 支持 |
| **高精度模型（IGEV 视差/SAM）** | **ModelOpt** + 跳过敏感层 / QAT | `quant_cfg` 禁用关键层量化，或 QAT 恢复精度 |

### 3.2 按模型分类

| 模型 | 平台 | 建议 |
|------|------|------|
| **qwen 系列（LLM）** | RTX 4090/5090 | ModelOpt + TensorRT-LLM（FP8 on 5090 / INT8/INT4 on 4090）|
| **qwen 系列（LLM）** | Orin | ModelOpt + TensorRT-Edge-LLM（FP16/INT8/INT4）|
| **SAM3（分割）** | RTX / Orin | ModelOpt INT8 + `quant_cfg` 跳过 image encoder 敏感层或 QAT（分割对精度敏感）|
| **YOLO（检测）** | RTX / Orin | ModelOpt INT8（Ultralytics 已默认集成 ModelOpt）|
| **IGEV（视差，高精度）** | Orin NX / AGX | ModelOpt，但**强烈建议跳过 cost volume / 迭代细化层量化，或 QAT**；对比 FP16 基线 |
| **机器人模型** | Orin | 视具体架构，ModelOpt 通用，注意延迟敏感用 FP16 基线 |

### 3.3 决策树

```
模型要部署在哪？哪个 TRT 版本？
├── RTX 30/40/50 系 (x86)  → ModelOpt（FP8/INT8/INT4 全支持），vLLM 或 TensorRT-LLM
├── AGX Orin / Orin NX (aarch64)
│     ├── 精度敏感（IGEV/SAM）→ ModelOpt + 跳过敏感层/QAT，FP16 作基线
│     ├── LLM → ModelOpt + TensorRT-Edge-LLM
│     └── 常规 CNN → ModelOpt INT8，但先实测 FP16 vs INT8
└── 旧 TRT 8.6 存量管线（已用 pytorch_quantization 写好且能跑）
      └── 保留 pytorch_quantization，同时规划迁移到 ModelOpt
```

### 3.4 按 TRT 版本细化（重要）

| 环境 | 推荐 | 说明 |
|------|------|------|
| **Orin + TRT 8.6.1.6**（你现有） | **pytorch_quantization 手动微调 QDQ** | 最贴合现有流程、完全可控；输出/refinement/视差回归层留 FP16；DLA 走 cuDLA Q/DQ Translator 转 PTQ |
| **Orin + TRT 10.3**（你新环境） | **ModelOpt**，但**固定用较保守版本**（0.31/0.35 已验证能编译，0.40/0.41 在默认 TRT 10.3 有 bug） | 建引擎用 `--stronglyTyped`；ModelOpt ONNX 量化要求 TRT ≥10.0 |
| **RTX 4090/5090** | ModelOpt 首选 | 支持 FP8/INT8/SmoothQuant；`auto_quantize`/`autotune` 自动平衡精度与性能 |
| **DLA** | 谨慎 | TRT10+ 上 DLA INT8 依赖已废弃的 implicit 量化；Orin 优先 GPU + ModelOpt 显式 QDQ；只有 TRT 8.6 DLA 才推荐 QAT 路线 |

**关键版本事实：**
- ModelOpt ONNX 量化（calibration EP）**要求 TRT ≥ 10.0**；但 TRT 8.6 本身已支持显式 QDQ（issue #4079 确认）→ TRT 8.6 能消费 ModelOpt/pytorch_quantization 生成的 QDQ ONNX
- TRT 10.3 建引擎用 `trtexec --onnx=quant.onnx --stronglyTyped`；**别用 `--best`**（会自动走已废弃的 implicit INT8）
- TRT 8.6 上自定义加法层量化精度坑：需手动设置量化定义（issue #4079）
- ModelOpt 官方系统要求：x86_64 + aarch64(SBSA)，PyTorch ≥2.8，TRT ≥10.0（可选）——**aarch64 官方声明支持**

证据：
- ModelOpt Installation：https://nvidia.github.io/Model-Optimizer/getting_started/_installation_for_Linux.html
- TRT 显式量化文档：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html
- TRT 8.6 精度坑：https://github.com/NVIDIA/TensorRT/issues/4079

---

## 4. 关键坑与提示（含 Orin/DLA 实战）

1. **别在 PyPI 上装 pytorch-quantization**：PyPI 上的同名包是**假包/警告包**（2.2.1, 2023-11-03 发布，专门提醒用户装错包）。真包在 pypi.nvidia.com（NGC 源），最后版本 2.2.1（2023 年末）。ModelOpt 走 `pip install nvidia-modelopt` 即可，无此坑。
   证据：https://pypi.org/project/pytorch-quantization/ （"A fake package to warn the user they are not installing the correct package"）

2. **Orin INT8 未必更快**：小模型/条件流/精度敏感模型，INT8 可能反而更慢或掉精度。逐个实测。实测案例：
   - Orin Nano ViT-S+DPT：ModelOpt INT8 导致 **2.7x 回退**
   - Orin NX DEIMv2：INT8 后性能更差
   - 原因：底层 INT8 kernel 收益有限、条件流/小模型开销、reformat 节点增加

3. **TensorRT 10.1+ 弃用隐式量化（IInt8Calibrator）**：TensorRT 11 直接移除。老教程的「校准器 + implicit INT8」路线在 TRT 10.1+ 已不推荐，应走 ModelOpt 的显式 QDQ 路线。这对你现有 TRT 8.6 环境（支持 implicit）和 10.3 环境（要走显式）都要注意。

4. **Jetson 无官方 ModelOpt 容器**：ModelOpt 官方推荐的 TensorRT-LLM / NGC PyTorch / NGC TensorRT 容器都是 **x86 数据中心镜像，不是 Jetson 容器**。Jetson 上需用 JetPack 自带环境 + `pip install nvidia_modelopt-core`（有 aarch64 wheel）。
   证据：https://pypi.org/project/nvidia-modelopt-core/

5. **ModelOpt 最新版可能与 TRT 10.3 编译失败**：实测（JetPack 6.2, Orin Nano）ModelOpt 0.40/0.41 在默认 TRT 10.3 上编译失败（`Assertion type() == expectedDataType<T>()`），官方回复"需等未来 TRT release 修复"。→ **用 ModelOpt 时务必匹配它要求的 TRT 版本**，不能随手升到最新。
   证据：https://forums.developer.nvidia.com/t/tensorrt-10-x-is-convtranspose3d-supported-in-int8-on-jetson-qat-workflow/353440

6. **Orin DLA 部署路径特殊**：
   - Orin DLA 只支持 **INT8/FP16**，**不支持 QAT 推理**，必须把 QAT 模型转成 PTQ（Q/DQ Translator 抽 scale → 导出去掉 Q/DQ 的 ONNX + 校准 cache）。
   - 官方验证过的 DLA 路径（cuDLA YOLOv5 博客）用的是 **pytorch_quantization QAT → Q/DQ Translator → PTQ**，**仅适用于 TRT 8.x**。
   - **TRT 10.x 之后 DLA 与 ModelOpt 的配合陷入困境**：TRT10 起废弃 implicit quantization（IInt8Calibrator），而 **DLA 用 INT8 必须走 implicit quantization**；FP16 在 Orin DLA 卷积上又因 FP19 问题很慢；TRT11 将彻底移除 implicit quantization → **DLA 在 TRT10+ 上两条精度路线都不划算**。→ 如果你要跑 DLA，要么留在 TRT 8.x + pytorch_quantization，要么接受 TRT 10.x DLA 支持受限的现实。
   证据：https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/
   DLA+TRT10 困境讨论：https://forums.developer.nvidia.com/t/is-there-a-plan-to-support-dla-on-the-next-tensorrt-version/313130
   另：TensorRT 11.0/11.1/11.2 不支持 DLA（官方 Best Practices），TRT 10.7 是最后一个完整支持 DLA 的版本。
   trtexec 命令（cuDLA YOLOv5）：`--useDLACore=0 --safe --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 --calib=qat2ptq.cache`

7. **校准数据集**：128–512 样本通常够，但要覆盖真实部署分布（对 IGEV 用真实双目图，对 SAM 用真实分割场景）。

8. **高精度模型（IGEV/SAM）**：优先「跳过关键层量化」而非全局 INT8；必要时上 QAT。ModelOpt 的 `quant_cfg` 就是干这个的。**用户已手动微调过 IGEV 的 QDQ → 说明快速 PTQ 精度不达标，应保留手动微调路线**（pytorch_quantization 在 TRT 8.6，或 ModelOpt quant_cfg 细粒度方案），并**把输出/refinement 层留 FP16**。

   **官方实证（cuDLA YOLOv5 博客）**：YOLOv5 在 Orin DLA 上 COCO mAP 从 FP32 37.4 → DLA INT8 37.3（几乎无损）；但**末尾 3 个卷积层改跑 FP16 时 mAP 从 35.9 → 37.1（+1.2 mAP）**，代价是 FPS 从 410 → 252。→ 结论：**输出/检测头/边界回归层对量化最敏感，保留 FP16 是保精度的核心手段**。证据：https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/

   **ModelOpt 官方跳层手段**（2026-05-07 官方 PTQ 博客）：
   - `mtq.disable_quantizer` 按正则匹配跳过特定层（示例跳过 `patch_embedding` 层）
   - ONNX 量化 CLI 的 `--op_types_to_exclude` / `--nodes_to_exclude`（支持正则）/ `--high_precision_dtype fp16`（未量化层保留 FP16）
   - 证据：https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer/ 和 https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html

   **IGEV 视差模型专项**：cost volume、soft-argmax 回归、亚像素视差输出层**绝不能 INT8**，head/refinement 走 FP16。证据（轻量立体匹配 INT8 论文）：https://www.csroc.org.tw/journal/JOC36-5/JOC3605-27.pdf

   **SAM 分割专项**：**AveragePooling 无法量化**，只能强制 FP16，导致 Reformat 层暴增、INT8 反而比 FP32 慢（5.6ms vs 4.6ms）。→ 分割模型应把池化/上采样/Upsample 层排除。证据：https://forums.developer.nvidia.com/t/post-training-quantization-ptq-for-semantic-segmentation-model-running-on-jetson-orin-nx/316535

   **Transformer attention 专项**：ModelOpt 的 SDPA/MHA 默认不被 walker 拦截，需注册 `_QuantAttention` 插件手动插 4 个 quantizer（q/k/v/bmm2）。FP8 博客实证 CLIP attention 必须显式处理才能保证质量。

9. **文档 URL 两套**：`Model-Optimizer/` 与旧 `TensorRT-Model-Optimizer/` 内容一致，别因 404 混淆。

---

## 5. 主要证据来源汇总

### 官方（强证据）
- pytorch_quantization README（弃用声明）：https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/README.md
- TensorRT 架构文档（Replaces deprecated toolkits）：https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html
- TensorRT GitHub issue #3994（ModelOpt 基于 pytorch_quantization 重构）：https://github.com/NVIDIA/TensorRT/issues/3994
- ModelOpt 官网：https://nvidia.github.io/Model-Optimizer/
- ModelOpt GitHub：https://github.com/NVIDIA/Model-Optimizer
- ModelOpt quant_cfg：https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html
- ModelOpt PyTorch 量化：https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html
- ModelOpt ONNX 量化：https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html
- ModelOpt 量化格式表：https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.config.html
- NVIDIA 官方 PTQ 博客（2026-05-07）：https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer/
- TensorRT-Edge-LLM（Jetson 流水线）：https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- ModelOpt aarch64 wheel：https://pypi.org/project/nvidia-modelopt-core/

### 社区/论坛（中强证据，实测案例）
- Orin Nano INT8 2.7x 回退：https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835
- Orin NX DEIMv2 INT8 更慢：https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549
- Orin NX 语义分割 PTQ：https://forums.developer.nvidia.com/t/post-training-quantization-ptq-for-semantic-segmentation-model-running-on-jetson-orin-nx/316535
- TensorRT 10.x ConvTranspose3d INT8 QAT（ModelOpt 0.40/0.41 + TRT 10.3 编译失败坑）：https://forums.developer.nvidia.com/t/tensorrt-10-x-is-convtranspose3d-supported-in-int8-on-jetson-qat-workflow/353440
- DLA 支持计划讨论：https://forums.developer.nvidia.com/t/is-there-a-plan-to-support-dla-on-the-next-tensorrt-version/313130
- PyPI pytorch-quantization 假包警告：https://pypi.org/project/pytorch-quantization/
- ModelOpt aarch64 wheel（Jetson 安装）：https://github.com/ajeetraina/jetson-orin-nano-super-guide
- YOLOv8 Jetson Orin Nano FP16 vs INT8（FP16 是 sweet spot）：https://hokwangchoi.com/blog/vision-benchmarks/
- NVIDIA cuDLA YOLOv5 QAT 博客（DLA 场景）：https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/
- NVIDIA QAT 恢复 FP32 精度博客：https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/

---

## 6. 待补充（research-deep 深挖项）

> ✅ 以下已通过第二轮 research-deep 深挖完成，详见第 7 章。原待办：
> - [x] ModelOpt 各默认 CFG 的完整字段定义 → 见 7.3
> - [x] 各模型（qwen/SAM3/YOLO/IGEV）在 ModelOpt 上的具体 recipe → 见 7.2/7.4/7.5
> - [x] IGEV/SAM 跳过敏感层的具体 quant_cfg 写法 → 见 7.1（附可运行示例）
> - [x] Orin DLA 与 ModelOpt 的具体配合细节 → 见 4.6
> - [x] TensorRT 8.6 环境与 ModelOpt 的兼容性边界 → 见 3.4

---

## 7. 深度调研：IGEV/SAM/YOLO/qwen 专项落地

> 本节来自第二轮 research-deep 深挖（3 个子代理，2026-08-12）。所有代码来自 ModelOpt 官方源码/config.py 核实，可直接抄。

### 7.1 IGEV 视差模型：quant_cfg 精确 API 与跳过敏感层（可运行示例）

**交付文件：** `/home/armin/repos/hermes/docs/free/modelopt_igev_ptq.py`（完整可运行脚本，含 5 个片段：quant_cfg API / 禁用层 / 混合精度 / 双目校准 forward_loop / 导出 QDQ-ONNX）

**quant_cfg 精确 API（源码核实）：**
- `QuantizeConfig`：`quant_cfg: list[QuantizerCfgEntry]`（默认 `[{"quantizer_name":"*","cfg":{"num_bits":8,"axis":None}}]`）+ `algorithm: str`（默认 "max"）
- `QuantizerCfgEntry` 字段：`quantizer_name`（必填，fnmatch 通配符）、`parent_class`（限父类如 `"nn.Conv2d"`）、`cfg`（属性完整替换）、`enable`（仅切开关）
- `QuantizerAttributeConfig`：`num_bits`（8=INT8，`(4,3)`=FP8 E4M3）、`axis`（None=per-tensor, 0=per-channel）、`trt_high_precision_dtype`（"Float"/"Half"/"BFloat16"）、`calibrator`、`block_sizes`
- **重要**：当前 `quant_cfg` 用 **list 格式**（后条目覆盖前条目、优先级更高），旧 dict 格式仍兼容
- IGEV 是 CNN → 官方明确：**CNN 只用 INT8**（`INT8_DEFAULT_CFG`）

**禁用指定层量化（IGEV 特征提取 conv）：**
```python
import copy, modelopt.torch.quantization as mtq
cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
# 禁用整个 feature_extractor 模块的量化
cfg["quant_cfg"].append({"quantizer_name": "*feature_extractor*", "enable": False})
# 禁用 stem 首个卷积（对极小输入敏感）
cfg["quant_cfg"].append({"quantizer_name": "*convs.0*", "enable": False})
# 只禁用某层"输入激活量化"，权重仍量化
cfg["quant_cfg"].append({"quantizer_name": "*stem_conv*input_quantizer", "enable": False})
```
或事后动态禁用：`mtq.disable_quantizer(model, filter_func)`（filter 用正则匹配模块名）。

**让某层保留 FP16（混合精度）两条路径：**
- **路径 A（推荐）**：量化器属性 `trt_high_precision_dtype: "Half"`——导出 QDQ-ONNX 时该层 QDQ 标记为 Half。**只对敏感层（如 cost volume / disp_head）设置**：
  ```python
  cfg["quant_cfg"].append({
      "quantizer_name": "*disp_head*",
      "cfg": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Half"},
  })
  ```
- **路径 B**：`enable: False` 彻底不量化该层（导出时无 QDQ，保持全精度）。

**校准 forward_loop（双目双输入）：**
```python
def forward_loop(model):
    for imgL, imgR in zip(calib_left_loader, calib_right_loader):
        imgL, imgR = imgL.cuda(), imgR.cuda()
        with torch.no_grad():
            model(imgL, imgR)   # IGEV: model(left, right)
```
校准数据子采样 128–512 张即可（官方建议，非全量）。

**导出 QDQ-ONNX：** 无独立 export_onnx，用标准 `torch.onnx.export(model, (imgL, imgR), "igev_int8.onnx", input_names=["left","right"], output_names=["disparity"], dynamic_axes={...}, opset_version=20, dynamo=False)`。双目输入传 tuple + dynamic_axes 标 batch。

> ⚠️ **默认禁用规则**：`default_disabled_quantizers.yaml` 已自动禁用 BatchNorm2d/LeakyReLU/Embedding——IGEV 特征提取常用的这些层默认不量化。源码：`modelopt_recipes/configs/ptq/units/default_disabled_quantizers.yaml`

### 7.2 SAM3 / SAM 分割模型：image encoder 最敏感，如何跳过

**结论：SAM 的 image encoder（ViT 主干）是量化最敏感的部分**（占 SAM2 base_plus 总参数 90%+，且线性层权重有离群值、激活重尾分布）。官方与社区一致推荐：**对编码器敏感层跳过 INT8 / 保持更高精度，或对编码器做 QAT**。

**跳过策略（由轻到重）：**
1. **最轻**：只跳过 patch-embed 主干（首个 3 通道 ≥7×7 大卷积）+ 把 LayerNorm 从 SmoothQuant 排除。**embedl SAM3 官方教程实证**：INT8 量化 encoder、跳过 patch-embed 主干，L4 上 12 QPS。证据：https://docs.embedl.com/embedl-deploy/latest/auto_tutorials/sam3.html
2. **中等**：`quant_cfg` 正则禁用敏感子模块（`*layer1.*`）或让整个 image encoder 走 FP16/FP32（NVIDIA-AI-IOT/nanosam：MobileSAM image encoder 连 FP16 都出错，改回 FP32）。证据：https://github.com/NVIDIA-AI-IOT/nanosam
3. **最重**：编码器 QAT（Q-SAM2 论文，PTQ 校准最高 +66% mIoU）。证据：https://arxiv.org/html/2506.09782v1

**ModelOpt 跳过层写法：**
```python
cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
cfg["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})  # 禁所有输入量化
cfg["quant_cfg"].append({"quantizer_name": "*layer1.*", "enable": False})          # 禁 layer1 正则匹配
```
**`mtq.auto_quantize`** 自动混合精度：高敏感层用 FP8、低敏感层用 NVFP4、极敏感层直接跳过。官方原文："AutoQuantize will automatically quantize highly sensitive layers in FP8_DEFAULT_CFG while keeping less sensitive layers in NVFP4_DEFAULT_CFG (and even skip quantization for any extremely sensitive layers)"。

### 7.3 YOLO 系列：Ultralytics 已集成 ModelOpt

**关键事实：Ultralytics 在 TensorRT 11+ 上用 ModelOpt 做 INT8 显式量化**（插入 Q/DQ 节点 → strongly-typed engine），FP16 用 ModelOpt AutoCast。官方原话：
> "TensorRT 11 removed implicit quantization and the IInt8Calibrator interface. On TensorRT 11 and newer, Ultralytics performs INT8 quantization with NVIDIA ModelOpt explicit quantization... ModelOpt is installed automatically on first use."
证据：https://docs.ultralytics.com/integrations/tensorrt

**官方命令：**
```bash
yolo export model=yolo26n.pt format=engine batch=8 workspace=4 quantize=8 data=coco.yaml
# 校准集默认取 val 集（COCO 5000 张）；GPU 默认 MINMAX 校准，Jetson DLA 用 ENTROPY_CALIBRATION_2
```
**注意**：校准缓存（.cache）**设备相关**，跨设备复用会导致校准差；小 batch 校准不准，尽量用大 batch。检测头（decoder/sigmoid）对量化敏感，INT8 可能偏移置信度，建议用 INT8 模型自己的 F1 曲线重选阈值。

### 7.4 qwen 系列 LLM：默认 CFG 字段定义 + RTX 选型

**ModelOpt 默认 CFG 完整字段定义（源码 YAML 核实）：**

| 常量 | 算法 | 权重 | 激活 | 格式 |
|------|------|------|------|------|
| `FP8_DEFAULT_CFG` | max | FP8 E4M3, per-tensor | FP8 E4M3, per-tensor | W8A8 FP8 |
| `INT8_DEFAULT_CFG` | max | INT8, **per-channel** (axis=0) | INT8, per-tensor | W8A8 INT8（CNN 用） |
| `INT8_SMOOTHQUANT_CFG` | smoothquant | INT8, per-channel | INT8, per-tensor | W8A8 + SmoothQuant（LLM 用） |
| `W4A8_AWQ_BETA_CFG` | awq_lite | INT4 block(128) + FP8 | FP8 per-tensor | W4A8（experimental，可能有较大精度损失） |
| `NVFP4_DEFAULT_CFG` | max | NVFP4 e2m1, block16 | NVFP4 e2m1, block16 | W4A4（Blackwell，scale=FP8 E4M3，每16元素一个scale, ~4.5bit/元素） |
| `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` | max | INT4 block(128) | 不量化 | W4A16（INT4 weight-only） |

**默认排除层**（所有配置通用）：`*lm_head*`, `*output_layer*`, `*router*`, MoE gate, `*linear_attn.conv1d*`, `*proj_out.*`, Embedding/BatchNorm/LeakyReLU, VLM 的 `*vision_tower*`/`*visual*` 等（官方避免量化视觉分支）。

**qwen 支持矩阵（官方 `examples/hf_ptq/README.md`）：**
| 模型 | fp8 | int8_sq | int4_awq | w4a8_awq | nvfp4 |
|------|-----|---------|----------|----------|-------|
| Qwen2, 2.5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen3, 3.5 MOE, Next | ✅ | ❌ | ❌ | ❌ | ✅ |

→ **Qwen3 只支持 FP8 和 NVFP4**（INT8/AWQ 不支持）；Qwen2/2.5 全格式支持。

**RTX 4090（SM89 Ada）推荐**：首选 **FP8（W8A8）**（基本 lossless）；要省显存跑大模型用 **INT4 AWQ**（W4A16，对 Qwen2/2.5）；**Qwen3 只能 FP8**（不支持 NVFP4）。不推荐 INT8 SmoothQuant（Ada 上收益不如 FP8）。
**RTX 5090（SM120 Blackwell）推荐**：首选 **NVFP4**（官方针对 Qwen/LLM 强烈推荐，2x 压缩），推荐用 `nvfp4_mlp_only` / `nvfp4_experts_only` 替代默认 `NVFP4_DEFAULT_CFG` 以获得更高精度（保留敏感 attention QKV）；次选 FP8 + FP8 KV cache。

**hf_ptq 命令（现代入口，旧 quantize.py 已弃用）：**
```bash
# FP8（Qwen3 推荐，含 FP8 KV cache）
python hf_ptq.py --pyt_ckpt_path <Qwen3模型> --qformat fp8 --kv_cache_qformat fp8_cast --export_path <path>
# INT4 AWQ（Qwen2/2.5 省显存）
python hf_ptq.py --pyt_ckpt_path Qwen/Qwen2.5-7B --qformat int4_awq --kv_cache_qformat fp8_cast --export_path <path>
# 导出后部署到 TRT-LLM / vLLM / SGLang（export_hf_checkpoint）
```

**ModelOpt ↔ TensorRT-LLM 版本对应（Release Notes 核实）：**
| TensorRT-LLM | 依赖 ModelOpt | TensorRT |
|------|------|------|
| 25.06 | 0.33 | ~10.8 |
| 25.10 | 0.37 | ~10.9 |
| 1.x (main) | 最新 | 10.9+ |

→ 官方建议**直接用 TensorRT-LLM docker 镜像**（内置 ModelOpt 版本与 TRT-LLM 匹配验证过），避免自行 pip 安装导致版本不兼容。NVFP4 需 TensorRT-LLM ≥0.17。

### 7.5 校准数据集规模（权威收敛值）

- **ModelOpt 官方 ONNX PTQ**：TensorRT 建议 CNN/ViT **至少 500 张**（INT4 则 64 张）。证据：https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq
- **Ultralytics**：NVIDIA 建议 **至少 500 张**代表性校准图。证据：https://docs.ultralytics.com/integrations/tensorrt
- **ModelOpt PyTorch PTQ**：128–512 样本。证据：https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html
- **内容**：必须与部署域匹配、多样、覆盖真实分布（视差模型要含各种视差范围；分割用 COCO/LVIS 或 SA-1B 子集）。校准/导出设备必须与部署设备一致。

### 7.6 落地建议速览

1. **先跑 INT8 全量化基线**（≥500 张代表性校准图，batch 尽量大）
2. **定位敏感层**：SAM 的 image encoder（patch-embed 主干/离群值层）、YOLO 的检测头（decoder/sigmoid）、IGEV 的 cost volume/视差回归——公认最敏感
3. **跳过策略由轻到重**：跳过 patch-embed 卷积 → `quant_cfg` 正则禁用敏感子模块 / ONNX `nodes_to_exclude` → 敏感段整体 FP16 → QAT 或 `auto_quantize` 兜底
4. **校准数据** ≥500 张、与部署域一致、用部署同设备

---

## 8. 参考文档（完整证据清单）

### 官方（强证据）
- pytorch_quantization README（弃用声明）：https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/README.md
- TensorRT 架构文档（Replaces deprecated toolkits）：https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html
- TensorRT GitHub issue #3994（ModelOpt 基于 pytorch_quantization 重构）：https://github.com/NVIDIA/TensorRT/issues/3994
- ModelOpt 官网：https://nvidia.github.io/Model-Optimizer/
- ModelOpt GitHub：https://github.com/NVIDIA/Model-Optimizer
- ModelOpt quant_cfg：https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html
- ModelOpt PyTorch 量化：https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html
- ModelOpt ONNX 量化：https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html
- ModelOpt 量化格式表：https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.config.html
- ModelOpt Installation（aarch64 支持、TRT≥10.0）：https://nvidia.github.io/Model-Optimizer/getting_started/_installation_for_Linux.html
- ModelOpt 预设配置 YAML：https://github.com/NVIDIA/Model-Optimizer/tree/main/modelopt_recipes/configs/ptq/presets/model
- ModelOpt hf_ptq（支持矩阵/命令）：https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md
- ModelOpt ONNX PTQ 示例（FAR3D 检测案例）：https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq
- ModelOpt config.py 源码：https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/config.py
- NVIDIA 官方 PTQ 博客（2026-05-07）：https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer/
- TensorRT-Edge-LLM（Jetson 流水线）：https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/
- ModelOpt aarch64 wheel：https://pypi.org/project/nvidia-modelopt-core/
- NVIDIA cuDLA YOLOv5 QAT 博客（DLA 场景）：https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/
- NVIDIA QAT 恢复 FP32 精度博客：https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/
- Ultralytics TensorRT + ModelOpt 集成：https://docs.ultralytics.com/integrations/tensorrt
- TensorRT-LLM 量化文档：https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- TensorRT-LLM Release Notes（版本依赖）：https://nvidia.github.io/TensorRT-LLM/release-notes.html

### 社区/论坛/论文（中强证据，实测案例）
- Orin Nano INT8 2.7x 回退：https://forums.developer.nvidia.com/t/tensorrt-model-optimizer-int8-quantization-causes-2-7x-performance-regression-on-jetson-orin-nano-4gb-vit-s-dpt-architecture/357835
- Orin NX DEIMv2 INT8 更慢：https://forums.developer.nvidia.com/t/worse-performance-after-quantization-on-tensorrt/355549
- Orin NX 语义分割 PTQ：https://forums.developer.nvidia.com/t/post-training-quantization-ptq-for-semantic-segmentation-model-running-on-jetson-orin-nx/316535
- TensorRT 10.x ConvTranspose3d INT8 QAT（ModelOpt 0.40/0.41 + TRT 10.3 编译失败坑）：https://forums.developer.nvidia.com/t/tensorrt-10-x-is-convtranspose3d-supported-in-int8-on-jetson-qat-workflow/353440
- DLA 支持计划讨论：https://forums.developer.nvidia.com/t/is-there-a-plan-to-support-dla-on-the-next-tensorrt-version/313130
- PyPI pytorch-quantization 假包警告：https://pypi.org/project/pytorch-quantization/
- NGC 真包：https://pypi.nvidia.com/pytorch-quantization/
- YOLOv8 Jetson Orin Nano FP16 vs INT8（FP16 是 sweet spot）：https://hokwangchoi.com/blog/vision-benchmarks/
- embedl SAM3 INT8 TensorRT 教程：https://docs.embedl.com/embedl-deploy/latest/auto_tutorials/sam3.html
- NVIDIA-AI-IOT/nanosam（SAM image encoder 敏感性实证）：https://github.com/NVIDIA-AI-IOT/nanosam
- Q-SAM2 论文（SAM2 编码器量化）：https://arxiv.org/html/2506.09782v1
- IGEV++ 立体匹配论文：https://arxiv.org/html/2409.00638v1
- 轻量立体匹配 INT8 论文（视差回归敏感性）：https://www.csroc.org.tw/journal/JOC36-5/JOC3605-27.pdf
- ModelOpt aarch64 wheel（Jetson 安装）：https://github.com/ajeetraina/jetson-orin-nano-super-guide

---

## 9. 本调研的局限与后续

- **证据强度分级**：官方文档/源码/README = 强证据；论坛实测/社区工程/论文 = 中强证据；个别性能数字（如 A100 INT8 1.4-1.6x）来自第三方博客，未在 NVIDIA 官方页核实到原始出处，标注为「间接证据」。
- **未实际跑验证**：本调研是纯文档调研，未在真实 Orin/RTX 上跑 ModelOpt 或 pytorch_quantization 验证。交付的 `modelopt_igev_ptq.py` 通过了语法校验（py_compile），但未接真实 IGEV 权重跑过。要在你的硬件上验证，需装 nvidia-modelopt + 真实模型权重跑一轮。
- **版本时效**：ModelOpt 迭代很快（0.31→0.41），quant_cfg 的 list/dict 格式、默认 CFG 常量名可能随版本变化。用前查对应版本 changelog。
