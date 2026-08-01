# Fast-FoundationStereo 复现文档

> 项目地址：https://github.com/NVlabs/Fast-FoundationStereo
> 复现环境：WSL2 (Ubuntu 24.04) + RTX 3060 12GB + Docker 29.5.1

---

## 第二章：ONNX 模型导出

### 2.1 前置条件

- Docker 镜像 `ffs-cpp:latest` 已构建（见第一章）
- 模型权重 `model_best_bp2_serialize.pth` 已放入容器内（68 MB）

模型位置：
```
/root/repos/Fast-FoundationStereo/models/23-36-37/
├── cfg.yaml
├── model_best_bp2_serialize.pth   # 68 MB, PyTorch checkpoint
└── 23_36_37_iters_8_res_576x960.onnx  # 预置的另一个 ONNX（576×960）
```

### 2.2 导出命令

```bash
# 进入容器
docker exec -it ffs-cpp bash

# 激活 conda 环境
eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate my

# 导出 ONNX
cd /root/repos/Fast-FoundationStereo
python scripts/make_single_onnx.py \
    --model_dir models/23-36-37/model_best_bp2_serialize.pth \
    --save_path output/onnx/ \
    --height 512 \
    --width 512 \
    --valid_iters 8 \
    --max_disp 192 \
    --onnx_name ffs_512_8_192_single
```

参数说明：

| 参数 | 值 | 含义 |
|---|---|---|
| `--model_dir` | `.pth` 路径 | 训练好的权重文件 |
| `--save_path` | `output/onnx/` | 输出目录 |
| `--height/width` | 512 | 输入分辨率（必须被 32 整除） |
| `--valid_iters` | 8 | GRU 迭代次数 |
| `--max_disp` | 192 | 最大视差 |
| `--onnx_name` | 自定义 | 输出文件名（不含扩展名） |

### 2.3 错误：PyTorch 2.9.1 不支持 `aten.adaptive_max_pool2d` 的 ONNX 导出

#### 2.3.1 现象

首次执行导出命令时报错：

```
[torch.onnx] Obtain model graph ... ✅
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX... ❌

ConversionError: Failed to convert the exported program to an ONNX model.
DispatchError: No ONNX function found for
  <OpOverload(op='aten.adaptive_max_pool2d', overload='default')>.
  Failure message: No decompositions registered for the real-valued input

Error when translating node %adaptive_max_pool2d :
  args = (%relu_21, [1, 1])
```

#### 2.3.2 根因分析

**现象**：ONNX 导出流程的三步中，前两步（获取模型图、运行 decomposition）成功，第三步（翻译为 ONNX 图）失败。

**为什么会发生**：

1. 模型在 `core/submodule.py:613` 使用了 `nn.AdaptiveMaxPool2d(1)`（即输出 1×1 的全局最大池化）：

```python
# core/submodule.py, ChannelAttentionEnhancement 类
self.max_pool = nn.AdaptiveMaxPool2d(1)
```

2. PyTorch 2.9.1 的新版 ONNX exporter（dynamo-based，默认启用）在翻译阶段遇到 `aten.adaptive_max_pool2d.default` 算子时，查找其 ONNX 映射函数——但该算子在 PyTorch 2.9.1 的 ONNX decomposition 注册表中**不存在**。

验证证据：
```python
from torch._decomp import get_decompositions
decomps = get_decompositions([torch.ops.aten.adaptive_max_pool2d.default])
print(decomps)  # 输出: {}  ← 空！没有注册任何 decomposition
```

3. 对比：`aten.adaptive_avg_pool2d` 有完整的 ONNX 支持（映射到 `GlobalAveragePool`），但 `adaptive_max_pool2d` 的 decomposition 在 PyTorch 主分支中一直未被合入——这是 PyTorch ONNX exporter 的一个已知 gap。

**为什么 dockerfile（torch 2.6.0）不报错？**

`dockerfile` 使用 `torch==2.6.0`，如果 `dynamo=True`（默认）同样会遇到此问题。但如果用户使用 `dockerfile` 时的 ONNX 导出脚本可能走了 `dynamo=False` 的旧 TorchScript 路径，旧 exporter 通过 `torch.jit.trace` 直接捕获底层算子的 ONNX 映射，不会经过 decomposition 注册表检查。

**错误链路**：

```
torch.onnx.export()
  → 步骤 1: torch.export.export() → ExportedProgram       ✅
  → 步骤 2: decompose_with_registry() → decomposed program  ✅
  → 步骤 3: _translate_fx_graph()                           ❌
       → _handle_call_function_node_with_lowering()
            → _dispatching.dispatch(node, registry)
                 → 查找 aten.adaptive_max_pool2d 的 ONNX 映射 → 未找到
                 → 尝试 decomposition → 无 registered decomposition
                 → raise DispatchError
```

#### 2.3.3 解决方案：注册 decomposition

在 `torch.onnx.export()` 之前，注册 `aten.adaptive_max_pool2d` 的 decomposition，将其分解为基础 ONNX 可翻译的 `max_pool2d_with_indices`。

**数学原理**：

`adaptive_max_pool2d(x, (oh, ow))` 等价于用以下参数调用 `max_pool2d`：

```
stride_h = floor(H / oh)
kernel_h = H - (oh - 1) * stride_h
stride_w = floor(W / ow)
kernel_w = W - (ow - 1) * stride_w
```

这是 adaptive pooling 的通用定义——kernel 和 stride 由输入/输出尺寸动态计算，确保输出精确为 `(oh, ow)`。

对于本模型的特殊场景 `output_size=(1,1)`（全局最大池化），退化为：

```
stride_h = H, kernel_h = H   → 核覆盖整个高度，无滑动
stride_w = W, kernel_w = W   → 核覆盖整个宽度，无滑动
```

即对整个 H×W 做一次 max_pool2d 得到 1×1。

**修改内容**（`scripts/make_single_onnx.py`）：

```diff
--- a/scripts/make_single_onnx.py
+++ b/scripts/make_single_onnx.py
@@ -62,6 +62,7 @@
 import torch.nn as nn
 import torch.nn.functional as F
 from omegaconf import OmegaConf
+from torch._decomp import register_decomposition   # ← 新增导入
 import core.foundation_stereo as _fs_module
 from core.foundation_stereo import FastFoundationStereo
 
@@ -196,6 +197,19 @@
     _fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
     _fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx
 
+    # Register decomposition for adaptive_max_pool2d
+    # (not supported by ONNX exporter in PyTorch 2.9)
+    @register_decomposition(torch.ops.aten.adaptive_max_pool2d.default)
+    def _adaptive_max_pool2d_decomp(x, output_size):
+        B, C, H, W = x.shape
+        oh, ow = output_size[0], output_size[1]
+        stride_h = H // oh
+        kernel_h = H - (oh - 1) * stride_h
+        stride_w = W // ow
+        kernel_w = W - (ow - 1) * stride_w
+        return torch.ops.aten.max_pool2d_with_indices.default(
+            x, [kernel_h, kernel_w], [stride_h, stride_w])
+
     torch.onnx.export(
         wrapper,
         (left_img, right_img),
```

**为什么能工作**：

- `aten.adaptive_max_pool2d.default` 返回 `(values, indices)` 元组
- `aten.max_pool2d_with_indices.default` 也返回 `(values, indices)` 元组——**签名完全匹配**
- `max_pool2d_with_indices` 在 ONNX exporter 中有完整的翻译支持（映射到 ONNX `MaxPool` op）
- `@register_decomposition` 将我们的 Python 函数注册到 PyTorch 的全局 decomposition 表，ONNX exporter 的第三步翻译时会自动调用

#### 2.3.4 验证修复

重新执行导出命令后，三步全部通过：

```
[torch.onnx] Obtain model graph ... ✅
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX... ✅

INFO ONNX model  : output/onnx/ffs_512_8_192_single.onnx
INFO Config      : output/onnx/ffs_512_8_192_single.yaml
INFO Resolution  : 512 x 512
```

### 2.4 生成文件

导出成功后产生三个文件：

| 文件 | 大小 | 说明 |
|---|---|---|
| `ffs_512_8_192_single.onnx` | 1.8 MB | ONNX 模型图结构（节点拓扑 + 外部数据引用） |
| `ffs_512_8_192_single.onnx.data` | 104 MB | 外部权重数据（大 tensor 的二进制存储） |
| `ffs_512_8_192_single.yaml` | 202 B | 模型配置参数 |

**`.onnx` + `.onnx.data` 的关系**：

PyTorch ONNX exporter 默认 `external_data=True`。因为 protobuf 有 2GB 单文件限制，大权重（超过阈值）不会内嵌在 `.onnx` 中，而是写入同目录下的 `.onnx.data` 文件。`.onnx` 中存储对这些外部数据的 offset+length 引用。

验证 `.onnx` 确实引用了外部文件：
```bash
$ strings ffs_512_8_192_single.onnx | grep ".onnx.data"
ffs_512_8_192_single.onnx.data   # ← 三个引用
ffs_512_8_192_single.onnx.data
ffs_512_8_192_single.onnx.data
```

转 TRT engine 时，`trtexec` 的 ONNX parser 会自动读取同目录的 `.data` 文件，只需传 `.onnx` 即可。

**`.yaml` 内容**：

```yaml
corr_levels: 2
corr_radius: 4
hidden_dims: [128]
image_size: [512, 512]
max_disp: 192
mixed_precision: false
n_downsample: 2
n_gru_layers: 1
normalize: true
valid_iters: 8
vit_size: vitl
```

此文件由 `make_single_onnx.py` 自动从 `model.args` 提取，供后续推理脚本读取（输入分辨率、最大视差等参数）。

**ONNX 模型结构**：

```python
import onnx
m = onnx.load("ffs_512_8_192_single.onnx")

# IR version: 10
# Opset: [('', 18)]
# Nodes: 1948
# Inputs:  ['left_image', 'right_image']
# Outputs: ['disparity']
```

### 2.5 ONNX 图简化（onnxsim）

#### 2.5.1 命令

```bash
python -m onnxsim \
    output/onnx/ffs_512_8_192_single.onnx \
    output/onnx/ffs_512_8_192_single_sim.onnx
```

#### 2.5.2 结果对比

| 指标 | 原始 (external data) | 简化后 (single file) |
|---|---|---|
| 文件数 | 2（.onnx + .onnx.data） | 1（.onnx） |
| 总大小 | 1.8 + 104 = **105.8 MB** | **60.7 MB** |
| 节点数 | 1948 | 1911 |
| IR 版本 | 10 | 10 |
| Opset | 18 | 18 |
| 外部数据引用 | ✅ 有 | ❌ 无（全内嵌） |
| 输入 | left_image, right_image | left_image, right_image |
| 输出 | disparity | disparity |

**体积缩减原因**：

- onnxsim 的常量折叠（constant folding）预计算了静态子图，消除了冗余的常量节点
- 权重从外部 `.data` 文件合并进单个 `.onnx` 文件（单文件更方便分发）

#### 2.5.3 数值精度验证

**验证方法**：用相同的随机输入（seed=42），在 CPU 上分别运行原始 ONNX 和简化 ONNX，对比输出 disparity。

**验证脚本**：

```python
import onnxruntime as ort
import numpy as np
import onnx

# 固定随机种子
np.random.seed(42)
left  = np.random.randn(1, 3, 512, 512).astype(np.float32)
right = np.random.randn(1, 3, 512, 512).astype(np.float32)

# 原始模型推理
sess_orig = ort.InferenceSession("ffs_512_8_192_single.onnx",
                                  providers=["CPUExecutionProvider"])
out_orig = sess_orig.run(None, {"left_image": left, "right_image": right})[0]

# 简化模型推理
sess_sim = ort.InferenceSession("ffs_512_8_192_single_sim.onnx",
                                 providers=["CPUExecutionProvider"])
out_sim = sess_sim.run(None, {"left_image": left, "right_image": right})[0]

# 对比
diff = np.abs(out_orig - out_sim)
print(f"Max absolute diff:  {diff.max():.10f}")
print(f"Mean absolute diff: {diff.mean():.10f}")
print(f"Max relative diff:  {(diff / (np.abs(out_orig) + 1e-8)).max():.10f}")
print(f"All close (atol=1e-5): {np.allclose(out_orig, out_sim, atol=1e-5)}")
print(f"All close (atol=1e-4): {np.allclose(out_orig, out_sim, atol=1e-4)}")
print(f"All close (atol=1e-3): {np.allclose(out_orig, out_sim, atol=1e-3)}")

# 图结构对比
m_orig = onnx.load("ffs_512_8_192_single.onnx")
m_sim  = onnx.load("ffs_512_8_192_single_sim.onnx")
print(f"Original nodes:    {len(m_orig.graph.node)}")
print(f"Simplified nodes:  {len(m_sim.graph.node)}")
print(f"Reduction: {len(m_orig.graph.node) - len(m_sim.graph.node)} nodes")
```

**验证结果**：

```
=== 推理结果 ===
Original output shape: (1, 1, 512, 512), range: [23.745523, 35.672344]
Simplified output shape: (1, 1, 512, 512), range: [23.745522, 35.672375]

=== 数值对比 ===
Max absolute diff:  0.0000858307    (8.58 × 10⁻⁵)
Mean absolute diff: 0.0000099928    (1.00 × 10⁻⁵)
Max relative diff:  0.0000028415    (2.84 × 10⁻⁶)
All close (atol=1e-5): True  ✅
All close (atol=1e-4): True  ✅
All close (atol=1e-3): True  ✅

=== 图结构对比 ===
Original nodes:    1948
Simplified nodes:  1911
Reduction:         37 nodes (1.9%)
```

#### 2.5.4 结论

**onnxsim 优化是安全的，数值精度无损。** 差异在 10⁻⁵ 量级，属于浮点运算顺序不同（常量折叠改变了中间计算顺序）导致的正常舍入误差，对实际推理结果的 disparity 精度没有可感知的影响。

建议后续 TRT engine 构建使用简化后的 `_sim.onnx`——单文件、更小、节点更少、转换更快。

### 2.6 完整操作命令汇总

```bash
# 1. 进入容器
docker exec -it ffs-cpp bash
eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate my
cd /root/repos/Fast-FoundationStereo

# 2. 备份原始脚本
cp scripts/make_single_onnx.py scripts/make_single_onnx.py.bak

# 3. 修改脚本（按 2.3.3 节的 diff 或手动编辑）
#    - 第 65 行: 添加 from torch._decomp import register_decomposition
#    - 第 200 行: 添加 @register_decomposition(...) 代码块

# 4. 导出 ONNX
python scripts/make_single_onnx.py \
    --model_dir models/23-36-37/model_best_bp2_serialize.pth \
    --save_path output/onnx/ \
    --height 512 --width 512 \
    --valid_iters 8 --max_disp 192 \
    --onnx_name ffs_512_8_192_single

# 5. ONNX 图简化（可选但推荐）
python -m onnxsim \
    output/onnx/ffs_512_8_192_single.onnx \
    output/onnx/ffs_512_8_192_single_sim.onnx

# 6. 验证
python -c "
import onnx
m = onnx.load('output/onnx/ffs_512_8_192_single_sim.onnx')
print(f'Nodes: {len(m.graph.node)}, Inputs: {[i.name for i in m.graph.input]}')
"
```

### 2.7 踩坑记录

#### 坑 1：onnxruntime-gpu 与容器 CUDA 版本不匹配

**现象**：导入 `onnxruntime` 时报错：
```
ImportError: libcudart.so.13: cannot open shared object file
```

**根因**：Dockerfile 中 `uv pip install onnxruntime-gpu` 安装了 GPU 版本的 onnxruntime，它编译时依赖 CUDA 13 的 `libcudart.so.13`。但容器内安装的是 CUDA 12.4 + PyTorch cu128（CUDA 12.8），没有 CUDA 13 的运行时库。

**解决**：安装 CPU 版本的 onnxruntime 覆盖 GPU 版本（用于精度验证，不是生产推理）：
```bash
pip install onnxruntime
```
CPU 版本不依赖 CUDA，可以直接运行。生产推理应使用 TRT engine，不需要 onnxruntime。

### 2.8 TRT Engine 构建

#### 2.8.1 trtexec 版本选择

ffs-cpp 容器中有**两个** trtexec：

| 来源 | 路径 | 版本 | `--fp16` |
|---|---|---|---|
| .deb 包（精简版） | `/usr/bin/trtexec` | v10.11.0 b106 | ❌ 无 |
| NVIDIA hub tarball（完整版） | `/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec` | v10.3.0.26 | ✅ 有 |

**必须使用 hub tarball 的完整版 trtexec**。`.deb` 包中的精简版缺少 `--fp16`、`--int8`、`--best` 等精度控制参数（详见本节约 2.8.5「踩坑」）。

#### 2.8.2 构建命令

```bash
cd /root/repos/Fast-FoundationStereo/output/onnx/trt/

/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --onnx=ffs_512_8_192_single_sim.onnx \
    --saveEngine=ffs_512_8_192_single_sim_fp16.engine \
    --fp16 \
    --useSpinWait \
    --verbose \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --timingCacheFile=ffs_512_8_192_single_sim_fp16_time.cache \
    --exportLayerInfo=ffs_512_8_192_single_sim_fp16_layerinfo.json \
    --exportProfile=ffs_512_8_192_single_sim_fp16_profile.json \
    > ffs_512_8_192_single_sim_fp16.log 2>&1
```

| 参数 | 含义 |
|---|---|
| `--onnx` | 输入的简化 ONNX 模型（2.5 节产物） |
| `--saveEngine` | TRT engine 输出路径 |
| `--fp16` | 启用 FP16 精度（FP32 master weights，FP16 计算） |
| `--useSpinWait` | CUDA kernel 间用 spin-wait 同步（减少 CPU 开销） |
| `--verbose` | 详细日志 |
| `--dumpLayerInfo` | 每层输入输出形状 |
| `--dumpProfile` | 每层性能剖析 |
| `--separateProfileRun` | 剖析与构建分离（先构建，再独立运行 profiling） |
| `--profilingVerbosity=detailed` | 剖析详细级别 |
| `--timingCacheFile` | 保存 timing cache，下次构建可复用（跳过 tactic 搜索） |
| `--exportLayerInfo` | 层信息导出 JSON |
| `--exportProfile` | 性能剖析导出 JSON |

#### 2.8.3 构建结果

**时间线**：

```
[00:28:37] 开始构建
[00:28:39] Timing cache 未命中，开始全局 tactic 搜索
[00:38:26] Formats and tactics selection completed in 587.254 seconds
[00:38:34] Engine built in 595.561 sec
[00:38:34] Timing cache 已保存（14,560 entries, 5.5 MB）
[00:38:46] Profiling 完成，PASSED
```

**关键指标**：

| 指标 | 值 |
|---|---|
| 构建总耗时 | **595.6 秒（~10 分钟）** |
| Tactic 搜索耗时 | 587.3 秒 |
| After reformat layers | 733 |
| Pre-optimized blocks | 602 |
| Optimized blocks | 18 |
| Timing cache entries | 14,560（5.5 MB） |
| Engine 文件大小 | **39 MB** |
| 推理延迟（mean） | **67.9 ms** |
| 推理延迟（min） | 67.5 ms |
| 辅助流数量 | 2（部分层可并行执行） |

**生成文件**（`output/onnx/trt/`）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `ffs_512_8_192_single_sim.onnx` | 61 MB | 输入 ONNX（从上层复制） |
| `ffs_512_8_192_single_sim_fp16.engine` | **39 MB** | TRT 序列化 engine |
| `ffs_512_8_192_single_sim_fp16.log` | 25 MB | 构建 + profiling 全日志 |
| `ffs_512_8_192_single_sim_fp16_layerinfo.json` | 853 KB | 每层 I/O 形状 |
| `ffs_512_8_192_single_sim_fp16_profile.json` | 154 KB | 每层延迟剖析 |
| `ffs_512_8_192_single_sim_fp16_time.cache` | 5.3 MB | Timing cache（可复用） |

**推理延迟解读**：

Total 行显示 `mean=67.8887ms, min=67.5446ms`（512×512 输入，RTX 3060）。TRT 报告使用 2 个辅助流，部分层可并行执行，实际端到端延迟可能略低于 Total 行的 sum。

**timing cache 的重要性**：

首次构建时，TRT 需要搜索最优 kernel 实现（tactic search），耗时 ~10 分钟。`--timingCacheFile` 将搜索结果序列化保存。**下次构建相同或相似的模型时，指定同一个 cache 文件可跳过 tactic 搜索**，构建时间从 10 分钟缩短到秒级。

#### 2.8.4 关于 TRT 版本不一致的说明

构建使用了 **TRT 10.3.0.26**（hub tarball），而容器内安装的是 **TRT 10.11.0**（.deb 包）。`trtexec` 只是构建工具链的一部分——它调用同一版本目录下的 `libnvinfer.so` 等库。因此：

- **engine 文件兼容性**：TRT 10.3 构建的 engine 可以在 TRT 10.11 runtime 上加载运行（TRT 10.x 版本间兼容）
- **timing cache 兼容性**：不同 TRT 版本的 timing cache **不兼容**，切换版本需重新生成
- **推荐做法**：生产环境统一使用相同版本的 TRT 进行构建和推理

#### 2.8.5 踩坑：.deb 包的 trtexec 缺少 `--fp16`

**现象**：使用容器内 `/usr/bin/trtexec`（通过 `.deb` 包安装的 TRT 10.11.0 b106）尝试 `--fp16`：

```
trtexec: unrecognized option '--fp16'
```

**根因**：`.deb` 包中的 `trtexec`（来自 `libnvinfer-bin`）是一个**精简版**二进制，只包含基础的 build/inference 功能，不含 `--fp16`、`--bf16`、`--int8`、`--best` 等精度控制开关。

而 NVIDIA 官网的 **hub tarball**（`TensorRT-10.x.x.x.Linux.x86_64-gnu.cuda-xx.x.tar.gz`）包含完整版 `trtexec`，带全部精度参数：

```bash
# .deb 包（精简版）→ trtexec --help 搜不到 --fp16
/usr/bin/trtexec --help | grep fp16
# (无输出)

# hub tarball（完整版）→ 有 --fp16 / --bf16 / --int8 / --fp8 / --int4 / --best
/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec --help | grep fp16
# --fp16    Enable fp16 precision, in addition to fp32 (default = disabled)
```

验证（sipe_inst 容器中 TRT 10.11.0.33 hub tarball 同样有 `--fp16`）：

```
$ /root/workspace/MixedAI_Datacenter/hub/TensorRT-10.11.0.33/bin/trtexec --help | grep fp16
  --fp16    Enable fp16 precision (default = disabled)
```

**解决**：使用 hub tarball 的 trtexec，或通过 Python API 设置 `trt.BuilderFlag.FP16`。
