# IGEV 模型 TRT 10.3 动态引擎构建失败排查与修复（AGX Orin）

> 技术总结 | 2026-08-01 | 环境：AGX Orin 64GB（192.168.11.39），容器 smr_asr_nsys，TensorRT 10.3.0

## 1. 背景与环境

- **硬件**：AGX Orin 64GB 统一内存，MAXN 电源模式（`nvpmodel -q` 确认）
- **软件**：容器 `smr_asr_nsys`，TensorRT **10.3.0**（`/usr/src/tensorrt`，python 包 tensorrt-10.3.0），onnx 1.19.1
- **模型**：IGEV（Iterative Geometry Encoding Volume）立体匹配，双输入 `image1`/`image2`，ONNX 51MB，**38997 层**（GRU 22 次迭代）
- **任务**：`model_fp32_optimized_simplified.onnx` → FP16 动态 shape engine
- **原始 shape 配置**（test.sh）：min=1x3x32x32，opt=1x3x544x960，max=1x3x960x1024

## 2. 现象

原始 trtexec 构建（用户执行）跑 38 分钟后失败：

```
[E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
    (Could not find any implementation for node /update_block/encoder/convc1_1/Conv + /update_block/encoder_1/Relu.)
```

同时日志有大量 warning：

```
[W] /Reshape_499: IShuffleLayer with zeroIsPlaceHolder=true has reshape dimension
    at position 4 that might or might not be zero. TensorRT resolves it at runtime,
    but this may cause excessive memory consumption and is usually a sign of a bug in the network.
[W] Cache result detected as invalid for node: /update_block/encoder/convd1_1/Conv, LayerImpl: CaskConvolution
```

## 3. 排查过程

### 3.1 阶段一：单 profile vs dynamic 定位

用 python TRT API 快速构建验证（避免 trtexec 全量 38 分钟）：

| 测试 | shape 配置 | 结果 |
|---|---|---|
| 单 profile | min=opt=max=544x960 | **SUCCESS** 51min |
| 单 profile | min=opt=max=960x1024 | **SUCCESS** 53min |
| dynamic | min=256x256, opt=544x960, max=960x1024 | FAILED 83s（cnet/layer1/conv2，Available 257MB）|
| dynamic | min=opt=544x960, max=960x1024 | FAILED 30min（convc1_1）|
| dynamic | min=512x512, opt=544x960, max=960x1024 | FAILED 32min（convc1_1）|

**结论**：单 profile 全成功，dynamic 区间必失败 → 问题在 TRT 的 dynamic shape 处理，不在模型单点。

### 3.2 阶段二：stub libcuda 显存误检

**现象**：dynamic 构建 83 秒失败，warning 显示：

```
Tactic Device request: 360MB  Available: 257MB. Device memory is insufficient to use tactic.
```

**疑点排查**（物理显存充足，与 TRT 报告矛盾）：

```python
# cudaMemGetInfo（runtime API，空闲时）
# → free=47.08GB total=61.37GB
# 构建进行中（45 秒时）再测 → free=44.24GB（几乎没掉！）

# cudaMalloc 连续分配测试 → 40GB+ 无压力
```

**定位**：容器默认 `LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:...`——**stub 库在首位**。系统里 libcuda 有多个版本：

```
/usr/local/cuda-12.6/targets/aarch64-linux/lib/stubs/libcuda.so   ← stub（编译用）
/usr/lib/aarch64-linux-gnu/stubs/libcuda.so                        ← stub
/usr/lib/aarch64-linux-gnu/libcuda.so                              ← 链接
/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1                     ← 真实驱动 ★
```

TRT 加载 stub libcuda → 设备显存检测返回假值（Available 257MB）→ 所有需要 >257MB 的 tactic 被拒。

**修复**：

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64
```

**验证**：干净环境构建从 83 秒推进到 32 分钟（Available 恢复，cnet/conv2 不再报错）。

> 注意：trtexec 能跑 38 分钟是因为它的 RPATH 可能指向真实库路径，但**同样的 convc1_1 失败**说明 stub 问题也间接影响它（或者它加载了 stub 但失败点在别处）——统一修复环境变量最稳。

### 3.3 阶段三：convc1 通道维丢失（核心根因）

干净环境后 dynamic 仍失败在 convc1_1（32 分钟）。深入排查：

**TRT 解析后的层结构**（python 查 network）：

```
38004 LayerType.CONV /update_block/encoder/convc1_18/Conv
  in 0 [1, -1, -1, -1]      ← 输入通道 C 也是动态的！
  out 0 [1, 64, -1, -1]
```

convc1 是 1x1 conv（group=1），kernel 形状 [64, C_in, 1, 1] **依赖输入通道 C_in**。C_in 动态 → kernel 无法固定 → 无实现。

**为什么 C 动态？** 输入链：

```
/Transpose_2 [0,3,4,1,2] → Reshape_520 → GridSample_72（cost volume 采样）
  → Reshape_637 → Concat_609 → Transpose_21 [0,3,1,2] → convc1_1
```

GridSample 输出的 shape 在 TRT dynamic shape 传播中无法解析（zeroIsPlaceHolder warning 的来源）→ 下游 reshape/concat 后 C 保持 -1。

**验证 C 实际恒定**（ORT 挂中间 tensor 到 graph output 实测）：

| 输入分辨率 | convc1_1 输入 shape | C |
|---|---|---|
| 512x512 | (1, 162, 128, 128) | 162 |
| 544x960 | (1, 162, 136, 240) | 162 |
| 512x768 | (1, 162, 128, 192) | 162 |

**22 个 convc1 节点（GRU 22 次迭代）C 全部恒为 162**（unique C = {162}）。

**TRT 解析网络验证**（代码 `list_convc1.py`）：直接用 TRT 解析原始 ONNX，打印所有 convc1 层的输入输出 shape，确认输入 C 为 -1（动态）。

```python
#!/usr/bin/env python3
"""TRT 解析 ONNX，打印所有 convc1 层的输入输出 shape，验证 C 是否动态。"""
import sys
import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

ok = parser.parse_from_file(sys.argv[1])
print("parse OK:", ok, "layers:", network.num_layers)

total_convc1 = 0
for i in range(network.num_layers):
    n = network.get_layer(i)
    if "convc1" not in n.name:
        continue
    total_convc1 += 1
    shape_in = [d for d in n.get_input(0).shape]
    shape_out = [d for d in n.get_output(0).shape]
    cin_dyn = shape_in[1] == -1 if len(shape_in) > 1 else False
    if cin_dyn:
        print("  layer %-5d %-8s in=%-20s out=%-20s C动态!" % (
            i, str(n.type)[:8], str(shape_in), str(shape_out)))
print("total convc1 layers:", total_convc1)
```

**运行方式**（AGX 或 NX 上）：

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64
python3 list_convc1.py model_fp32_optimized_simplified.onnx
```

**预期输出**（原始 ONNX，未修复）：

```
layer 38004 CONV    in=[1, -1, -1, -1]  out=[1, 64, -1, -1]  C动态!
...（共 22 层，全部 C=-1）
```

**修复后模型验证**：用 `model_fp32_optimized_simplified_fixed.onnx` 跑，输出变为：

```
layer 33412 SHUFFLE  in=[1, -1, -1, -1]  out=[1, 162, -1, -1]  C动态!
...（22 层 /convc1_fix/ Reshape 的输入 C=-1 但输出 C=162 固定）
```
原始的 convc1 Conv 因输入 C 已被 Reshape 固定为 162（不再是 -1），不再打印。证明修复生效。

**修复方案**：给每个 convc1 输入插入**恒等 Reshape [0,162,0,0]**（0=复制对应维，162 固定通道），强制 TRT 解析出 C=162。数值等价（reshape 不改变数据）。

### 3.4 附加发现：模型 Reshape 0 维 bug（min 建议 ≥512，理由：运行时安全）

ORT 直接推理小分辨率报错：

```
Reshape_80: dimension with value zero exceeds the dimension size of the input tensor
```

**机制**（probe 实测，见 `diag_reshape_bug.py`）：Reshape_320 的目标 shape 是动态构造的 `[1, 8, 12, d3, d4]`（5 元素），输入是 4D `[1, 96, H, W]`。d4 在小分辨率下算出 0：

| 分辨率 | Reshape_320 shape 参数 | 判定 |
|---|---|---|
| 64x64 / 128x128 / 256x256 | [1, 8, 12, d3, **0**] | ✗ 0 维 bug |
| 384x384 起 | [1, 8, 12, d3, ≥32] | ✓ 合法 |

ONNX Reshape 的 0 语义（zeroIsPlaceHolder）：0 = 复制输入对应维，要求 0 的位置 < 输入 rank。位置 4 的 0 对 4D 输入越界 → 非法。

**重要修正（实测验证，三轮）**：
- **TRT 构建容忍此 bug**：zeroIsPlaceHolder 只是 warning（"resolves at runtime"），修复 convc1 后 min=256×256 dynamic 构建实测 **SUCCESS**（47min）
- **TRT 运行时也容忍**：min=256 engine 用 256×256 输入推理实测 **EXIT=0、输出正常**（mean=118.9，非 0 非崩溃）——TRT 的 IShuffleLayer 处理比 ONNX 更宽容
- 真正阻塞 TRT 构建的是 convc1 通道维问题（3.3 节），**与 reshape 0 维无关**
- **min ≥512 是保守工程建议，非必需**：ONNX 层面 <384 输入非法（ORT 会崩，换后端/版本会踩雷），且 TRT warning 提示潜在内存问题；但当前 TRT 10.3 下 min=256 实测可用
- 测试脚本：`test_engine.py`（加载+推理验证）+ `diag_reshape_bug.py`（shape 矩阵）

## 4. 最终方案与验证

### 4.1 test.sh 改动（3 处，均带注释）

```bash
#!/bin/bash
# 修复1: LD_LIBRARY_PATH 排除 cuda stubs，用真实驱动路径
# 修复2: 用 convc1 通道维修复后的 ONNX
# 调整:  min 从 32x32 改为 512x512（模型 Reshape 在 <512 分辨率下有 0 维 bug）
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64

/usr/src/tensorrt/bin/trtexec \
--onnx=model_fp32_optimized_simplified_fixed.onnx \
--saveEngine=model_fp32_optimized_simplified_fp16.engine \
--fp16 \
--minShapes=image1:1x3x512x512,image2:1x3x512x512 \
--optShapes=image1:1x3x544x960,image2:1x3x544x960 \
--maxShapes=image1:1x3x960x1024,image2:1x3x960x1024 \
--useSpinWait --dumpProfile --separateProfileRun \
--profilingVerbosity=detailed \
--timingCacheFile=model_fp32_optimized_simplified_fp16_time.cache \
> model_fp32_optimized_simplified_fp16.log 2>&1
```

### 4.2 验证链

```
单 profile 544×960       → SUCCESS（51min）
单 profile 960×1024      → SUCCESS（53min）
dynamic 原模型           → FAILED（convc1_1）
dynamic 修复后模型       → SUCCESS（40min）★
正式 trtexec（修复版）   → PASSED（2691.89s ≈ 44.9min）
```

### 4.3 最终 engine 与 workspace 峰值

```
engine: model_fp32_optimized_simplified_fp16.engine（53.7MB）
Total Activation Memory = 1527618048 bytes = 1.42 GB   （trtexec 日志官方值）
engine+context 实测 GPU 占用 = 1.80 GB                   （python 加载实测）
构建期 GPU allocator 峰值 = 1369 MiB
构建期 CPU 峰值 = 43127 MiB（43GB！AGX 64GB 刚好扛住）
```

## 5. 修复代码

### 5.1 fix_convc1.py（核心修复：插恒等 Reshape 固定 C=162）

```python
#!/usr/bin/env python3
"""IGEV convc1 通道维修复：给所有 convc1 Conv 的输入插入恒等 Reshape [0,162,0,0]，
强制 TRT 解析出 C=162（模型 GridSample 链导致 TRT dynamic shape 传播丢通道维）。
用法: python3 fix_convc1.py <input.onnx> <output.onnx>
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto

def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    g = m.graph

    shape_name = '/convc1_fix_shape'
    g.initializer.append(helper.make_tensor(
        shape_name, TensorProto.INT64, [4], np.array([0, 162, 0, 0], dtype=np.int64)))

    fixed = {}          # 原输入 tensor -> reshape 输出
    n_fixed = 0
    nodes = list(g.node)
    for i, n in enumerate(nodes):
        if n.op_type == 'Conv' and 'convc1' in n.name:
            t = n.input[0]
            if t not in fixed:
                out = t + '_convc1_fixed'
                rn = helper.make_node(
                    'Reshape', inputs=[t, shape_name], outputs=[out],
                    name='/convc1_fix/' + t.replace('/', '_')[:40])
                g.node.insert(i + n_fixed, rn)
                n_fixed += 1
                fixed[t] = out
            n.input[0] = fixed[t]

    assert n_fixed > 0, 'no convc1 nodes found'
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print('OK: fixed %d convc1 inputs with Reshape [0,162,0,0]' % n_fixed)
    print('saved:', dst)

if __name__ == '__main__':
    main()
```

### 5.2 数值一致性验证（恒等 Reshape 无损）

```python
import numpy as np, onnxruntime as ort
np.random.seed(0)
x = np.random.rand(1,3,512,512).astype(np.float32)
xr = np.random.rand(1,3,512,512).astype(np.float32)
outs = []
for name in ['model_fp32_optimized_simplified.onnx', 'igev_fixed.onnx']:
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(name, so, providers=['CPUExecutionProvider'])
    outs.append(s.run(None, {'image1': x, 'image2': xr})[0])
print('max_abs_diff =', np.abs(outs[0]-outs[1]).max())   # → 0.0
```

### 5.3 qbuild2.py（参数化快速构建验证工具）

```python
#!/usr/bin/env python3
"""参数化构建验证: dynamic 区间测试。用法: python3 qbuild2.py <onnx> <minHxW> <optHxW> <maxHxW> [cache_file]"""
import sys, os, time
import tensorrt as trt

def parse(s):
    h, w = s.split('x')
    return int(h), int(w)

min_s, opt_s, max_s = parse(sys.argv[2]), parse(sys.argv[3]), parse(sys.argv[4])
cache_file = sys.argv[5] if len(sys.argv) > 5 else None

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(sys.argv[1]):
    for i in range(parser.num_errors):
        print('PARSE ERR:', parser.get_error(i))
    sys.exit(1)

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
ws = int(os.environ.get('WS_MB', '0'))
if ws > 0:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws * (1 << 20))
if cache_file and os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        config.set_timing_cache(f.read(), name='')

profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    inp = network.get_input(i)
    profile.set_shape(inp.name, (1, 3, min_s[0], min_s[1]),
                      (1, 3, opt_s[0], opt_s[1]), (1, 3, max_s[0], max_s[1]))
config.add_optimization_profile(profile)

t0 = time.time()
engine = builder.build_serialized_network(network, config)
if engine is None:
    print('RESULT: FAILED %.1fs' % (time.time() - t0), flush=True)
    sys.exit(2)
print('RESULT: SUCCESS %.1fs' % (time.time() - t0), flush=True)

if cache_file:
    with open(cache_file, 'wb') as f:
        f.write(config.get_timing_cache().serialize())
```

## 6. 排查工具命令速查

```bash
# ① ORT 中间张量 shape 实测（挂 tensor 到 graph output）
python3 -c "
import onnx
m = onnx.load('model.onnx')
for t in ['/Transpose_21_output_0']:
    vi = onnx.helper.ValueInfoProto(); vi.name = t; m.graph.output.append(vi)
onnx.save(m, 'probe.onnx')"
# 然后 ORT run 取 shape

# ② TRT 解析后层结构（查 conv 输入是否动态）
python3 -c "
import tensorrt as trt
b = trt.Builder(trt.Logger(trt.Logger.ERROR))
n = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
trt.OnnxParser(n, trt.Logger(trt.Logger.ERROR)).parse_from_file('model.onnx')
for i in range(n.num_layers):
    l = n.get_layer(i)
    if 'convc1_1' in l.name:
        print(l.name, [l.get_input(j).shape for j in range(l.num_inputs)])"

# ③ cudaMemGetInfo 实测（区分物理可用 vs TRT 报告值）
python3 -c "
import ctypes
cuda = ctypes.CDLL('libcudart.so.12')
free = ctypes.c_size_t(); total = ctypes.c_size_t()
cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
print('free_GB=%.2f total_GB=%.2f' % (free.value/2**30, total.value/2**30))"

# ④ 修正后的构建环境（关键！）
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64

# ⑤ 后台构建 + 轮询（避免 SSH 断连杀进程）
nohup python3 qbuild2.py model.onnx 512x512 544x960 960x1024 cache.ent > build.log 2>&1 &
# 轮询: while kill -0 <PID>; do sleep 30; done; cat build.log
```

## 7. 经验教训

1. **容器 LD_LIBRARY_PATH 里的 stubs 是隐形杀手**——TRT 显存检测假值（257MB）会让所有大 tactic 被拒，报错看起来像显存不足，实际是驱动加载错误
2. **单 profile 成功 ≠ dynamic 成功**——TRT dynamic shape 传播（尤其 GridSample/reshape 链）是独立风险点
3. **conv 输入通道动态是"无实现"的高发原因**——kernel 形状依赖 C_in，C 动态则无法选 kernel；恒等 Reshape 固定通道是通用修复手段（数值无损）
4. **模型导出时就可能有小分辨率 bug**——ORT 实测比 TRT 构建快得多，先验证模型本身在各分辨率下的合法性
5. **AGX 构建吃 CPU 内存巨大**（43GB）——64GB 卡刚够，重跑前注意内存占用
