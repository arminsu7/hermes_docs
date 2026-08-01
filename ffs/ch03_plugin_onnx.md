# Fast-FoundationStereo 复现文档

> 项目地址：https://github.com/NVlabs/Fast-FoundationStereo
> 复现环境：WSL2 (Ubuntu 24.04) + RTX 3060 12GB + Docker 29.5.1

---

## 第三章：Plugin ONNX 导出与 TRT Engine 构建

### 3.1 两种 ONNX 导出策略对比

FFS 提供两种 ONNX 导出脚本，对应不同的 cost volume 实现策略：

| | `make_single_onnx.py`（第二章） | `make_plugin_onnx.py`（本章） |
|---|---|---|
| Cost Volume 实现 | ONNX 原生 op（pad+slice+stack 循环） | 单个自定义 TRT Plugin 节点 |
| 模型拆分 | 无（完整单一 ONNX） | 拆为 FeatureRunner + Plugin + PostRunner |
| ONNX exporter | dynamo-based（新，PyTorch 2.6+ 默认） | TorchScript-based（旧，`dynamo=False`） |
| 输入归一化 | 外部预归一化（剥离 ImageNet mean/std） | 模型内部处理（输入乘 255） |
| 转 TRT Engine | `trtexec` 直接可转 | 需先编译 plugin .so 并加载 |
| 性能 | Cost volume 用原生 op，较慢 | Cost volume 用 CUDA kernel，理论更快 |

### 3.2 Plugin ONNX 架构

`make_plugin_onnx.py` 将模型拆为三段：

```
输入 (left, right)
    │
    ▼
┌─────────────────────┐
│  TrtFeatureRunner   │  ViT backbone + 多尺度特征提取
│  (ONNX 原生 op)     │
└─────────┬───────────┘
          │ features_left_04, features_right_04, ...
          ▼
┌─────────────────────┐
│   FFSGWCVolume      │  自定义 Plugin 节点
│   (TRT Plugin)      │  Group-wise Correlation cost volume
└─────────┬───────────┘
          │ gwc_volume
          ▼
┌─────────────────────┐
│   TrtPostRunner     │  Cost Aggregation + 8 次 GRU refinement
│   (ONNX 原生 op)    │
└─────────┬───────────┘
          │
          ▼
      disparity
```

关键类 `FastFoundationStereoPluginOnnx`：

```python
class FastFoundationStereoPluginOnnx(nn.Module):
    def __init__(self, model, max_disp_levels, cv_group, normalize):
        self.feature_runner = TrtFeatureRunner(model)  # ViT 特征提取
        self.post_runner = TrtPostRunner(model)          # Cost Aggregation + GRU

    def forward(self, left, right):
        # Step 1: ViT backbone 提取多尺度特征
        features = self.feature_runner(left, right)

        # Step 2: GWC cost volume（Plugin 节点，ONNX 中不展开）
        gwc_volume = FFSGWCVolumeOp.apply(
            features_left_04, features_right_04,
            self.max_disp_levels, self.cv_group, self.normalize
        )

        # Step 3: Cost aggregation + 8 次 GRU 迭代 → 最终 disparity
        disp = self.post_runner(features..., gwc_volume)
        return disp
```

### 3.3 自定义 Plugin Op：FFSGWCVolumeOp

```python
class FFSGWCVolumeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features_left, features_right, max_disp, cv_group, normalize):
        # 返回正确 shape 的零张量作为占位——实际计算由 TRT Plugin 完成
        return features_left.new_zeros((batch, cv_group, max_disp, h, w))

    @staticmethod
    def symbolic(g, features_left, features_right, max_disp, cv_group, normalize):
        # 在 ONNX 图中生成自定义节点 "FFSGWCVolume"
        out = g.op('FFSGWCVolume', features_left, features_right,
                   max_disp_i=..., cv_group_i=..., normalize_i=...)
```

- `forward()`：PyTorch 直接调用时返回占位张量（不参与实际计算，只提供 shape 信息给 ONNX tracer）
- `symbolic()`：ONNX 导出时插入自定义算子节点，而不是展开为原生 ONNX op

导出的 ONNX 图中有一个节点：
```
FFSGWCVolume(features_left, features_right, max_disp=48, cv_group=8, normalize=1)
```

此节点在标准 ONNX runtime 上无法执行——**必须**由 TRT Plugin 提供 CUDA kernel 实现。

### 3.4 ONNX 导出命令

```bash
cd /root/repos/Fast-FoundationStereo

python scripts/make_plugin_onnx.py \
    --model_dir models/23-36-37/model_best_bp2_serialize.pth \
    --save_path output/onnx-w-plugin/ \
    --height 512 --width 512 \
    --valid_iters 8 --max_disp 192 \
    --onnx_name ffs_512_8_192_single_w_plugin

# ONNX 图简化
onnxsim \
    output/onnx-w-plugin/ffs_512_8_192_single_w_plugin.onnx \
    output/onnx-w-plugin/ffs_512_8_192_single_w_plugin_sim.onnx
```

### 3.5 Plugin .so 编译

#### 3.5.1 编译环境

ffs-cpp 容器内：

| 组件 | 版本/路径 |
|---|---|
| nvcc | 12.4.131 |
| cmake | 3.22.1 |
| TRT headers（用于编译） | `/root/repos/hub/TensorRT-10.3.0.26/include/` |
| TRT libs（用于链接） | `/root/repos/hub/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/` |
| Plugin 源码 | `cpp/src/gwc_volume_plugin.cpp` + `cpp/src/depth_kernels.cu` |

**TRT 版本须与 trtexec 一致**：trtexec 用 TRT 10.3.0.26（hub tarball），编译 plugin 时也须指向同一版本。不能混用 .deb 包的 TRT 10.11 头文件。

#### 3.5.2 编译命令

```bash
cd /root/repos/Fast-FoundationStereo/cpp
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=/root/repos/hub/TensorRT-10.3.0.26 \
    -DCMAKE_CUDA_ARCHITECTURES=86

make ffs_gwc_plugin -j$(nproc)
```

参数说明：

| 参数 | 值 | 说明 |
|---|---|---|
| `TENSORRT_ROOT` | TRT 10.3.0.26 hub tarball | 必须和 trtexec 用的 TRT 一致 |
| `CMAKE_CUDA_ARCHITECTURES` | `86`（RTX 3060 = SM 8.6） | 只编译目标架构，加快速度 |

**CMakeLists.txt 的库查找逻辑**：

CMakeLists.txt 中 `TENSORRT_HINT_DIRS` 包含了 hub tarball 的标准路径结构。`find_library(NVINFER_LIB NAMES nvinfer nvinfer.so.10 HINTS ...)` 会在 `${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib/` 下找到 `libnvinfer.so.10`。

验证链接的库（确认指向 TRT 10.3 而非系统 TRT 10.11）：

```bash
$ ldd build/libffs_gwc_plugin.so | grep nvinfer
libnvinfer.so.10 => /root/repos/hub/TensorRT-10.3.0.26/lib/libnvinfer.so.10
```

#### 3.5.3 编译结果

```
libffs_gwc_plugin.so: 121 KB
```

导出符号：

```bash
$ nm -D build/libffs_gwc_plugin.so | grep -iE "register|getPlugin|FFSGWC"
0000000000004940 T _ZN9ffs_depth20registerFFSGWCPluginEv
0000000000004a10 T ffs_register_gwc_plugin
                 U getPluginRegistry
```

- `registerFFSGWCPlugin`：C++ 命名空间内的注册函数
- `ffs_register_gwc_plugin`：C ABI 包装，供 Python ctypes 调用
- `getPluginRegistry`：链接自 TRT runtime（未解析，运行时绑定）

#### 3.5.4 Orin NX 交叉编译说明

Plugin .so 包含 CUDA kernel（`depth_kernels.cu`），**不能从 x86 交叉编译到 aarch64**——nvcc 编译的 cubin 与 GPU 架构绑定，且 CUDA 工具链不支持跨 CPU 架构交叉编译。必须在 Orin 上本地编译：

```bash
# Orin NX 上
cd Fast-FoundationStereo/cpp
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=/usr \
    -DCMAKE_CUDA_ARCHITECTURES=87

make ffs_gwc_plugin -j$(nproc)
```

注意 Orin 上：
- `TENSORRT_ROOT=/usr`（JetPack 自带 TRT，头文件在 `/usr/include/aarch64-linux-gnu/`）
- `CMAKE_CUDA_ARCHITECTURES=87`（Orin SM 8.7）

### 3.6 TRT Engine 构建（Plugin 模式）

#### 3.6.1 trtexec 版本选择

ffs-cpp 容器中有**两个** trtexec：

| 来源 | 路径 | 版本 | `--fp16` | 用途 |
|---|---|---|---|---|
| .deb 包 | `/usr/bin/trtexec` | v10.11.0 b106 | ❌ 无 | 精简版，功能不全 |
| hub tarball | `.../TensorRT-10.3.0.26/bin/trtexec` | v10.3.0.26 | ✅ 有 | **完整版，推荐使用** |

#### 3.6.2 关于 .deb 包 trtexec 缺少 --fp16 的更正

**之前的错误分析**（已更正）：最初认为是 TRT 10.x 移除了 `--fp16` 参数。经 sipe_inst 容器验证（TRT 10.11.0.33 hub tarball 同样有 `--fp16`），正确结论是：

- **hub tarball**（`TensorRT-10.x.x.x.Linux.x86_64-gnu.cuda-xx.x.tar.gz`）：完整版 trtexec，有 `--fp16`/`--bf16`/`--int8`/`--best` 等全部精度参数
- **.deb 包**（`libnvinfer-bin`）：精简版 trtexec，缺少精度控制参数

这是 NVIDIA 打包策略的差异，与 TRT 版本号无关。

验证证据：

```bash
# .deb 包（精简版）
$ /usr/bin/trtexec --help | grep fp16
# (无输出)

# hub tarball（完整版，TRT 10.3.0.26）
$ .../TensorRT-10.3.0.26/bin/trtexec --help | grep fp16
--fp16    Enable fp16 precision (default = disabled)

# hub tarball（完整版，TRT 10.11.0.33，来自 sipe_inst）
$ .../TensorRT-10.11.0.33/bin/trtexec --help | grep fp16
--fp16    Enable fp16 precision (default = disabled)
```

#### 3.6.3 方案 A：--staticPlugins（plugin 不序列化进 engine）

```bash
cd /root/repos/Fast-FoundationStereo/output/onnx-w-plugin/trt-fuse/

/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --onnx=ffs_512_8_192_single_w_plugin_sim.onnx \
    --saveEngine=ffs_512_8_192_single_w_plugin_sim_fp16.engine \
    --staticPlugins=/root/repos/Fast-FoundationStereo/cpp/build/libffs_gwc_plugin.so \
    --fp16 \
    --useSpinWait \
    --verbose \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --profilingVerbosity=detailed \
    --exportLayerInfo=ffs_512_8_192_single_w_plugin_sim_fp16_layerinfo.json \
    --exportProfile=ffs_512_8_192_single_w_plugin_sim_fp16_profile.json \
    > ffs_512_8_192_single_w_plugin_sim_fp16.log 2>&1
```

**特点**：plugin .so 在构建时加载，但**不写入 engine 文件**。推理时需要单独提供 .so。

#### 3.6.4 方案 B：--dynamicPlugins + --setPluginsToSerialize（plugin 序列化进 engine）

期望命令：

```bash
/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --onnx=ffs_512_8_192_single_w_plugin_sim.onnx \
    --saveEngine=ffs_512_8_192_single_w_plugin_sim_fp16.engine \
    --dynamicPlugins=/root/repos/Fast-FoundationStereo/cpp/build/libffs_gwc_plugin.so \
    --setPluginsToSerialize=/root/repos/Fast-FoundationStereo/cpp/build/libffs_gwc_plugin.so \
    --fp16 \
    ...（其余参数同方案 A）
```

三个 plugin 参数区别：

| 参数 | 作用 |
|---|---|
| `--staticPlugins=xxx.so` | 构建时加载，**不**序列化进 engine |
| `--dynamicPlugins=xxx.so` | 构建时加载，**可被**序列化进 engine |
| `--setPluginsToSerialize=xxx.so` | 标记此 .so **写入** engine 文件 |

方案 A（`--staticPlugins`）已验证通过。方案 B 的完整实现见 [3.11 V3 Plugin 迁移](#311-v3-plugin-迁移与-plugin-序列化) — 需将 plugin 从 V2 迁移到 V3，增加 `getCreators`/`setLoggerFinder` 符号，并修正 TRT 10.3 的函数签名。最终 engine 构建成功，plugin 已序列化进 engine，加载时无需任何 `--*Plugins` 参数。

### 3.7 问题：TRT 10.x --dynamicPlugins 要求 V3 Plugin

#### 3.7.1 现象

使用方案 B（`--dynamicPlugins`）时报错：

```
[E] [TRT] IPluginRegistry::loadLibrary: Error Code 3: API Usage Error
(SymbolAddress for getCreators could not be loaded, check function name
against library symbol In untypedSymbolAddress at
/_src/runtime/dispatch/libLoader.cpp:378)
```

#### 3.7.2 根因分析

**阶段一：符号缺失**

TRT 10.x 的 `--dynamicPlugins` 通过 `dlsym` 查找以下符号：

| 符号 | 作用 | 在 V2 plugin 中 |
|---|---|---|
| `getCreators` | 返回 plugin creator 列表 | ❌ 未实现 |
| `getPluginCreators` | 备选名称（TRT 10.3 也检查） | ❌ 未实现 |
| `setLoggerFinder` | 设置 logger（必选） | ❌ 未实现 |

当前 V2 plugin 使用旧 API（`REGISTER_TENSORRT_PLUGIN` + `ffs_register_gwc_plugin` 手动注册），没有 `--dynamicPlugins` 所需的符号。

**阶段二：V2 Creator 与 --dynamicPlugins 不兼容**

给 V2 plugin 添加了 `getCreators` + `setLoggerFinder` 符号后，trtexec 在加载 .so 时立即 segfault。

**根因**：`FFSGWCVolumePluginCreator` 继承的是 V2 的 `IPluginCreator`（deprecated），但 TRT 10.3 的 `--dynamicPlugins` 期望的是 V3 的 `IPluginCreatorV3One`。两类 creator 的 `getInterfaceInfo()` 返回不同值，TRT 按 V3 接口调用 V2 对象导致崩溃。

**结论**：`--dynamicPlugins` + `--setPluginsToSerialize` 必须使用 V3 plugin（`IPluginCreatorV3One` + `IPluginV3`）。V2 plugin 只能用 `--staticPlugins`。

#### 3.7.3 V3 迁移方案

见 [3.11 V3 Plugin 迁移与 Plugin 序列化](#311-v3-plugin-迁移与-plugin-序列化)。

---

### 3.8 Plugin ONNX 的 shape_inference 修复与简化

#### 3.8.1 问题

`make_plugin_onnx.py`（`dynamo=False`）导出的 ONNX 包含 FFSGWCVolume 自定义节点，但 TorchScript exporter 不会为 plugin 节点的输出填充 value_info。直接运行 `onnxsim` 时，shape_inference 在 plugin 节点处被阻断，无法传播形状信息到下游节点，导致大量常数折叠未完成。

**现象**：
- plugin ONNX 原始节点数 5142，直接 onnxsim 后节点数 5142（无效果）
- value_info 为空（TorchScript exporter 不填充中间形状）

**根因**：
1. `torch.onnx.export(dynamo=False)` 不填充中间 value_info
2. `onnxsim` 依赖 `onnx.shape_inference` 传播形状
3. FFSGWCVolume 是自定义节点，无 schema，shape_inference 在此处停止
4. 下游节点的形状无法推断，常数折叠失败

#### 3.8.2 解决方案

参考 SIPE NOCS 项目的 `_simplify_with_plugin` 函数（`sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/submodule_export.py:426`），采用三层策略：

1. **形状修补**：运行 `shape_inference.infer_shapes`（有能力推断 plugin 之前的全部节点形状），手动根据 plugin 的属性（`cv_group`, `max_disp`）和输入特征图尺寸计算输出形状，写入 value_info
2. **域名标记**：将 FFSGWCVolume 的 domain 从 `""` 改为 `"trt.plugins"`，避免 onnxsim 的 checker 报错
3. **迭代简化**：运行 onnxsim → 重新运行 shape_inference → 再次 onnxsim，循环直到收敛

#### 3.8.3 脚本

```bash
cd /root/repos/Fast-FoundationStereo/output/onnx-w-plugin
python simplify_ffs_plugin.py ffs_512_8_192_single_w_plugin.onnx
```

脚本路径：`/home/armin/repos/hermes/docs/ffs/simplify_ffs_plugin.py`（已复制到容器 `output/onnx-w-plugin/` 下）

#### 3.8.4 效果

| 指标 | 简化前 | 简化后 |
|---|---|---|
| 节点数 | 5142 | 2472 |
| 缩减比例 | - | 51.9% |
| 模型大小 | - | 60 MB |
| FFSGWCVolume 节点 | domain="" | domain="trt.plugins" |
| 输出形状 | 未知 | [1, 8, 48, 128, 128] |

`onnxoptimizer` 因模型中有未消解的 Constant 引用而无法运行（onnxoptimizer 的 checker 比 onnxsim 严格），不影响 TRT 解析。

---

### 3.9 策略：将 Plugin 节点注入已简化的 Single ONNX

#### 3.9.1 动机

`make_single_onnx.py`（dynamo exporter）导出的 ONNX 已被 onnxsim 充分简化（1911 节点），但其中 GWC cost volume 由原生 ONNX ops（94 Pad + 94 Slice + Unsqueeze + Stack + Mul + ReduceSum）实现，TRT 会把它们展开为大量低效 kernel。**将这部分替换为 FFSGWCVolume plugin 节点**，获得「高度简化的大图 + 高效 plugin 算子」的最优组合。

#### 3.9.2 图结构分析

Single ONNX 中 `cat_9`（Concat）将两个 cost volume 合并：
```
cat_9 inputs: ['sum_1', 'expand_9', 'stack_3']
```
- `sum_1` = GWC volume（ReduceSum 输出，组内相关性求和）
- `stack_3` = concat volume（拼接的左右特征堆叠）
- `expand_9` = 扩展的特征张量

Plugin 只替换 **GWC path**（`sum_1`），保留 concat volume path（`stack_3`）不变。

GWC 子图输入：`slice_7`（左特征）和 `slice_11`（右特征），两者均为 `conv2d_32`（conv4 输出）的 Slice，对应 plugin 模型中的 `/feature_runner/Slice_output_0` 和 `/feature_runner/Slice_4_output_0`。

> **注意**：不要用 `conv2d_36`（`Conv(slice_11, proj_cmb)` 的投影结果）作为 plugin 输入——plugin 期望原始特征，投影在 `TrtFeatureRunner` 内部已完成。

#### 3.9.3 脚本设计

脚本：`/home/armin/repos/hermes/docs/ffs/inject_plugin_gs.py`（已复制到容器 `output/onnx-wo-plugin/` 下）

核心流程：

1. **定位 cat_9**：找到 GWC+concat 的合并 Concat 节点
2. **识别 GWC 输入**：检查 cat_9 的哪个输入由 ReduceSum 产生 → 即 `sum_1`
3. **GWC 子图边界**：前向 BFS 从 `{slice_7, slice_11}` + 后向 BFS 从 `sum_1`，取交集
4. **共享节点排除**：只删除**全部输出都在 GWC 子图内**的节点，保护被 concat volume 路径或 cost aggregation 共用的节点（如 conv2d_32、slice_7 等 ViT backbone 节点）
5. **插入 Plugin**：创建 FFSGWCVolume 节点，替换 GWC 子图

#### 3.9.4 迭代过程与踩坑

| 版本 | 问题 | 现象 | 修复 |
|---|---|---|---|
| v1 | GWC output 选错 | 选了 `stack_3`（concat volume）而非 `sum_1`（GWC） | 改为检测 ReduceSum producer |
| v2 | 共享节点误删 | conv2d_32, add_26, slice_7 等 backbone 节点被删除，导致孤儿 tensor | 加入交集 + 消费者检查，只删 GWC-exclusive 节点 |
| v3 | Plugin inputs 错误 | 使用了 `slice_11` + `conv2d_36`（投影特征），推理时 illegal memory access | 改为 `slice_7` + `slice_11`（原始左右特征切片） |

#### 3.9.5 运行

```bash
cd /root/repos/Fast-FoundationStereo/output/onnx-wo-plugin
python inject_plugin_gs.py \
    --single ffs_512_8_192_single_sim.onnx \
    --plugin ../onnx-w-plugin/trt-part-sim/ffs_512_8_192_single_w_plugin_sim.onnx \
    --output ffs_512_8_192_single_plugin_sim.onnx
```

结果：1911 节点 → 1755 节点，FFSGWCVolume 节点 `domain="trt.plugins"`，输入 `['slice_7', 'slice_11']`，输出 `sum_1`（接入 cat_9）。

---

### 3.10 验证：Single ONNX vs Plugin-Injected ONNX 输出一致性

#### 3.10.1 方法

onnxruntime 无法执行 FFSGWCVolume 自定义节点，因此采用 TRT engine 对比：

1. 分别构建两个 FP16 TRT engine（single 和 plugin-injected）
2. 用相同随机输入（seed=42）运行 inference
3. 像素级对比输出 disparity map

#### 3.10.2 TRT Engine 构建

```bash
# Single engine
/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --onnx=ffs_512_8_192_single_sim.onnx \
    --saveEngine=single.engine \
    --fp16 --useSpinWait --skipInference

# Plugin engine（--staticPlugins 方案）
/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --onnx=ffs_512_8_192_single_plugin_sim.onnx \
    --saveEngine=plugin.engine \
    --staticPlugins=/root/repos/Fast-FoundationStereo/cpp/build/libffs_gwc_plugin.so \
    --fp16 --useSpinWait --skipInference
```

#### 3.10.3 验证结果

| 指标 | 值 |
|---|---|
| 输出形状 | (1, 1, 512, 512) = (1, 1, 512, 512) ✅ |
| Max absolute diff | 0.4375 |
| Mean absolute diff | 0.0569 |
| Max relative diff | 1.47% |
| All close (atol=1.0) | True ✅ |
| 100% pixels within 0.5 | True ✅ |
| 84.6% pixels within 0.1 | - |

Single 和 Plugin 输出**在 FP16 精度下一致**（差异来自 plugin CUDA kernel 和 ONNX ops 的浮点计算路径差异，属于正常范围）。

---

### 3.11 V3 Plugin 迁移与 Plugin 序列化

#### 3.11.1 背景

如 [3.7](#37-问题-trt-10x---dynamicplugins-要求-v3-plugin) 所述，TRT 10.3 的 `--dynamicPlugins` + `--setPluginsToSerialize` 要求 plugin 为 V3 接口。需将 FFSGWCVolume 从 V2（`IPluginV2DynamicExt` + `IPluginCreator`）迁移到 V3（`IPluginV3` + `IPluginCreatorV3One`）。

#### 3.11.2 文件结构

在 `cpp-v3/` 下新建，保持与 `cpp/` 相同的目录结构：

```
cpp-v3/
├── CMakeLists.txt          # TRT 10.3.0.26 路径
├── include/
│   └── ffs_gwc_plugin.hpp  # 同 V2（未改）
└── src/
    ├── gwc_volume_plugin_v3.cpp  # V3 重写
    └── depth_kernels.cu          # 同 V2（未改）
```

**CMakeLists.txt** 关键差异：`TENSORRT_ROOT` 指向 hub tarball 10.3.0.26。

#### 3.11.3 V2 → V3 接口变化

| 方面 | V2 (`cpp/`) | V3 (`cpp-v3/`) |
|---|---|---|
| Plugin 基类 | `IPluginV2DynamicExt` | `IPluginV3` + 3 Capability 类 |
| Creator 基类 | `IPluginCreator` | `IPluginCreatorV3One` |
| 架构 | 单一类实现全部接口（~12 方法） | 主类 + Core/Build/Runtime 三个子类 |
| 输出维度 | `getOutputDimensions()` | `getOutputShapes()`（返回 int32_t） |
| 格式组合检查 | `supportsFormatCombination(PluginTensorDesc*)` | `supportsFormatCombination(DynamicPluginTensorDesc*)`（用 `.desc.format` 和 `.desc.type`） |
| 序列化 | `serialize()` + `getSerializationSize()` | `getFieldsToSerialize()`（返回 PluginField 列表） |
| 上下文挂载 | `attachToContext()` 空实现 | `attachToContext()` 返回 clone（**必须非 null**） |
| 注册 API | `registerCreator(IPluginCreator&)` (deprecated) | `registerCreator(IPluginCreatorInterface&)` |

#### 3.11.4 Capability 类架构

V3 将 plugin 拆分为三个专注的接口：

```
IPluginV3 (主类)
  └── getCapabilityInterface(type)
       ├── kCORE    → FFSGWCVolumePluginCore   (name, version, namespace)
       ├── kBUILD   → FFSGWCVolumePluginBuild  (output shapes, format, data types)
       └── kRUNTIME → FFSGWCVolumePluginRuntime (enqueue, attachToContext, serialize)
```

#### 3.11.5 必需的导出符号

`--dynamicPlugins` 通过 `dlsym` 查找三个符号：

```cpp
// 正确签名（参照 libnvinfer_vc_plugin.so）：
// 返回 creator 数组指针，通过 numCreators 输出 count
extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(
    int32_t* numCreators);

// 必选：logger 设置（空实现即可）
extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* /*finder*/);
```

> **踩坑**：最初使用 `int32_t getCreators(int32_t version, IPluginCreatorInterface*** out)` 签名，反汇编 `libnvinfer_vc_plugin.so` 中的 `getCreators` 后发现实际签名是 `IPluginCreatorInterface* const* getCreators(int32_t* numCreators)`。第一个指令 `movl $0x1, (%rdi)` 将 count 写入 `*numCreators`，然后 `lea` + `ret` 返回静态数组指针。

#### 3.11.6 `attachToContext` 必须返回有效 clone

V3 的 `IPluginV3OneRuntime::attachToContext()` 必须返回非 null 的 `IPluginV3*` clone。返回 `nullptr` 会导致 engine build 阶段 segfault。

由于 `FFSGWCVolumePluginRuntime` 在 `FFSGWCVolumePlugin` 之前定义（前向依赖），需用 out-of-line 定义：

```cpp
// 声明（在 Runtime 类内）
nvinfer1::IPluginV3* attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept override;

// 定义（在 Plugin 类之后）
nvinfer1::IPluginV3* FFSGWCVolumePluginRuntime::attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept {
    return new FFSGWCVolumePlugin(params_);
}
```

#### 3.11.7 编译

```bash
cd /root/repos/Fast-FoundationStereo/cpp-v3/build
cmake .. -DTENSORRT_ROOT=/root/repos/hub/TensorRT-10.3.0.26 -DCMAKE_CUDA_ARCHITECTURES=86
make ffs_gwc_plugin -j$(nproc)

# 验证符号
nm -D libffs_gwc_plugin.so | grep -E "getCreators|setLoggerFinder|ffs_register"
# 预期输出:
# 00000000000059a0 T getCreators
# 00000000000059c0 T setLoggerFinder
```

#### 3.11.8 构建 plugin 序列化的 Engine

```bash
cd /root/repos/Fast-FoundationStereo/output/onnx-w-plugin/trt-fused
/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --onnx=ffs_512_8_192_single_w_plugin_sim.onnx \
    --saveEngine=ffs_512_8_192_single_w_plugin_sim_fp16.engine \
    --dynamicPlugins=/root/repos/Fast-FoundationStereo/cpp-v3/build/libffs_gwc_plugin.so \
    --setPluginsToSerialize=/root/repos/Fast-FoundationStereo/cpp-v3/build/libffs_gwc_plugin.so \
    --fp16 --useSpinWait
```

> **注**：trtexec 在构建后会尝试验证反序列化，此时 `--dynamicPlugins` 注册的 creator 和 engine 内嵌的 library 中的 creator 会产生重复注册冲突（`exists already` 错误）。这是 trtexec 的已知行为，engine 文件本身已正确生成。

#### 3.11.9 加载 engine（无需任何 --*Plugins 参数）

```bash
/root/repos/hub/TensorRT-10.3.0.26/bin/trtexec \
    --loadEngine=ffs_512_8_192_single_w_plugin_sim_fp16.engine \
    --useSpinWait --iterations=10 --warmUp=1000
```

#### 3.11.10 验证结果

| 指标 | 值 |
|---|---|
| Engine 文件 | 37 MB |
| 构建耗时 | 451 秒 (~7.5 分钟) |
| GPU latency (mean) | 43.4 ms |
| GPU latency (median) | 43.3 ms |
| 推理验证 | **PASSED** ✅ |
| 加载时需 .so | **否** ✅（plugin 已序列化进 engine） |

#### 3.11.11 多平台 CMakeLists.txt（跨平台 TRT 自动检测）

参照 `cpp/CMakeLists.txt`，更新 `cpp-v3/CMakeLists.txt` 支持多平台自动检测 TRT 路径。

**改动**：

| 项目 | 旧版（硬编码） | 新版（自动检测） |
|---|---|---|
| `TENSORRT_ROOT` 默认值 | `/root/repos/hub/TensorRT-10.3.0.26` | `/usr`（可 `-D` 覆盖） |
| 搜索路径 | 仅 hub tarball | hub tarball + conda + 系统路径 + CUDA 路径 |
| `PATH_SUFFIXES` | `lib`, `include` | 新增 `lib64`, `lib/x86_64-linux-gnu`, `lib/aarch64-linux-gnu`, `targets/*/lib`, `targets/*/include` |
| conda Python 搜索 | 无 | 自动搜索 `$CONDA_PREFIX/lib/python*/site-packages/tensorrt*` |

**各平台用法**：

```bash
# 本机（RTX 3060 + hub tarball）
cmake .. -DTENSORRT_ROOT=/root/repos/hub/TensorRT-10.3.0.26 -DCMAKE_CUDA_ARCHITECTURES=86

# 本机（.deb 包，默认自动检测）
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86

# Orin NX（JetPack TRT）
cmake .. -DTENSORRT_ROOT=/usr -DCMAKE_CUDA_ARCHITECTURES=87

# 自定义路径
cmake .. -DTENSORRT_ROOT=/opt/tensorrt/custom -DCMAKE_CUDA_ARCHITECTURES=89
```

`CMAKE_CUDA_ARCHITECTURES` 默认值 `80;86;89;90`，Orin 上需手动指定 `87`。

#### 3.11.12 运行 Demo（修复 Tensor 名不匹配）

##### 问题

`run_demo_fused_plugin_trt.py` 中的 tensor 名与 engine 不匹配：

| | 脚本期望 | Engine 实际 |
|---|---|---|
| Input | `left_image`, `right_image` | `left`, `right` |
| Output | `disparity` | `disp` |

**根因**：`make_plugin_onnx.py` 导出 ONNX 时使用 `input_names=['left', 'right'], output_names=['disp']`，而脚本是针对另一套 ONNX（`input_names=['left_image', 'right_image']`）写的。

##### 修复

在 `run_demo_fused_plugin_trt.py` 中修改两行（第 288-289 行）：

```python
# 修改前
outputs = runner({'left_image': t_left, 'right_image': t_right})
disp = outputs['disparity']

# 修改后
outputs = runner({'left': t_left, 'right': t_right})
disp = outputs['disp']
```

##### 验证

```bash
cd /root/repos/Fast-FoundationStereo
python scripts/run_demo_fused_plugin_trt.py \
    --model_file output/onnx-w-plugin/trt-fused/ffs_512_8_192_single_w_plugin_sim_fp16.engine \
    --left_file demo_data/left.png --right_file demo_data/right.png \
    --intrinsic_file demo_data/K.txt --out_dir output_trt/ \
    --get_pc 1 --remove_invisible 0 --denoise_cloud 1 --zfar 100
```

结果：✅ Engine 加载成功（无 `--*Plugins` 参数），推理正常，点云保存到 `output_trt/`。

---

## 后续章节（已完成 / 待补充）

- [x] 第三章：Plugin ONNX 导出、简化、编译与 TRT Engine 构建
- [x] 3.8：Plugin ONNX shape_inference 修复与 iterative onnxsim
- [x] 3.9：Plugin 节点注入已简化的 Single ONNX（inject_plugin_gs.py）
- [x] 3.10：TRT engine inference 对比验证
- [x] 3.11：V3 Plugin 迁移与 Plugin 序列化（cpp-v3/）
- [ ] 第四章：性能对比（Single vs Plugin-Injected vs Plugin-Serialized engine）
- [ ] 第五章：Orin NX 端到端部署
