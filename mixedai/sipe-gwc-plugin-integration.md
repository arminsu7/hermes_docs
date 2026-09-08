# SIPE build_gwc_volume Plugin 集成文档

> 环境：gwc_cuda128 容器（CUDA 12.8, TRT 10.11, py310, torch 2.7.0+cu128）
> 日期：2026-08-18
> 脚本目录：`/root/repos/hermes/docs/mixedai/scripts/`

---

## 一、概述

SIPE 模型的 `build_gwc_volume`（groupwise correlation cost volume）原本是纯 PyTorch 实现（Python for 循环逐 disparity 做 groupwise correlation），导出 ONNX 后展开成几十个 slice/mul/reduce_mean 节点。本文档描述将其替换为 `FFSGWCVolume` TensorRT plugin 节点的完整方案。

涉及两个仓库的改动：
- **mixedai**（`/root/repos/dex_lib/mixedai`）：ONNX 导出侧，加 `--use-plugin` 开关
- **glia**（`/root/repos/dex_lib/glia`）：plugin 源码集成 + 编译

---

## 二、具体改动清单

### 2.1 mixedai 改动

#### 新增文件

**`mixedai/engine/exporter/plugin_export.py`**（新建，~240 行）

FFSGWCVolume plugin ONNX 导出辅助模块，包含：

| 组件 | 作用 |
|------|------|
| `_FFSGWCVolumeOp(torch.autograd.Function)` | forward() 返回正确 shape 的零 tensor；symbolic() 发出 `FFSGWCVolume` 自定义 ONNX 节点 |
| `_build_gwc_volume_plugin_shim()` | `build_gwc_volume` 的 drop-in 替换，调用 `_FFSGWCVolumeOp.apply` |
| `patch_build_gwc_volume()` | context manager，monkey-patch sipe.py 里的 `build_gwc_volume` 为 shim |
| `export_onnx_with_plugin()` | `torch.onnx.export` 包装，跳过 ONNX proto checker（plugin 节点无 schema） |
| `_tag_plugin_domain()` | 将 `FFSGWCVolume` 节点移入 `trt.plugins` domain |
| `_patch_plugin_output_shapes()` | 给 plugin 输出 tensor 注入 shape 信息，让 onnxsim 能传播 |
| `post_process_plugin_onnx()` | tag domain + patch shape + onnxsim 简化 |

源码路径：`/root/repos/dex_lib/mixedai/mixedai/engine/exporter/plugin_export.py`

#### 修改文件

**`mixedai/scripts/export.py`**（3 处改动）

1. **加 `--use-plugin` 命令行参数**（~第 391 行）：
   ```python
   parser.add_argument("--use-plugin", default=False, action="store_true",
       help="Export ONNX with FFSGWCVolume TensorRT plugin node ...")
   ```

2. **加 plugin 导出逻辑**（~第 269 行）：在 `export_handle.export_tracing` 之前判断 `--use-plugin`，走 plugin 路径：
   ```python
   if getattr(args, "use_plugin", False) and model_arch == "SIPE":
       # 1. _PluginWrapper 包装 SIPE 模型
       #    - forward 调 SIPE.forward (eval) -> build_gwc_volume (被 monkey-patch)
       #    - 调 decode_outputs 对齐原始导出的 seg_output 格式
       # 2. formatting_inputs 准备 NCHW dummy 输入
       # 3. patch_build_gwc_volume() context 下 export_onnx_with_plugin
       # 4. post_process_plugin_onnx (tag domain + simplify)
   else:
       export_handle.export_tracing(model, inputs, onnx_path)  # 原始路径
   ```

3. **`_PluginWrapper.forward` 里加 `decode_outputs`**：
   ```python
   outputs, seg_output = self.model.decode_outputs(
       outputs, seg_output[-1].permute(0, 2, 3, 1).contiguous()
   )
   ```
   跟 `SIPE.inference` 的 exporting 分支对齐，确保 seg_output 经过 decode（上采样 + permute）。

### 2.2 glia 改动

#### 新增文件（3 个）

源码从 `sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/plugin/` 移植到 glia：

| 原始文件 | glia 目标路径 | 说明 |
|----------|--------------|------|
| `src/gwc_volume_plugin.cpp` | `cpp/glia/dl/ops/tensorrt/ffs_gwc_plugin.cpp` | plugin 实现（重命名） |
| `src/depth_kernels.cu` | `cpp/glia/dl/ops/tensorrt/ffs_gwc_depth_kernels.cu` | CUDA kernel |
| `include/ffs_gwc_plugin.hpp` | `cpp/glia/dl/ops/tensorrt/ffs_gwc_plugin.hpp` | 头文件 |

#### 不需要改的文件

- **`cpp/glia/dl/ops/CMakeLists.txt`**：不用改。已有的 `file(GLOB TRT_CUSTOM_OPS tensorrt/*.cpp)` 和 `file(GLOB TRT_CUSTOM_CU_OPS tensorrt/*.cu)` 自动收集新文件，编进 `libtrt_ops.so`。
- **`scripts/compile_linux_x86.sh`**：不用改。已开启 `BUILD_TENSORRT=ON`。
- **include 路径**：不用改。`trt_ops` target 的 `target_include_directories` 已含 CUDA 头文件和 `tensorrt/common/`；TRT 头文件由 `GLIA::3rdparty_tensorrt` 提供。

---

## 三、增量编译

### 3.1 mixedai

mixedai 是 editable 安装（`pip install -e . --no-build-isolation`），改 Python 代码即改即生效，不需要编译。

### 3.2 glia

glia 的 `libtrt_ops.so` 是 C++/CUDA 编译产物，需要重新编译。

**增量编译（推荐）：**

```bash
cd /root/repos/dex_lib/glia/build

# 1. 重跑 cmake configure（让 GLOB 扫到新文件，必须做）
cmake .. -DBUILD_CUDA_MODULE=ON \
         -DBUILD_DL_MODULE=ON \
         -DBUILD_TENSORRT=ON \
         -DUSE_SYSTEM_TENSORRT=ON \
         -DCMAKE_BUILD_TYPE=Release

# 2. 只编译 trt_ops target
make -j8 trt_ops

# 3. 更新到 glia pip 包
cp lib/Release/libtrt_ops.so \
   /root/miniconda3/envs/py310/lib/python3.10/site-packages/glia/lib/libtrt_ops.so
```

**全量编译（需重装 pip 包时）：**

```bash
cd /root/repos/dex_lib/glia
bash scripts/compile_linux_x86.sh
```

注意：`compile_linux_x86.sh` 会 `rm -rf build` 然后全量编译 + `pip install`，耗时较长（10-20 分钟）。增量编译只需 1-2 分钟。

### 3.3 验证编译结果

```bash
# 检查 libtrt_ops.so 是否包含 FFSGWCVolume 注册符号
nm -D /root/miniconda3/envs/py310/lib/python3.10/site-packages/glia/lib/libtrt_ops.so \
   | grep ffs_register_gwc_plugin

# 期望输出：
# 000000000001d700 T ffs_register_gwc_plugin
```

---

## 四、导出带 plugin 的 ONNX

### 4.1 命令

```bash
cd /root/repos/dex_lib/mixedai

python -m mixedai.scripts.export \
  --config-file /root/repos/dex_lib/sipe-config/sipe_phone_fix_811_da_kpt10_50k_simplify.yaml \
  --model-file /root/repos/dex_lib/sipe-config/sipe_phone_fix_811_da_kpt10_50k_simplify.ckpt \
  --sample-image /root/repos/dex_lib/sipe-config/val_data/rgb/000071.png \
                 /root/repos/dex_lib/sipe-config/val_data/rightrgb/000071.png \
  --no-cad \
  --use-plugin
```

### 4.2 导出产物

| 文件 | 说明 |
|------|------|
| `model_plugin_sim.onnx` | 带 FFSGWCVolume plugin 节点的简化 ONNX（主产物） |
| `export_cfg.yaml` | 导出时合并后的完整配置快照 |
| `run.py` | ONNX 推理脚本 |
| `dummy_input_left.png` | 导出用的左图样本 |
| `dummy_input_right.png` | 导出用的右图样本 |

输出目录：`<ALGO_PROJECT export path>/<时间戳>/`

### 4.3 ONNX 结构

**输入（2 个，动态 shape，与原始导出一致）：**
- `sipe_image1`: `['batch', 'c', 'h', 'w']`（左图，全动态）
- `sipe_image2`: `['batch', 'c', 'h', 'w']`（右图，全动态）

**输出（4 个）：**
- `sipe_predictions`: 检测输出
- `sipe_segweights`: 分割权重
- `sipe_keypoints`: 关键点
- `sipe_disp`: 视差预测

**plugin 节点：**
```
FFSGWCVolume (domain: trt.plugins)
  inputs: match_left, match_right
  attrs: max_disp=24, cv_group=8, normalize=0
  output: [B, 8, 24, H, W]
```

### 4.4 对比：原始导出 vs plugin 导出

| | 原始导出（不加 `--use-plugin`） | plugin 导出（`--use-plugin`） |
|---|---|---|
| ONNX 节点 | ~2000+（含 build_gwc_volume 展开的 slice/mul/reduce） | ~1548（GWC 合并为 1 个 plugin 节点） |
| 输入 shape | 动态 `[batch, c, h, w]` | 动态 `[batch, c, h, w]`（一致） |
| 第 4 输出名 | `sipe_visibility`（实为 disp） | `sipe_disp` |
| seg_output | decode 后 | decode 后（`_PluginWrapper` 加了 `decode_outputs`） |

---

## 五、ONNX 转 TRT engine（使用 glia trt_converter）

glia 自带 ONNX 转 TRT engine 的接口：`glia.dl.utils.trt_converter.onnx2trt()`。它会自动：

1. 对 SIPE 模型跑 `onnxoptimizer` 优化
2. **自动加 `--dynamicPlugins=libtrt_ops.so`**（不需要手动指定 plugin .so）
3. 用 trtexec 构建 engine
4. 自动设置 bbox-mode metadata

### 5.1 脚本

脚本路径：`/root/repos/hermes/docs/mixedai/scripts/convert_sipe_plugin_onnx2trt.py`

### 5.2 用法

```bash
# FP32（默认）
python /root/repos/hermes/docs/mixedai/scripts/convert_sipe_plugin_onnx2trt.py \
  --onnx model_plugin_sim.onnx \
  --saveEngine model_plugin_sim_fp32.engine

# FP16
python /root/repos/hermes/docs/mixedai/scripts/convert_sipe_plugin_onnx2trt.py \
  --onnx model_plugin_sim.onnx \
  --fp16 \
  --saveEngine model_plugin_sim_fp16.engine

# FP16 + noTF32（纯精度基准）
python /root/repos/hermes/docs/mixedai/scripts/convert_sipe_plugin_onnx2trt.py \
  --onnx model_plugin_sim.onnx \
  --fp16 --noTF32 \
  --saveEngine model_plugin_sim_fp16.engine
```

注意：脚本默认 `--model-type deepac`（跟 sipe 一样调 `optimize_onnx_model`，但不强制 `--optShapes`），因为 plugin ONNX 是固定 shape（`[1,3,512,512]`），trtexec 对静态模型不接受 `--optShapes`。如果将来导出动态 shape 的 ONNX，改用 `--model-type sipe` 并传 `--optShapes`。

### 5.3 核心原理

`onnx2trt()` 内部构建 trtexec 命令时，会自动查找 glia pip 包里的 `libtrt_ops.so` 并加上 `--dynamicPlugins` 参数：

```python
# glia/dl/utils/trt_converter.py 第 238-242 行
glia_lib_path = os.path.join(glia.__path__[0], "lib")
plugin_lib_path = os.path.join(glia_lib_path, "libtrt_ops.so")
command += f" --dynamicPlugins={plugin_lib_path}"
```

`--dynamicPlugins` 让 trtexec 在构建 engine 时加载 `libtrt_ops.so`，从中找到 FFSGWCVolume plugin creator，将 ONNX 里的 plugin 节点编译进 engine。

### 5.4 注意事项

- `onnx2trt()` 对 SIPE 模型会先跑 `onnxoptimizer.optimize()`（第 201-211 行），优化标准 ONNX 算子（fuse_bn_into_conv 等）。自定义 plugin 节点（`FFSGWCVolume` in `trt.plugins` domain）不会被这些标准 pass 触碰，应该安全。但需实测验证优化后 plugin 节点完好。
- 输出 engine 默认与 ONNX 同名但扩展名改为 `.engine`，也可通过 `--saveEngine` 指定。
- `onnx2trt()` 构建完后会自动调用 `set_model_box_mode()` 设置 engine 的 bbox-mode metadata。

---

## 六、使用 glia 推理带 plugin 的 TRT engine

### 6.1 脚本

脚本路径：`/root/repos/hermes/docs/mixedai/scripts/run_sipe_plugin_inference.py`

### 6.2 用法

```bash
cd /root/repos/dex_lib/sipe-test

python /root/repos/hermes/docs/mixedai/scripts/run_sipe_plugin_inference.py \
  --engine ./onnx-w-plugin/fp32/model_plugin_sim_fp32.engine \
  --image ./val/rgb/iphones-000068.png \
  --right-image ./val/rightrgb/iphones-000068.png \
  --fx 977.724660805709 \
  --fy 945.037290277317 \
  --cx 977.755290913497 \
  --cy 537.263015148434 \
  --baseline 0.059943003472581
```

### 6.3 核心原理

glia 的 `E2ESIPEInterface(backend="trt")` 支持通过 `custom_ops_paths` 参数加载 plugin .so：

```python
config = dict(
    model_path=engine_path,
    model_input_size=(512, 512),
    cameras=[camera_cfg, camera_cfg],
    num_cameras=2,
    custom_ops_paths=[plugin_lib_path],   # 关键：传 libtrt_ops.so
)
interface = E2ESIPEInterface(backend="trt", **config)
```

glia C++ 内部加载链路：
```
E2ESIPEInterface(custom_ops_paths=[libtrt_ops.so])
  -> TensorRTInterface.SetCustomOpsPaths(paths)
    -> TensorRTEngine.Initialize(custom_ops_paths)
      -> dlopen(libtrt_ops.so)                    # 加载 .so 到进程
      -> runtime_->getPluginRegistry().loadLibrary(path)  # 注册到 TRT registry
      -> initLibNvInferPlugins()                  # 初始化 TRT 内置 plugin（只调一次）
      -> engine 反序列化                           # 从 registry 找到 FFSGWCVolume creator
```

### 6.4 输出

脚本会打印所有输出（box/mask/keypoints/disparity/depth）的 shape，并保存：
- `sipe_plugin_result.png` -- 检测框可视化
- `sipe_plugin_disparity.png` -- 视差图可视化

---

## 七、精度对比（debug_onnx_vs_trt.py）

带 plugin 的 engine 可以用 `debug_onnx_vs_trt.py` 跟原始 ONNX 对比精度：

### 7.1 用法

```bash
cd /root/repos/dex_lib/sipe-test

python debug_onnx_vs_trt.py \
  --gt ./onnx-wo-plugin/sipe_sim.onnx \
  --pred ./onnx-w-plugin/fp32/model_plugin_sim_fp32.engine \
  --image-dir ./val/rgb \
  --right-image-dir ./val/rightrgb \
  --fx 977.724660805709 --fy 945.037290277317 \
  --cx 977.755290913497 --cy 537.263015148434 \
  --baseline 0.059943003472581 \
  --plugin-lib /root/miniconda3/envs/py310/lib/python3.10/site-packages/glia/lib/libtrt_ops.so
```

### 7.2 --plugin-lib 参数说明

`debug_onnx_vs_trt.py` 加了 `--plugin-lib` 参数，在 TRT engine 加载前用 ctypes 加载 .so 并调用 `ffs_register_gwc_plugin()` 注册到 TRT registry：

```python
if plugin_lib is not None:
    import ctypes
    _lib = ctypes.CDLL(plugin_lib)
    if hasattr(_lib, "ffs_register_gwc_plugin"):
        result = _lib.ffs_register_gwc_plugin()
```

### 7.3 预期结果

FP32 plugin engine vs 原始 ONNX FP32，预期各项指标接近 0 偏差：
- 视差 EPE mean < 0.1px（plugin 数值等价）
- BBox AP@0.95 ~1.0
- 关键点 dx/dy < 0.5px

---

## 八、文件清单

### 文档和脚本

| 文件 | 说明 |
|------|------|
| `/root/repos/hermes/docs/mixedai/sipe-gwc-plugin-integration.md` | 本文档 |
| `/root/repos/hermes/docs/mixedai/scripts/convert_sipe_plugin_onnx2trt.py` | ONNX 转 TRT engine 脚本（使用 glia trt_converter） |
| `/root/repos/hermes/docs/mixedai/scripts/run_sipe_plugin_inference.py` | TRT engine 推理脚本 |

### mixedai 改动文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `mixedai/engine/exporter/plugin_export.py` | 新增 | FFSGWCVolume plugin 导出辅助模块 |
| `mixedai/scripts/export.py` | 修改 | 加 `--use-plugin` 开关 + plugin 导出逻辑 + decode_outputs |

### glia 改动文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `cpp/glia/dl/ops/tensorrt/ffs_gwc_plugin.cpp` | 新增 | plugin 实现 |
| `cpp/glia/dl/ops/tensorrt/ffs_gwc_depth_kernels.cu` | 新增 | CUDA kernel |
| `cpp/glia/dl/ops/tensorrt/ffs_gwc_plugin.hpp` | 新增 | 头文件 |

### sipe-test 改动文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `debug_onnx_vs_trt.py` | 修改 | 加 `--plugin-lib` 参数 + ctypes 加载 plugin |
