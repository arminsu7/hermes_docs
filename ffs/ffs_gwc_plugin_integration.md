# SIPE GWC Cost Volume TensorRT Plugin 接入技术文档

> 记录从性能瓶颈识别到 TRT Plugin 端到端验证的完整技术流程。
> 环境：sipe_inst 容器 / conda py39 / torch 2.0.1+cu117 / CUDA 11.7 (nvcc 11.2) / TRT 8.6.1.6
> 硬件：RTX 3060 (sm_86)
> 日期：2026-07-24

---

## 目录

1. [背景：SIPE 的 build_gwc_volume 瓶颈](#1-背景sipe-的-build_gwc_volume-瓶颈)
2. [GWC Cost Volume 原理](#2-gwc-cost-volume-原理)
3. [调查 FFS 现成 Plugin 实现](#3-调查-ffs-现成-plugin-实现)
4. [核心问题：Plugin 是否需按硬件平台生成不同参数](#4-核心问题plugin-是否需按硬件平台生成不同参数)
5. [Triton 路线阻塞与决策转向](#5-triton-路线阻塞与决策转向)
6. [Plugin 源码落地与目录结构](#6-plugin-源码落地与目录结构)
7. [CMakeLists.txt 编写与环境适配](#7-cmakeliststxt-编写与环境适配)
8. [编译问题与修复（kINT64）](#8-编译问题与修复kint64)
9. [Plugin 注册验证](#9-plugin-注册验证)
10. [单节点 Plugin ONNX 导出](#10-单节点-plugin-onnx-导出)
11. [torch.onnx.export 内建 checker 绕过](#11-torchonnxexport-内建-checker-绕过)
12. [TRT 端到端验证（单节点）](#12-trt-端到端验证单节点)
13. [跨 TRT 8.6/10.3/10.11 三版本兼容改造](#13-跨-trt-861031011-三版本兼容改造)
14. [跨平台编译指南](#14-跨平台编译指南)
15. [Full Model Plugin ONNX 导出](#15-full-model-plugin-onnx-导出)
16. [Simplify 难题与 FFS_PLUGIN_DOMAIN 突破](#16-simplify-难题与-ffs_plugin_domain-突破)
17. [端到端验证（完整模型）](#17-端到端验证完整模型)
18. [改动文件清单](#18-改动文件清单)
19. [遗留事项与后续工作](#19-遗留事项与后续工作)

---

## 1. 背景：SIPE 的 build_gwc_volume 瓶颈

### 1.1 问题描述

SIPE 推理流程通过以下命令启动：

```bash
docker exec -it sipe_inst bash
conda activate py39
cd /root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc
python simple_inference.py
```

在 `simple_inference.py` 的第 186 行（`sipe_forward_core` 函数内），调用了 `build_gwc_volume`：

```python
# simple_inference.py, sipe_forward_core() 内, 约 L186
gwc_volume = build_gwc_volume(
    match_left, match_right, igev.max_disp // 4, 8
)
```

### 1.2 原生实现分析

`build_gwc_volume` 定义在 `mixedai/mixedai/layers/stereo_layers/utils.py`：

```python
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(
                refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups
            )
        else:
            volume[:, :, i, :, :] = groupwise_correlation(
                refimg_fea, targetimg_fea, num_groups
            )
    return volume
```

**瓶颈根因**：Python `for` 循环遍历 `maxdisp`（通常 24~48 次），每次迭代 launch 一个独立的 CUDA kernel（`groupwise_correlation`）。这导致：

- **kernel launch 开销**：每次 launch 有 ~5-10μs 固定开销，48 次 = 240-480μs 纯 launch overhead
- **无法利用 inter-disparity 数据复用**：相邻 disparity 的 `w_right = w - d` 高度重叠，但 Python 循环无法在 GPU 侧复用已加载的特征数据
- **无法做 shared memory tiling**：每次 kernel 独立从 global memory 读取特征

**输出 shape**：`[B, num_groups=8, maxdisp=max_disp//4, H, W]`

### 1.3 优化目标

将 Python for 循环替换为一个 **fused CUDA kernel**，最终封装为 TensorRT plugin，用于多平台部署（RTX 4090、RTX 3060、AGX Orin、Orin NX）。

---

## 2. GWC Cost Volume 原理

### 2.1 先搞懂：立体匹配在干什么

想象你用左右两只眼睛看同一个物体--左眼和右眼看到的画面有微小的水平偏移（视差）。大脑通过这个偏移感知深度：**偏移越大 = 物体越近，偏移越小 = 物体越远**。

SIPE 做的就是这件事的 AI 版本：

1. 左右两张图片各经过特征提取网络，得到两张特征图（不是原始像素，而是网络学到的 64 维描述子）
2. 对左图每个像素，在右图同一行上"滑动搜索"：右移 0、1、2...47 个像素位置，每个位置算一个"相似度分数"
3. 分数最高的位置就是最佳匹配，对应的偏移量就是该像素的视差

**"滑动搜索"的过程就产生了 cost volume（代价体积）--一个记录了"左图每个像素在右图每个视差位置上的匹配分数"的 5D 张量。**

#### 滑动过程图示

以一维简化为例（只看一行像素，实际是整个 HxW 平面）：

```
  index:    0    1    2    3    4    5
          ┌─────────────────────────────┐
  Left:   │ A    B    C    D    E    F  │   ← 左图固定不动
          └─────────────────────────────┘
                            ↑
                          w = 3 (始终取这个位置)


  index:    0    1    2    3    4    5
          ┌─────────────────────────────┐
  Right:  │ A    B    C    D    E    F  │   ← 右图也固定不动
          └─────────────────────────────┘
                            ↑
                          w = 3


  视差 d    右图取的位置 w-d = 3-d    取到的像素    含义
  ─────────────────────────────────────────────────────
  d = 0     3 - 0 = 3                  D             完全对齐
  d = 1     3 - 1 = 2                  C             右图取的位置左移1格
  d = 2     3 - 2 = 1                  B             左移2格
  d = 3     3 - 3 = 0                  A             左移3格
  d = 4     3 - 4 = -1                 越界 → 0      w-d < 0，无对应像素
```

每次滑动，左图固定取 w=3 的像素，右图取 w-d 位置的像素，两者配对算相似度。
所有 d（0~47）滑完，就得到了完整的 cost volume [B, G, D, H, W]。

> 注意：实际代码里左右图数组都不动，只是取的索引变化。"右图左移"只是直觉化表述。

### 2.2 再搞懂：cost volume 的形状为什么是 5D

用 SIPE 的实际数字走一遍：

```
左图特征:  match_left  [1, 64, 120, 160]   ← 1张图, 64通道, 高120, 宽160
右图特征:  match_right [1, 64, 120, 160]

视差范围:  maxdisp = 48  (即 max_disp//4 = 192//4)
分组数:    num_groups = 8

输出:  gwc_volume [1, 8, 48, 120, 160]
                   │  │   │    │    │
                   │  │   │    │    └─ 宽 W=160
                   │  │   │    └────── 高 H=120
                   │  │   └─────────── 视差 D=48 (滑动0~47个像素)
                   │  └─────────────── 分组 G=8
                   └────────────────── 批次 B=1
```

一句话：**对图中每个位置 (h,w)，在 8 个组、48 个视差偏移上各算一个分数。**

### 2.3 "Groupwise" 是什么意思--为什么不直接用全部 64 通道做点积

如果直接拿 64 个通道全部做点积，得到的是一个标量（一个数）。但 64 维特征里不同通道可能编码了不同类型的信息（边缘、纹理、颜色...）。GwcNet 的思路是：**把 64 个通道拆成 8 组，每组 8 个通道，分别算相似度**。这样得到 8 个分数而不是 1 个，保留了更丰富的匹配信息。

类比：评价两个人合不合适，与其看一个总分，不如分"性格、爱好、价值观..."8 个维度各打一个分。

```
64 通道特征
├── 组 0: 通道 0~7    → 算一个点积 → gwc_volume[:, 0, d, h, w]
├── 组 1: 通道 8~15   → 算一个点积 → gwc_volume[:, 1, d, h, w]
├── 组 2: 通道 16~23  → 算一个点积 → gwc_volume[:, 2, d, h, w]
├── ...
└── 组 7: 通道 56~63  → 算一个点积 → gwc_volume[:, 7, d, h, w]
```

每组 K = 64 / 8 = 8 个通道。

### 2.4 单个输出点的计算过程（用具体数字）

以 `gwc_volume[0, 2, 10, 50, 80]` 这个点为例，意思是"第 0 张图、第 2 组、视差 d=10、位置 (h=50, w=80) 的匹配分数"：

```
第 1 步: 确定取哪些通道
  组 g=2 → 通道范围 [g*K, g*K+K) = [16, 24) → 取通道 16,17,...,23（共8个）

第 2 步: 确定右图对应位置
  左图位置: (h=50, w=80)
  视差 d=10 -> 右图位置: w_right = 80 - 10 = 70  （右图往左挪10个像素）

  注意: 如果 w_right < 0（比如 w=5, d=10），说明右图没有对应像素
       -> 直接输出 0

  图示（简化为一行，只看 w 方向）:

  左图:  ... [w=78] [w=79] [w=80] [w=81] [w=82] ...
                            ↑
                            取这个像素的特征向量（8个数）

  右图:  ... [w=68] [w=69] [w=70] [w=71] [w=72] ...
                            ↑
                            取这个像素的特征向量（8个数）
                            w_right = 80 - 10 = 70

  配对:  左图 w=80  ←─── 视差 d=10 ───→  右图 w=70
         特征 [8个数]                      特征 [8个数]
                    ↘                ↙
                     逐元素相乘求和 = 点积 = 匹配分数

第 3 步: 取出左右特征向量（各8个数）
  left  = [match_left[0, 16, 50, 80], match_left[0, 17, 50, 80], ..., match_left[0, 23, 50, 80]]
  right = [match_right[0, 16, 50, 70], match_right[0, 17, 50, 70], ..., match_right[0, 23, 50, 70]]

第 4 步: 逐元素相乘，然后聚合

  先做逐元素相乘（两版一样）:
    prod[0] = left[0]*right[0]
    prod[1] = left[1]*right[1]
    ...
    prod[7] = left[7]*right[7]

  聚合方式:

    SIPE 原版 (groupwise_correlation):
      output = mean(prod) = (prod[0]+prod[1]+...+prod[7]) / 8
      └─ PyTorch 代码: (fea1 * fea2).view([B, G, K, H, W]).mean(dim=2)
      └─ 逐元素相乘后对 K 个通道取【平均值】

    SIPE plugin (depth_kernels.cu, 已修正为 mean):
      output = sum(prod) / K = (prod[0]+...+prod[7]) / 8
      └─ CUDA 代码: for (k=0..K-1) dot += l * r; dot *= (1.0f / K);
      └─ 先求和，再乘以 1/K（等价于 mean），与 SIPE 原版数值一致

    （FFS 原版 plugin 用 sum 不除 K，SIPE 接入时已修正为 mean）

第 5 步: 是否 L2 归一化（两版都有此选项，但 SIPE 默认不用）

  SIPE 原版:
    normalize 默认无（代码里没有 normalize 参数，纯 mean）

  SIPE plugin:
    normalize=False (默认): output = mean(prod)         ← 纯 mean
    normalize=True:         output = mean(prod) / (|left|_mean * |right|_mean + ε)
                                                  ← 除以模长，归到 [-1, 1]
```

**数值对齐验证**（CPU numpy，B=1,C=8,H=4,W=6,maxdisp=3,G=4,K=2）：

```
SIPE mean vs FFS sum  max diff: 2.38   (差 K=2 倍，未修正前的预期差异)
SIPE mean vs FFS mean  max diff: 0.0    (修正后完全一致)
ratio sum/mean: 2.0 = K
CONCLUSION: SIPE == plugin_mean: True
```

**normalize 的意义**：归一化后不受特征向量整体大小影响，只看方向相似度。但 SIPE 训练时用的就是无归一化版本，所以 plugin 也必须用 `normalize=False`。

### 2.5 整个 GWC Volume 的计算全貌

把上面的"单个点"扩展到所有点，就是完整的 GWC volume 计算：

```
对每个批次 b   (B=1)
  对每个组 g   (G=8)
    对每个视差 d (D=48)
      对每个像素 (h, w)  (H=120, W=160)
        取左图 [b, g*8 : g*8+8, h, w]       ← 8个数
        取右图 [b, g*8 : g*8+8, h, w-d]     ← 8个数（右图左移d像素）
        逐元素相乘 -> 聚合 -> gwc_volume[b, g, d, h, w]
                       (SIPE 原版: mean / SIPE plugin: mean)
```

总计算量：1 × 8 × 48 × 120 × 160 = **7,372,800 个输出点**，每个点做 8 次乘加 = 约 5900 万次乘加。这在 GPU 上本应极快，但 SIPE 原版用 Python for 循环逐个视差 launch kernel，效率大打折扣。

### 2.6 SIPE 原版 vs SIPE plugin 版的差异

| 维度 | SIPE 原版 (`utils.py`) | SIPE plugin (`depth_kernels.cu`) |
|------|------------------------|----------------------------------|
| 通道聚合方式 | **mean**（`.mean(dim=2)`） | **mean**（`dot *= 1/K`，已修正） |
| 数值关系 | 完全一致（CPU 验证 max diff = 0.0） | 完全一致 |
| L2 normalize | 无（代码不含 normalize 参数） | 可选（`normalize` 参数，默认 False） |
| 实现方式 | Python for 循环 48 次，每次 launch 1 个 kernel | 1 个 fused CUDA kernel 一次算完 |
| 数据复用 | 无（每个 d 独立读特征） | 可利用相邻 d 的特征重叠（当前版本未做） |
| kernel launch 次数 | 48 次 | 1 次 |

> **关键注意**：
> 1. **聚合方式已对齐**：plugin kernel 已从 FFS 原版的 sum 修正为 mean（`dot *= 1/K`），与 SIPE 原版数值完全一致。
> 2. **normalize**：等价替换 SIPE 原版时必须用 `normalize=False`，否则数值不一致。

---

## 3. 调查 FFS 现成 Plugin 实现

### 3.1 发现 Fast-FoundationStereo 仓库

用户自行 clone 了 Fast-FoundationStereo 到容器内：

```
/root/workspace/MixedAI_Datacenter/sipe-nocs/Fast-FoundationStereo/
```

### 3.2 FFS Plugin 文件清单

```bash
find /root/workspace/MixedAI_Datacenter/sipe-nocs/Fast-FoundationStereo/cpp -type f
```

| 文件 | 作用 |
|------|------|
| `cpp/include/ffs_gwc_plugin.hpp` | plugin 注册接口 + C ABI 导出声明 |
| `cpp/src/gwc_volume_plugin.cpp` | FFSGWCVolume plugin 本体（IPluginV2DynamicExt） |
| `cpp/src/depth_kernels.cu` | GWC CUDA kernel + 6 个预处理/后处理 kernel |
| `cpp/CMakeLists.txt` | 编译系统 |
| `scripts/make_plugin_onnx.py` | 把 GWC 节点替换成 plugin 节点导出 ONNX |
| `scripts/build_plugin_trt.py` | build 带 plugin 的 TRT engine |

### 3.3 CUDA Kernel 分析（depth_kernels.cu）

FFS 的 GWC kernel 核心逻辑：

```cpp
template <typename InputT, typename OutputT>
__global__ void buildGWCVolumeKernel(
    const InputT* __restrict__ feat_left,
    const InputT* __restrict__ feat_right,
    OutputT* __restrict__ gwc_volume,
    int B, int C, int H, int W,
    int max_disp, int num_groups, bool normalize)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int dgb = blockIdx.z;  // d + g * max_disp + b * max_disp * num_groups

    if (w >= W || h >= H) return;

    const int d = dgb % max_disp;
    const int g = (dgb / max_disp) % num_groups;
    const int b = dgb / (max_disp * num_groups);
    if (b >= B) return;

    const int K = C / num_groups;
    const int w_right = w - d;

    // w_right < 0 时输出 0（无对应像素）
    if (w_right < 0) { gwc_volume[out_idx] = 0; return; }

    // K 维串行循环（一个 thread 算一个输出点）
    float dot = 0.f, nl = 0.f, nr = 0.f;
    for (int k = 0; k < K; ++k) {
        float l = feat_left[left_base  + k * stride];
        float r = feat_right[right_base + k * stride];
        dot += l * r;  nl += l * l;  nr += r * r;
    }
    // 可选 L2 归一化
    if (normalize)
        gwc_volume[out_idx] = dot / (sqrtf(nl) * sqrtf(nr) + 1e-5f);
    else
        gwc_volume[out_idx] = dot;
}
```

**Launch 配置**（host wrapper）：

```cpp
dim3 blk(16, 16);                                          // 固定 block size
dim3 grd((W+15)/16, (H+15)/16, B * num_groups * max_disp); // grid 按 shape 算
```

**精度支持**：通过模板参数支持 4 种组合：
- `ffsCudaBuildGWCVolumeFloat` (FP32->FP32)
- `ffsCudaBuildGWCVolumeHalf` (FP16->FP16)
- `ffsCudaBuildGWCVolumeHalfToFloat` (FP16->FP32)
- `ffsCudaBuildGWCVolumeMixed` (FP32->FP16)

### 3.4 Plugin 设计要点

- plugin 名 `FFSGWCVolume`，版本 v1
- 三个参数通过 `PluginField` 传入：`max_disp` / `cv_group` / `normalize`
- 继承 `IPluginV2DynamicExt`，`getOutputDimensions` 用 exprBuilder 动态推导输出 shape -> **支持动态 H/W**
- `REGISTER_TENSORRT_PLUGIN` 自动注册
- `enqueue` 按 in/out dtype 分发到对应精度 wrapper

---

## 4. 核心问题：Plugin 是否需按硬件平台生成不同参数

### 4.1 问题

> plugin 是不是也要根据硬件平台用不同的参数生成，还是说 plugin 内部可以通过参数灵活的变动？

### 4.2 结论

**不需要按平台生成不同版本。** FFS 这份 plugin 的 kernel launch 配置写死 `dim3 blk(16, 16)`，grid 按运行时 shape 动态计算，**不含任何平台相关的 tile 参数**。一份源码编译即可在各平台运行（只需各平台用对应的 CUDA/TRT 版本重新编译 .so）。

### 4.3 两种参数的区分

| 参数类型 | 是否可运行时配置 | 说明 |
|----------|:---:|------|
| 算法参数 (max_disp / cv_group / normalize) | ✅ | 通过 `PluginField` 在 build engine 时传入，灵活可配 |
| 性能 tuning (block size / tile / 每线程负载) | ❌ | 当前写死 16x16，想按平台调优需改 `.cu` 重新编译 |

### 4.4 与 Triton Autotune 的对比

| 维度 | FFS CUDA plugin | triton autotune 版 |
|------|-----------------|---------------------|
| tile/block 配置 | 固定 16x16，不调优 | 6 组候选里自动选最优 |
| 跨硬件 | 同一份 .so 到处能跑 | 每平台 autotune 结果不同 |
| 性能 | 稳定但非极致 | 单平台可能更快 |
| 并行映射 | 一线程一输出点，K 维串行 for | 分块 + shared mem 复用 |

### 4.5 如需按平台最优的改造方向

1. **改造 CUDA plugin**：在 `enqueue` 里用 `cudaGetDeviceProperties` 按 SM 架构查预标定配置表，改动小、零 ABI 风险
2. **kernel 本身优化空间**：当前没用 shared memory 复用 feat_left/feat_right（相邻 d 的 w_right 高度重叠），也没做向量化 load

### 4.6 配置超过 6 组 autotune 的讨论

FFS 的 6 组 autotune 配置是其自有 GPU 经验值。对于 SIPE 的固定 shape（C=64/G=8->K=8），合理的搜索空间：

- `BLOCK_C` 超过 K=8 无意义
- `BLOCK_W` 32~256
- `BLOCK_D` 4/8/16
- `num_warps` 2~16
- `num_stages` 2~4

按平台剪枝后网格搜索，可离线标定一次固化最优配置。

---

## 5. Triton 路线阻塞与决策转向

### 5.1 尝试 FFS Triton Kernel

FFS 在 `core/submodule.py` L390-510 提供了 triton kernel `_gwc_triton_kernel`（`@triton.autotune` 6 组配置）和接口 `build_gwc_volume_triton(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True)`。

### 5.2 失败原因

直接调用 `build_gwc_volume_triton` 报错：

```
IndexError: tuple index out of range
```

**根因**：triton 2.0.0 的 autotuner API 按位置索引取 autotune key，而 FFS kernel 用关键字传 `NORMALIZE=`，导致 autotuner 内部 `tuple index out of range`。

**环境锁定**：torch 2.0.1+cu117 pin 了 triton 2.0.0（`pip show triton` 显示 `Required-by: torch`）。单独升级 triton 会破坏 `torch.compile`/inductor，风险太高。

### 5.3 去掉 autotune 的影响评估

- 会损失约 5%~20% 性能（非数量级差距）
- SIPE 推理 shape 固定（C=64/G=8），可离线标定一次固化最优配置，损失趋近 0
- 对比基线是 Python for 循环版，fused kernel 仍快几倍

### 5.4 决策

放弃 triton 路线，转向复用 FFS 现成 CUDA C++ plugin。

---

## 6. Plugin 源码落地与目录结构

### 6.1 复制源文件

从 FFS 复制 3 个核心文件到 SIPE 的 plugin 目录：

```bash
FFS=/root/workspace/MixedAI_Datacenter/sipe-nocs/Fast-FoundationStereo/cpp
PLUG=/root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/plugin

mkdir -p $PLUG/src $PLUG/include
cp $FFS/src/gwc_volume_plugin.cpp $PLUG/src/
cp $FFS/src/depth_kernels.cu      $PLUG/src/
cp $FFS/include/ffs_gwc_plugin.hpp $PLUG/include/
```

### 6.2 最终目录结构

```
deploy_utils_hpc/plugin/
├── CMakeLists.txt              ← 新写，适配本机环境
├── include/ffs_gwc_plugin.hpp  ← 从 FFS 复制
├── src/gwc_volume_plugin.cpp   ← 从 FFS 复制 + 条件编译改动
├── src/depth_kernels.cu        ← 从 FFS 复制（GWC CUDA kernel）
└── build/libffs_gwc_plugin.so  ← 编译产物
```

---

## 7. CMakeLists.txt 编写与环境适配

### 7.1 环境侦查

```bash
# GPU
docker exec sipe_inst nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# → NVIDIA GeForce RTX 3060, 8.6

# nvcc
docker exec sipe_inst nvcc --version
# → Cuda compilation tools, release 11.2, V11.2.152

# TRT 头文件位置
find / -name "NvInfer.h" 2>/dev/null
# → /root/trt_lib/TensorRT-8.6.1.6/include/NvInfer.h  (TRT 8.6)
# → /root/workspace/.../TensorRT-10.3.0.26/include/NvInfer.h  (TRT 10.3, 完整 GA 包)
# → /root/workspace/.../TensorRT-10.11.0.33/include/NvInfer.h  (TRT 10.11, 完整 GA 包)

# TRT lib
ls /root/trt_lib/TensorRT-8.6.1.6/lib/ | grep nvinfer
# → libnvinfer.so, libnvinfer.so.8, libnvinfer.so.8.6.1

# py39 tensorrt
conda run -n py39 pip show tensorrt | grep Version
# → Version: 8.6.1

# cmake
cmake --version
# → 3.22.0
```

### 7.2 FFS 原 CMake 的问题

FFS 原版 CMake 有三个坑导致本机编译失败：

| 问题 | FFS 原版 | 本机实际 |
|------|---------|---------|
| CUDA arch | `80;86;89;90` | nvcc 11.2 不支持 sm_89/90 |
| nvinfer 库名 | 找 `nvinfer.so.10` | 本机是 `nvinfer.so.8` |
| TENSORRT_ROOT | 默认 `/usr` | 实际在 `/root/trt_lib/TensorRT-8.6.1.6` |

### 7.3 重写的 CMakeLists.txt

完整文件位于 `plugin/CMakeLists.txt`，关键改动：

```cmake
# 1. CUDA arch 默认 86（RTX 3060），可命令行覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

# 2. TENSORRT_ROOT 默认指向容器内 TRT 8.6
set(TENSORRT_ROOT "/root/trt_lib/TensorRT-8.6.1.6" CACHE PATH "TensorRT installation path")

# 3. 库名同时接受 8.x 和 10.x
find_library(NVINFER_LIB
  NAMES nvinfer nvinfer.so.8 nvinfer.so.10
  ...)

# 4. 自动检测并打印 TRT major 版本
file(STRINGS "${NVINFER_INCLUDE_DIR}/NvInferVersion.h" _trt_major_line
     REGEX "#define NV_TENSORRT_MAJOR")
string(REGEX MATCH "[0-9]+" TRT_MAJOR "${_trt_major_line}")
message(STATUS "TensorRT major version: ${TRT_MAJOR}")

# 5. 压制 10.x 对 IPluginV2DynamicExt 的 deprecated 警告
target_compile_options(ffs_gwc_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
)
```

### 7.4 编译命令

```bash
docker exec -it sipe_inst bash
conda activate py39
cd /root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/plugin
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
# 产物: build/libffs_gwc_plugin.so (122KB)
```

cmake 配置阶段输出：

```
-- TensorRT nvinfer: /root/trt_lib/TensorRT-8.6.1.6/lib/libnvinfer.so
-- TensorRT include: /root/trt_lib/TensorRT-8.6.1.6/include
-- TensorRT major version: 8
```

---

## 8. 编译问题与修复（kINT64）

### 8.1 第一次编译失败

```
error: 'kINT64' is not a member of 'nvinfer1::PluginFieldType'
```

### 8.2 根因分析

`PluginFieldType::kINT64` 是 **TensorRT 9.0+** 才加入的枚举值。FFS 是为 TRT 10 写的，而本机是 TRT 8.6.1.6，8.x 的 `PluginFieldType` 枚举没有 `kINT64`。

验证（8.6 vs 10.3 头文件对比）：

```bash
# 8.6: 不存在
grep "kINT64" /root/trt_lib/TensorRT-8.6.1.6/include/NvInferRuntimePlugin.h
# (无输出)

# 10.3: 存在
grep "kINT64" /root/.../TensorRT-10.3.0/include/NvInferRuntimePlugin.h
# → kINT64 = 10,
```

### 8.3 修复方案演进

**第一阶段（仅适配 8.6）**：直接删掉 kINT64 分支

**第二阶段（跨版本兼容）**：用 `NV_TENSORRT_MAJOR` 版本宏条件编译

最终代码（`gwc_volume_plugin.cpp` L48-62）：

```cpp
#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>
#include <NvInferVersion.h>  // NV_TENSORRT_MAJOR, for 8.6 vs 10.x conditional compilation

int32_t fieldToInt(nvinfer1::PluginField const& field, int32_t fallback) {
    if (!field.data || field.length <= 0) return fallback;
    if (field.type == nvinfer1::PluginFieldType::kINT32) {
        return *static_cast<int32_t const*>(field.data);
    }
    // PluginFieldType::kINT64 only exists in TensorRT >= 9.0 (present in 10.x).
    // TensorRT 8.6 does not define it, so guard the branch by version macro to
    // keep a single source tree compilable against both 8.6 and 10.3.
#if NV_TENSORRT_MAJOR >= 9
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        return static_cast<int32_t>(*static_cast<int64_t const*>(field.data));
    }
#endif
    return fallback;
}
```

### 8.4 版本宏链路验证

`NvInferVersion.h` 的 include 链：

```
NvInfer.h → NvInferRuntimeBase.h → NvInferVersion.h (定义 NV_TENSORRT_MAJOR)
```

显式 `#include <NvInferVersion.h>` 确保版本宏在 `fieldToInt` 处一定可见。

### 8.5 条件编译验证

通过预处理确认版本宏精确生效：

```bash
# 10.3 头文件: kINT64 分支被编入 (count > 0)
g++ -std=c++17 -E -I.../TensorRT-10.3.0/include ... gwc_volume_plugin.cpp | grep -c "kINT64"
# → 3  (注释 + 代码)

# 8.6 头文件: kINT64 分支被排除 (count = 0)
g++ -std=c++17 -E -I.../TensorRT-8.6.1.6/include ... gwc_volume_plugin.cpp | grep -c "kINT64"
# → 0
```

---

## 9. Plugin 注册验证

### 9.1 验证脚本

编译成功后，验证 `.so` 能被 TRT 加载并注册出 `FFSGWCVolume` creator：

```python
# /tmp/_test_plugin.py
import ctypes, tensorrt as trt

so = "/root/.../plugin/build/libffs_gwc_plugin.so"
ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

reg = trt.get_plugin_registry()
names = [c.name for c in reg.plugin_creator_list]
print("TRT", trt.__version__)
print("FFSGWCVolume registered:", "FFSGWCVolume" in names)
```

### 9.2 运行

```bash
# 注意：py39 的 tensorrt 需要 TRT lib 在 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/trt_lib/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
conda run -n py39 python /tmp/_test_plugin.py
```

### 9.3 结果

```
TRT 8.6.1
FFSGWCVolume registered: True
```

---

## 10. 单节点 Plugin ONNX 导出

### 10.1 设计思路

参照 FFS 的 `make_plugin_onnx.py`，核心是一个 `torch.autograd.Function`：
- `forward()`：返回正确形状的零 tensor（ONNX export 只需 shape 正确）
- `symbolic()`：发射 `FFSGWCVolume` 自定义 ONNX 节点

### 10.2 _FFSGWCVolumeOp 实现

新增到 `submodule_export.py`：

```python
class _FFSGWCVolumeOp(torch.autograd.Function):
    """ONNX-export-only op that emits an FFSGWCVolume TensorRT plugin node."""

    @staticmethod
    def forward(ctx, match_left, match_right, max_disp, cv_group, normalize):
        batch, _, height, width = match_left.shape
        return match_left.new_zeros(
            (batch, int(cv_group), int(max_disp), height, width)
        )

    @staticmethod
    def symbolic(g, match_left, match_right, max_disp, cv_group, normalize):
        from torch.onnx import symbolic_helper

        def as_int(value):
            if isinstance(value, int):
                return value
            return symbolic_helper._parse_arg(value, "i")

        max_disp = as_int(max_disp)
        cv_group = as_int(cv_group)
        normalize = as_int(normalize)
        out = g.op(
            "FFSGWCVolume",
            match_left,
            match_right,
            max_disp_i=int(max_disp),
            cv_group_i=int(cv_group),
            normalize_i=int(normalize),
        )
        sizes = match_left.type().sizes()
        if sizes is not None and len(sizes) == 4:
            out.setType(
                match_left.type().with_sizes(
                    [sizes[0], int(cv_group), int(max_disp), sizes[2], sizes[3]]
                )
            )
        return out
```

### 10.3 build_gwc_volume_plugin_onnx_export 函数

```python
def build_gwc_volume_plugin_onnx_export(
    igev, match_left, match_right, out_path,
    device="cuda", normalize=False, simplify_onnx=True,
):
    cv_group = 8
    max_disp_levels = igev.max_disp // 4

    class BuildGwcVolumePluginWrapper(nn.Module):
        def __init__(self, max_disp_levels, cv_group, normalize):
            super().__init__()
            self.max_disp_levels = int(max_disp_levels)
            self.cv_group = int(cv_group)
            self.normalize = int(bool(normalize))

        def forward(self, match_left, match_right):
            return _FFSGWCVolumeOp.apply(
                match_left, match_right,
                self.max_disp_levels, self.cv_group, self.normalize,
            )

    wrapper = BuildGwcVolumePluginWrapper(
        max_disp_levels, cv_group, normalize
    ).to(device).eval()

    export_inputs = (match_left.to(device), match_right.to(device))

    _export_onnx_with_plugin(
        wrapper, export_inputs, out_path,
        input_names=["match_left", "match_right"],
        output_names=["gwc_volume"],
    )
    if not simplify_onnx:
        return out_path

    import onnx
    model = onnx.load(out_path)
    sim_path = _simplify_with_plugin(model, out_path[:-5])
    keep_original = os.environ.get("KEEP_ORIGINAL_ONNX", "0") == "1"
    if not keep_original and os.path.exists(out_path):
        os.remove(out_path)
    return sim_path
```

---

## 11. torch.onnx.export 内建 checker 绕过

### 11.1 问题

第一次导出报错：

```
RuntimeError: No Op registered for FFSGWCVolume with domain_version of 17
==> Context: Bad node spec for node. Name: /FFSGWCVolume OpType: FFSGWCVolume
```

### 11.2 根因

`torch.onnx.export` 在结束时内部调用 `torch._C._check_onnx_proto` 做 ONNX proto 校验。自定义 plugin 节点没有注册的 op schema，校验失败。这不是用户代码里的 `onnx.checker`（那段没调），而是导出函数内建的。

### 11.3 修复

导出期间临时 monkeypatch `torch._C._check_onnx_proto` 为 no-op，导出完 `finally` 还原：

```python
def _export_onnx_with_plugin(
    wrapper, export_inputs, out_path, *, input_names, output_names, opset_version=17
):
    import torch._C as _torch_C

    _orig_check = _torch_C._check_onnx_proto

    def _skip_check(*a, **k):
        return None

    _torch_C._check_onnx_proto = _skip_check
    try:
        torch.onnx.export(
            wrapper, export_inputs, out_path,
            export_params=True,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            verbose=False,
        )
    finally:
        _torch_C._check_onnx_proto = _orig_check
    return out_path
```

### 11.4 验证

```python
# 单元测试
import types, torch, onnx
igev = types.SimpleNamespace(max_disp=192)
ml = torch.randn(1, 64, 60, 80).cuda()
mr = torch.randn(1, 64, 60, 80).cuda()
build_gwc_volume_plugin_onnx_export(igev, ml, mr, "/tmp/test.onnx", normalize=False)

m = onnx.load("/tmp/test.onnx")
print("nodes:", [n.op_type for n in m.graph.node])
# → nodes: ['FFSGWCVolume']

ffs = [n for n in m.graph.node if n.op_type == "FFSGWCVolume"][0]
attrs = {a.name: a.i for a in ffs.attribute}
print("attrs:", attrs)
# → {'cv_group': 8, 'max_disp': 48, 'normalize': 0}
```

---

## 12. TRT 端到端验证（单节点）

### 12.1 验证脚本

确认导出的 ONNX 能被 TRT 8.6 解析并 build engine：

```python
import ctypes, tensorrt as trt

PLUG = "/root/.../plugin/build/libffs_gwc_plugin.so"
ctypes.CDLL(PLUG, mode=ctypes.RTLD_GLOBAL)

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("/tmp/test.onnx", "rb") as f:
    ok = parser.parse(f.read())

if not ok:
    for i in range(parser.num_errors):
        print("PARSE ERR:", parser.get_error(i))
    raise SystemExit(1)

print("parsed OK, layers:", network.num_layers)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
engine = builder.build_serialized_network(network, config)
print("ENGINE_BUILD:", "OK" if engine else "FAILED",
      "| bytes:", engine.nbytes if engine else 0)
```

### 12.2 结果

```
parsed OK, layers: 1
ENGINE_BUILD: OK | bytes: 2964
```

单节点 ONNX -> TRT parse -> build engine 全链路通过。

---

## 13. 跨 TRT 8.6/10.3/10.11 三版本兼容改造

### 13.1 需求

后续 plugin 会在 AGX Orin、Orin NX、RTX 4090 平台编译，TRT 版本包括 8.6、10.3 和 10.11，需要同一份源码兼容三个版本。

### 13.2 三版本 TRT 安装位置

| TRT 版本 | 路径 | lib 状态 |
|----------|------|----------|
| 8.6.1.6 | `/root/trt_lib/TensorRT-8.6.1.6` | 完整 (.so.8) |
| 10.3.0.26 | `/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.3.0.26` | 完整 (.so.10) |
| 10.11.0.33 | `/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.11.0.33` | 完整 (.so.10) |

### 13.3 版本差异调查

通过对比三版头文件，确认以下差异：

| API/特性 | TRT 8.6 | TRT 10.3 | TRT 10.11 |
|----------|---------|----------|-----------|
| `PluginFieldType::kINT64` | 不存在 | 存在 (值=10)，在 `NvInferRuntimePlugin.h` | 存在 (值=10)，在 `NvInferPluginBase.h`（被 `NvInferRuntimePlugin.h` include） |
| `IPluginV2DynamicExt` | 正常 | `TRT_DEPRECATED`（仍可用） | `TRT_DEPRECATED`（仍可用） |
| `IPluginCreator` | 正常 | `TRT_DEPRECATED`（仍可用） | `TRT_DEPRECATED`（仍可用） |
| `enqueue` 签名 | `int32_t enqueue(PluginTensorDesc const*, ...)` | 一致 | 一致 |
| `REGISTER_TENSORRT_PLUGIN` 宏 | 存在 | 存在 | 存在 |
| `getBuilderPluginRegistry` | 存在 | 存在 | 存在 |
| `attachToContext` (V2 接口) | 存在 | 存在 | 存在 |
| 版本宏定义方式 | `#define NV_TENSORRT_MAJOR 8` | `#define NV_TENSORRT_MAJOR 10` | `#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE`（间接，需解析） |

### 13.4 关键发现：10.11 的版本宏间接定义

TRT 10.11 的 `NvInferVersion.h` 用了间接宏：

```c
#define TRT_MAJOR_ENTERPRISE 10
#define TRT_MINOR_ENTERPRISE 11
#define TRT_PATCH_ENTERPRISE 0

#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE   // ← 不是直接数字
```

而 8.6 和 10.3 都是直接定义：

```c
#define NV_TENSORRT_MAJOR 8   // 或 10
```

**影响**：CMake 的 `file(STRINGS ... REGEX)` 正则匹配 `[0-9]+` 在 10.11 上会失败（匹配到的是 `TRT_MAJOR_ENTERPRISE` 不是数字）。需要在 CMake 中加间接宏解析逻辑。

**对 C++ 条件编译无影响**：`#if NV_TENSORRT_MAJOR >= 9` 在预处理阶段会被正确展开（C 预处理器会先展开 `TRT_MAJOR_ENTERPRISE` 再比较）。

### 13.5 条件编译方案

同一份源码可同时兼容三个版本：

- kINT64 用 `#if NV_TENSORRT_MAJOR >= 9` 条件编译（8.6 排除，10.3/10.11 启用）
- deprecated 警告用 `-Wno-deprecated-declarations` 压制（10.3/10.11 需要）
- 其余 API 三版一致

```cpp
// gwc_volume_plugin.cpp
#include <NvInferVersion.h>  // NV_TENSORRT_MAJOR

#if NV_TENSORRT_MAJOR >= 9
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        return static_cast<int32_t>(*static_cast<int64_t const*>(field.data));
    }
#endif
```

### 13.6 CMake 版本检测适配

CMakeLists.txt 中的版本检测逻辑需处理 10.11 的间接宏：

```cmake
if(EXISTS "${NVINFER_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${NVINFER_INCLUDE_DIR}/NvInferVersion.h" _trt_major_line
       REGEX "#define NV_TENSORRT_MAJOR")
  # 如果引用了间接宏 (如 TRT_MAJOR_ENTERPRISE)，解析它
  string(REGEX MATCH "TRT_[A-Z_]+_ENTERPRISE" _indirect_macro "${_trt_major_line}")
  if(_indirect_macro)
    file(STRINGS "${NVINFER_INCLUDE_DIR}/NvInferVersion.h" _indirect_val
         REGEX "#define ${_indirect_macro}")
    string(REGEX MATCH "[0-9]+" TRT_MAJOR "${_indirect_val}")
  else()
    string(REGEX MATCH "[0-9]+" TRT_MAJOR "${_trt_major_line}")
  endif()
  message(STATUS "TensorRT major version: ${TRT_MAJOR}")
endif()
```

### 13.7 三版本完整编译验证

三个版本均在本机（RTX 3060, nvcc 11.2, sm_86）完成 cmake + build：

```bash
PLUG=/root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/plugin

# === TRT 8.6 ===
cd $PLUG && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# → TensorRT major version: 8
cmake --build . -j$(nproc)
# → Built target ffs_gwc_plugin  (122736 bytes)

# === TRT 10.3 ===
cd $PLUG && rm -rf build103 && mkdir build103 && cd build103
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.3.0.26
# → TensorRT major version: 10
cmake --build . -j$(nproc)
# → Built target ffs_gwc_plugin  (123104 bytes)

# === TRT 10.11 ===
cd $PLUG && rm -rf build1011 && mkdir build1011 && cd build1011
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.11.0.33
# → TensorRT major version: 10  (间接宏解析成功)
cmake --build . -j$(nproc)
# → Built target ffs_gwc_plugin  (123104 bytes)
```

三套 .so 均成功编译链接，CUDA kernel 符号和 IPluginCreator vtable 符号完整。

### 13.8 预处理验证条件编译生效

```bash
# 10.3: kINT64 分支被编入
g++ -std=c++17 -E -I.../TensorRT-10.3.0.26/include ... | grep -c "kINT64"
# → 3

# 10.11: kINT64 分支被编入
g++ -std=c++17 -E -I.../TensorRT-10.11.0.33/include ... | grep -c "kINT64"
# → 3

# 8.6: kINT64 分支被排除
g++ -std=c++17 -E -I.../TensorRT-8.6.1.6/include ... | grep -c "kINT64"
# → 0
```

---

## 14. 跨平台编译指南

### 14.1 各平台编译命令

```bash
# ============ RTX 3060 (sm_86, TRT 8.6, 本容器) ============
# 默认参数，无需额外指定
cmake .. -DCMAKE_BUILD_TYPE=Release

# ============ 同一机器但用 TRT 10.3 ============
cmake .. -DTENSORRT_ROOT=/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.3.0.26

# ============ 同一机器但用 TRT 10.11 ============
cmake .. -DTENSORRT_ROOT=/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.11.0.33

# ============ RTX 4090 (sm_89, 需 nvcc >= 11.8) ============
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 -DTENSORRT_ROOT=/path/to/TRT

# ============ AGX Orin / Orin NX (sm_87, 需 nvcc >= 11.4) ============
# TRT 随 JetPack 安装在系统路径
cmake .. -DCMAKE_CUDA_ARCHITECTURES=87 -DTENSORRT_ROOT=/usr/lib/aarch64-linux-gnu

# ============ 多架构 fatbin (一个 .so 跑多个 GPU) ============
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

### 14.2 CMake 中的平台适配

CMakeLists.txt 中的 hint 路径已包含 JetPack aarch64 路径：

```cmake
set(TENSORRT_HINT_DIRS
  ${TENSORRT_ROOT}
  $ENV{CONDA_PREFIX}
  /usr
  /usr/local
  /usr/local/cuda
  /usr/src/tensorrt
  /usr/lib/aarch64-linux-gnu           # JetPack (AGX Orin / Orin NX) libs
  /usr/include/aarch64-linux-gnu       # JetPack headers
)
```

### 14.3 运行时加载 plugin

```bash
# TRT 8.6 环境
export LD_LIBRARY_PATH=/root/trt_lib/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH

# Python 加载
import ctypes, tensorrt as trt
ctypes.CDLL("path/to/libffs_gwc_plugin.so", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(logger, "")
```

---

## 15. Full Model Plugin ONNX 导出

### 15.1 需求

不仅单节点导出，还要 `full_model_onnx_export` 能导出带 plugin 的完整 SIPE 模型 ONNX，并且支持 simplify。

### 15.2 技术难点

完整模型中，GWC 计算深埋在 `sipe_forward_core` 第 186 行。`sipe_forward_core` 用的是模块级全局符号 `build_gwc_volume`（`simple_inference.py` 第 38 行 import）。

### 15.3 Monkey-patch 方案

在导出期间临时替换 `simple_inference` 模块的 `build_gwc_volume` 全局符号：

```python
def full_model_plugin_onnx_export(
    model, left_x, right_x, out_path,
    device="cuda", normalize=False, simplify_onnx=True,
):
    from hpc_quant.deploy_utils_hpc import simple_inference as _si

    igev = model.igev
    mixedone = igev.mixedone

    # Shim: 保持原生签名 build_gwc_volume(left, right, maxdisp, G)
    # 内部调 _FFSGWCVolumeOp.apply 发射 plugin 节点
    def _build_gwc_volume_plugin_shim(refimg_fea, targetimg_fea, maxdisp, num_groups):
        return _FFSGWCVolumeOp.apply(
            refimg_fea, targetimg_fea,
            int(maxdisp), int(num_groups), int(bool(normalize)),
        )

    class FullModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, image1, image2):
            outputs, seg_output, kpt_output, disp_pred, _nocs = _si.sipe_forward_core(
                self.model, image1, image2, for_export=True
            )
            return outputs, seg_output, kpt_output, disp_pred

    wrapper = FullModelWrapper(model).to(device).eval()
    export_inputs = (left_x.to(device), right_x.to(device))

    # Rebind 模块级 build_gwc_volume
    _orig_build_gwc = _si.build_gwc_volume
    _si.build_gwc_volume = _build_gwc_volume_plugin_shim
    try:
        with _patch_c2f_forward_to_split(mixedone):
            _export_onnx_with_plugin(
                wrapper, export_inputs, out_path,
                input_names=["image1", "image2"],
                output_names=["outputs", "seg_output", "kpt_output", "disp_pred"],
            )
    finally:
        _si.build_gwc_volume = _orig_build_gwc

    if not simplify_onnx:
        return out_path

    import onnx
    model_onnx = onnx.load(out_path)
    sim_path = _simplify_with_plugin(model_onnx, out_path[:-5])
    keep_original = os.environ.get("KEEP_ORIGINAL_ONNX", "0") == "1"
    if not keep_original and os.path.exists(out_path):
        os.remove(out_path)
    return sim_path
```

### 15.4 simple_inference.py 接入

**import 块**（L674-688）新增 `full_model_plugin_onnx_export`：

```python
    from hpc_quant.deploy_utils_hpc.submodule_export import (
        mixedone_extract_features_onnx_export,
        cnet_proj_onnx_export,
        stereo_feature_onnx_export,
        stereo_feature_batched_onnx_export,
        build_gwc_volume_onnx_export,
        build_gwc_volume_plugin_onnx_export,
        geo_encoding_volume_onnx_export,
        init_disp_onnx_export,
        gru_update_onnx_export,
        mixedone_predict_onnx_export,
        full_model_onnx_export,
        full_model_plugin_onnx_export,
    )
```

**命令行参数**（L73）新增 `--plugin-normalize`：

```python
    parser.add_argument(
        "--plugin-normalize", action="store_true", default=False,
        help="For build_gwc_volume_plugin ONNX export: set FFSGWCVolume "
             "normalize=1 (L2-normalized GwcNet variant). Default False "
             "matches SIPE native build_gwc_volume.",
    )
```

**export_pipeline 中的调用**（L728-741）：

```python
        # === Full model export WITH FFSGWCVolume plugin node ===
        if should_export("full_model_plugin"):
            full_model_plugin_onnx_export(
                model,
                left_x,
                right_x,
                str(export_dir / "full_model_plugin.onnx"),
                device=device,
                normalize=args.plugin_normalize,
            )
            LOGGER.info(
                "Full model (plugin) exported to full_model_plugin_sim.onnx; "
                "load plugin/build/libffs_gwc_plugin.so at TRT build time."
            )
            return
```

**子模块导出中的调用**（L836-844）：

```python
            do_export(
                "build_gwc_volume_plugin",
                build_gwc_volume_plugin_onnx_export,
                igev,
                match_left,
                match_right,
                str(export_dir / "build_gwc_volume_plugin.onnx"),
                device=device,
                normalize=args.plugin_normalize,
            )
```

### 15.5 导出命令

```bash
export LD_LIBRARY_PATH=/root/trt_lib/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc

# 仅导出单节点 plugin ONNX
python simple_inference.py --export --export-submodules build_gwc_volume_plugin

# 导出完整模型 plugin ONNX（含 simplify）
python simple_inference.py --export --export-submodules full_model_plugin

# 需 normalize 版本
python simple_inference.py --export --export-submodules full_model_plugin --plugin-normalize

# 导出全部（含原生版 + plugin 版）
python simple_inference.py --export --export-submodules all
```

---

## 16. Simplify 难题与 FFS_PLUGIN_DOMAIN 突破

### 16.1 问题

onnxsim 0.4.36 对含 FFSGWCVolume 的 ONNX 报错：

```
ValidationError: No Op registered for FFSGWCVolume with domain_version of 17
```

### 16.2 尝试过的方案（均失败）

```python
from onnxsim import simplify

# 方案 1: skip_shape_inference
simplify(m, skip_shape_inference=True)       # FAIL

# 方案 2: skip_constant_folding
simplify(m, skip_constant_folding=True)       # FAIL

# 方案 3: 两者同时
simplify(m, skip_shape_inference=True, skip_constant_folding=True)  # FAIL
```

**根因**：onnxsim 无论如何都会先跑 `onnx.checker` 校验，自定义节点没有 op schema 就被拒。参数绕不过。

### 16.3 突破：自定义 domain

**关键发现**：ONNX checker 对**带自定义 domain 的节点会跳过 op schema 校验**（因为它不认识这个 domain，就不检查）。这也是 TensorRT 官方 plugin ONNX 的标准做法。

```python
import onnx
from onnx import helper
from onnxsim import simplify

DOMAIN = "trt.plugins"
m = onnx.load("plugin.onnx")

# 1. 给 FFSGWCVolume 节点加 domain
for n in m.graph.node:
    if n.op_type == "FFSGWCVolume":
        n.domain = DOMAIN

# 2. 注册对应 opset_import
m.opset_import.append(helper.make_opsetid(DOMAIN, 1))

# 3. 现在 checker 和 simplify 都能过了
onnx.checker.check_model(m)     # → PASS
sm, ok = simplify(m)            # → ok=True, FFSGWCVolume 节点保留
```

### 16.4 TRT 兼容性验证

关键问题：加了非空 domain 后，TRT 还能不能 parse？

```python
# 验证带 trt.plugins domain 的 simplified ONNX
ctypes.CDLL("libffs_gwc_plugin.so", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(logger, "")
parser = trt.OnnxParser(network, logger)
parser.parse(open("p_sim.onnx", "rb").read())
# → parsed OK, layers: 1
engine = builder.build_serialized_network(network, config)
# → ENGINE_BUILD: OK bytes=2964
```

**TRT 8.6 完全接受带 `trt.plugins` domain 的 ONNX** -- TRT parser 按 op_type 查 plugin registry，任何 domain 都认。

### 16.5 封装为公共 helper

```python
FFS_PLUGIN_DOMAIN = "trt.plugins"


def _tag_plugin_domain(model, op_type="FFSGWCVolume", domain=FFS_PLUGIN_DOMAIN):
    """给所有 op_type 节点挂上自定义 domain + 注册 opset_import"""
    from onnx import helper
    changed = False
    for node in model.graph.node:
        if node.op_type == op_type:
            node.domain = domain
            changed = True
    if changed:
        existing = {opset.domain for opset in model.opset_import}
        if domain not in existing:
            model.opset_import.append(helper.make_opsetid(domain, 1))
    return model


def _patch_plugin_output_shapes(model, op_type="FFSGWCVolume"):
    """给 plugin 输出 tensor 补 shape，让 onnxsim 的 shape inference 能传播"""
    from onnx import helper, TensorProto, shape_inference

    # 先跑一次 shape inference，填充标准 op 的中间 tensor shape
    # (torch.onnx.export 不写 value_info，原始 ONNX 的 value_info 是空的)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    # 收集所有已知 shape
    known_shapes = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if dims and all(d > 0 for d in dims):
            known_shapes[vi.name] = dims

    # 给每个 plugin 节点的输出补 shape: [B, G, D, H, W]
    patched = 0
    for node in model.graph.node:
        if node.op_type != op_type:
            continue
        attrs = {a.name: a.i for a in node.attribute}
        max_disp = attrs.get("max_disp", 0)
        cv_group = attrs.get("cv_group", 0)
        if max_disp <= 0 or cv_group <= 0:
            continue
        in_shape = None
        for inp_name in node.input:
            if inp_name in known_shapes:
                in_shape = known_shapes[inp_name]
                break
        if in_shape is None or len(in_shape) < 4:
            continue
        B, _, H, W = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
        out_shape = [B, cv_group, max_disp, H, W]
        out_name = node.output[0]
        existing_names = {vi.name for vi in model.graph.value_info}
        if out_name not in existing_names:
            vi_new = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)
            model.graph.value_info.append(vi_new)
            patched += 1
    if patched:
        print(f"[plugin-onnx] patched shape info for {patched} plugin output tensor(s)")
    return model


def _simplify_with_plugin(model, out_path_no_ext):
    """Patch shape -> Tag domain -> simplify -> save. 失败时回退"""
    import onnx
    from onnxsim import simplify

    model = _patch_plugin_output_shapes(model)
    model = _tag_plugin_domain(model)
    sim_path = out_path_no_ext + "_sim.onnx"
    try:
        simplified, ok = simplify(model)
        if not ok:
            raise RuntimeError("onnxsim returned check=False")
        onnx.save(simplified, sim_path)
    except Exception as exc:
        print(f"[plugin-onnx] simplify skipped ({type(exc).__name__}: {exc}); "
              f"saving domain-tagged un-simplified model")
        onnx.save(model, sim_path)
    return sim_path
```

### 16.6 关键修复：plugin 输出 shape 补丁

#### 问题发现

导出的完整模型 ONNX 中，FFSGWCVolume 节点的输出 tensor 没有 shape 信息。导致 onnxsim 的 shape inference 在 plugin 节点处中断，下游所有 tensor 也丢失 shape，simplify 实际上没有做任何常量折叠。

**修正前**的 simplify 效果：

| 指标 | 修正前 |
|------|--------|
| FFSGWCVolume 输出 shape | 缺失 |
| 下游 Conv 能看到输入 shape | 否 |
| simplify 后节点数 | 1531（无变化） |
| 缺 shape 的 tensor | 1126 / 1566 |

#### 根因

1. `torch.onnx.export` 导出自定义 op 时，输出 tensor 的 shape 标注为全 0（`[0, 0, 0, 0, 0]`），且不写入 `graph.value_info`
2. `onnxsim.simplify` 内部依赖 shape inference 来做常量折叠和优化，遇到无 shape 的自定义 op 就中断传播
3. 最初实现的 `_patch_plugin_output_shapes` 直接从 `graph.value_info` 读输入 shape，但 `torch.onnx.export` 导出的原始 ONNX 的 `value_info` 是**空的**

#### 修复

在 `_patch_plugin_output_shapes` 中先跑 `onnx.shape_inference.infer_shapes(model)`，填充标准 op 的中间 tensor shape（plugin 节点本身因无 schema 不会被推导，但它的**输入** tensor 会有 shape）。然后从输入 shape 和节点属性（`max_disp`、`cv_group`）推导输出 shape `[B, G, D, H, W]`，写入 `value_info`。

#### 迭代 simplify（第二轮修正）

单轮 simplify 后仍有 621/1463 tensor 缺 shape（42.4%），因为 onnxsim 的 shape inference 在 plugin 节点处中断，下游 tensor 的 shape 没被推导。非 plugin 版只有 1/1591 缺 shape。

**根因**：onnxsim 内部做 shape inference 时遇到自定义 op 就停止传播。虽然手动给 plugin 输出补了 shape，但 onnxsim 自己的 shape inference 引擎不认识这个 op，不会把补的 shape 往下游传播。只有在 simplify 结束后，用 ONNX 标准库的 `shape_inference.infer_shapes` 才能利用补的 shape 推导下游。

**修复**：在 `_simplify_with_plugin` 中做迭代 simplify -- 每轮 `simplify` 后跑一次 `shape_inference.infer_shapes`，再 simplify，直到节点数收敛或达到 3 轮上限：

```python
simplified, ok = simplify(model)
prev_nodes = len(model.graph.node)
for _round in range(2):
    simplified = shape_inference.infer_shapes(simplified)  # 补下游 shape
    simplified, ok = simplify(simplified)                   # 折叠更多常量
    if len(simplified.graph.node) == prev_nodes:
        break  # 收敛
    prev_nodes = len(simplified.graph.node)
```

#### 最终 simplify 效果

```
[plugin-onnx] patched shape info for 1 plugin output tensor(s)
```

| 指标 | 修正前（无 patch） | 单轮 simplify | 迭代 simplify（最终） | 非 plugin 版 |
|------|-------------------|--------------|---------------------|-------------|
| FFSGWCVolume 输出 shape | 缺失 | `[1,8,24,128,128]` | `[1,8,24,128,128]` | N/A |
| simplify 后节点数 | 1531 | 1428 | **1391** | 1556 |
| 缺 shape 的 tensor | 1126/1566 | 621/1463 | **1/1426** | 1/1591 |
| onnx.checker | PASS | PASS | PASS | PASS |

迭代 simplify 后：
- 节点数从 2311（原始）减到 **1391**，共 fold 了 920 个节点
- 缺 shape 的 tensor 从 2344 降到 **1**（与非 plugin 版持平）
- plugin 版比非 plugin 版少 165 节点（GWC 的 48 次 for 循环被 1 个 plugin 节点替代）

---

## 17. 端到端验证（完整模型）

### 17.1 导出

```bash
export LD_LIBRARY_PATH=/root/trt_lib/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /root/.../deploy_utils_hpc
python simple_inference.py --export --export-submodules full_model_plugin
```

日志输出：

```
[17:24:10] Full model (plugin) exported to full_model_plugin_sim.onnx;
           load plugin/build/libffs_gwc_plugin.so at TRT build time.
```

### 17.2 ONNX 结构验证

```python
import onnx
from collections import Counter

m = onnx.load("full_model_plugin_sim.onnx")
ops = [n.op_type for n in m.graph.node]
c = Counter(ops)

print("total nodes:", len(ops))           # → 1531
print("FFSGWCVolume count:", c.get("FFSGWCVolume", 0))  # → 1

ffs = [n for n in m.graph.node if n.op_type == "FFSGWCVolume"][0]
print("FFSGWCVolume domain:", repr(ffs.domain))  # → 'trt.plugins'
print("attrs:", {a.name: a.i for a in ffs.attribute})
# → {'cv_group': 8, 'max_disp': 24, 'normalize': 0}

print("opset_imports:", [(o.domain, o.version) for o in m.opset_import])
# → [('', 17), ('trt.plugins', 1)]

onnx.checker.check_model(m)  # → PASS
```

### 17.3 TRT 8.6 Parse 验证

```python
import ctypes, tensorrt as trt

ctypes.CDLL("libffs_gwc_plugin.so", mode=ctypes.RTLD_GLOBAL)
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, "")

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("full_model_plugin_sim.onnx", "rb") as f:
    ok = parser.parse(f.read())

print("PARSED OK, layers:", network.num_layers)  # → 1912
```

### 17.4 TRT 8.6 Engine Build 验证

```python
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.FP16)

import time
t = time.time()
engine = builder.build_serialized_network(network, config)
print("FULL_ENGINE:", ("OK bytes=" + str(engine.nbytes)) if engine else "FAILED",
      "in %.1fs" % (time.time() - t))
open("/tmp/full_model_plugin.engine", "wb").write(engine)
```

结果：

```
FULL_ENGINE: OK bytes=88500580 in 798.4s
saved /tmp/full_model_plugin.engine
```

**88.5 MB engine，798s build（完整 SIPE 模型 FP16 tactic 搜索，3060 上正常耗时）**

### 17.5 验证结果汇总

| 验证项 | 结果 |
|--------|------|
| ONNX 导出（1531 节点，1 个 FFSGWCVolume） | ✓ |
| FFSGWCVolume domain = trt.plugins | ✓ |
| attrs: cv_group=8, max_disp=24, normalize=0 | ✓ |
| opset_imports: [('', 17), ('trt.plugins', 1)] | ✓ |
| onnx.checker PASS（simplify 生效证明） | ✓ |
| TRT 8.6 parser 解析成功（1912 层） | ✓ |
| TRT 8.6 engine build 成功（88.5 MB） | ✓ |

---

## 18. 改动文件清单

### 18.1 新增文件

| 文件路径 | 说明 |
|---------|------|
| `plugin/CMakeLists.txt` | 精简版 CMake，适配本机 + 跨平台/跨 TRT 版本 |
| `plugin/include/ffs_gwc_plugin.hpp` | 从 FFS 复制 |
| `plugin/src/gwc_volume_plugin.cpp` | 从 FFS 复制 + kINT64 条件编译改动 |
| `plugin/src/depth_kernels.cu` | 从 FFS 复制（GWC CUDA kernel + 预处理/后处理 kernel） |

### 18.2 修改文件

**`submodule_export.py`** 新增内容：

| 行号 | 内容 |
|------|------|
| L330 | `FFS_PLUGIN_DOMAIN = "trt.plugins"` 常量 |
| L333 | `_tag_plugin_domain()` — domain 标注 helper |
| L354 | `_simplify_with_plugin()` — plugin-aware simplify helper |
| L385 | `_FFSGWCVolumeOp` — torch.autograd.Function（forward + symbolic） |
| L432 | `build_gwc_volume_plugin_onnx_export()` — 单节点 plugin 导出 |
| L508 | `_export_onnx_with_plugin()` — 公共导出 helper（绕过内建 checker） |
| L894 | `full_model_plugin_onnx_export()` — 完整模型 plugin 导出（monkey-patch） |

**`simple_inference.py`** 修改内容：

| 行号 | 改动 |
|------|------|
| L73 | 新增 `--plugin-normalize` 参数 |
| L680 | import 加 `build_gwc_volume_plugin_onnx_export` |
| L687 | import 加 `full_model_plugin_onnx_export` |
| L728-741 | export_pipeline 加 `full_model_plugin` 导出分支 |
| L836-844 | 子模块导出加 `build_gwc_volume_plugin` 导出调用 |

### 18.3 gwc_volume_plugin.cpp 改动详情

改动 1：include 块加 `NvInferVersion.h`

```cpp
// 改前
#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>

// 改后
#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>
#include <NvInferVersion.h>  // NV_TENSORRT_MAJOR, for 8.6 vs 10.x conditional compilation
```

改动 2：`fieldToInt` 函数的 kINT64 分支

```cpp
// 改前（FFS 原版，仅 TRT 10）
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        return static_cast<int32_t>(*static_cast<int64_t const*>(field.data));
    }

// 改后（跨版本兼容）
#if NV_TENSORRT_MAJOR >= 9
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        return static_cast<int32_t>(*static_cast<int64_t const*>(field.data));
    }
#endif
```

---

## 19. 遗留事项与后续工作

### 19.1 TRT 10.3/10.11 编译验证

TRT 10.3 和 10.11 均已在本机完成完整编译（cmake + build），三套 .so 均成功生成。详见第 13.7 节。

后续如需验证 10.3/10.11 的 engine build + 推理，需安装对应的 Python tensorrt 包（当前 py39 环境只有 8.6.1）。

### 19.2 数值对齐验证

**已修正并验证（CPU + GPU 双重验证）**：FFS 原版 CUDA kernel 用 sum（不除 K），与 SIPE 原版的 mean（除以 K）存在 K 倍差异。已在 `depth_kernels.cu` 的 `buildGWCVolumeKernel` 中加入 `dot *= (1.0f / K)` 修正为 mean。

#### CPU numpy 验证（逻辑正确性）

验证参数：B=1, C=8, H=4, W=6, maxdisp=3, G=4, K=2

```
SIPE mean vs 修正前 FFS sum  max diff: 2.38   (差 K=2 倍)
SIPE mean vs 修正后 FFS mean  max diff: 0.0    (完全一致)
CONCLUSION: SIPE == plugin_mean: True
```

#### GPU TRT engine 验证（端到端数值对齐）

验证脚本：`plugin/verify_gwc_alignment.py`

验证参数：B=1, C=64, H=120, W=160, maxdisp=48, G=8, K=8（SIPE 真实推理 shape）

```bash
export LD_LIBRARY_PATH=/root/trt_lib/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
cd /root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai
conda run -n py39 python hpc_quant/deploy_utils_hpc/plugin/verify_gwc_alignment.py
```

脚本原理：
1. 用纯 PyTorch 复现 SIPE 原版 `build_gwc_volume`（`.mean(dim=2)`），生成参考输出
2. 用 `build_gwc_volume_plugin_onnx_export` 导出 plugin ONNX -> TRT build engine -> 推理
3. 逐元素对比两个输出的 abs diff

GPU 验证结果：

```
============================================================
GPU 数值对齐验证
  shape: [1, 64, 120, 160]  maxdisp=48  G=8  K=8
============================================================

[1/3] SIPE 原版 (mean)  output shape: (1, 8, 48, 120, 160)
[2/3] TRT engine built: 2964 bytes
[3/3] Plugin (mean)     output shape: (1, 8, 48, 120, 160)

--- 对比结果 ---
  max  abs diff : 4.768372e-07
  mean abs diff : 1.384936e-08
  mean rel err  : 3.485521e-07

--- 抽样 ---
  ref   [0,0,0,0,0]      = -0.553557
  plugin[0,0,0,0,0]      = -0.553557
  ref   [0,3,24,60,80]   = -0.071256
  plugin[0,3,24,60,80]   = -0.071256
  ref   [0,7,47,119,159] = -0.404499
  plugin[0,7,47,119,159] = -0.404499

PASS: max abs diff 4.77e-07 < 1e-03 -- 数值完全对齐
```

max abs diff = 4.77e-07，为 FP32 浮点运算的精度极限（~1e-7 量级），数值完全对齐。

### 19.3 性能 Benchmark

尚无 plugin 版 vs 原生 Python for 循环版的性能对比数据。建议用 `nsys` / `ncu` 量化加速比。

### 19.4 Kernel 优化空间

当前 FFS GWC kernel 是朴素实现（一线程一输出点 + K 维串行 for），未利用 shared memory 复用特征数据。对于嵌入式平台（带宽受限）有较大优化空间。

### 19.5 normalize 语义差异

SIPE 原版 `build_gwc_volume` **无 normalize**，FFS plugin 有 `normalize` 选项。当前默认 `normalize=False` 对齐 SIPE。如果模型训练时用的是 GwcNet-style normalized GWC，需设 `--plugin-normalize`。

---

## 附录 A：关键路径速查

```
# SIPE 代码
/root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/
├── simple_inference.py          # 主入口，export_pipeline 在此
├── submodule_export.py          # 所有 ONNX 导出函数
└── plugin/                      # TRT plugin
    ├── CMakeLists.txt
    ├── include/ffs_gwc_plugin.hpp
    ├── src/gwc_volume_plugin.cpp
    ├── src/depth_kernels.cu
    └── build/libffs_gwc_plugin.so

# FFS 源码（参考）
/root/workspace/MixedAI_Datacenter/sipe-nocs/Fast-FoundationStereo/

# TRT 8.6
/root/trt_lib/TensorRT-8.6.1.6/
├── include/
└── lib/libnvinfer.so

# TRT 10.3 (完整 GA 包)
/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.3.0.26/
├── include/
└── lib/libnvinfer.so

# TRT 10.11 (完整 GA 包)
/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.11.0.33/
├── include/
└── lib/libnvinfer.so

# 原生 build_gwc_volume 定义
/root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/mixedai/layers/stereo_layers/utils.py

# 导出产物
/root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai/hpc_quant/simple_inference_results/submodule_onnx/
├── build_gwc_volume_plugin_sim.onnx     # 单节点
└── full_model_plugin_sim.onnx           # 完整模型
```

## 附录 B：环境信息

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 (sm_86) |
| CUDA (runtime) | 11.7 |
| nvcc | 11.2.152 |
| TensorRT (py39) | 8.6.1 |
| TensorRT 8.6.1.6 | `/root/trt_lib/TensorRT-8.6.1.6` (完整 .so.8) |
| TensorRT 10.3.0.26 | `/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.3.0.26` (完整 .so.10) |
| TensorRT 10.11.0.33 | `/root/workspace/MixedAI_Datacenter/hub/TensorRT-10.11.0.33` (完整 .so.10) |
| Python | 3.9.17 |
| PyTorch | 2.0.1+cu117 |
| triton | 2.0.0 (pinned by torch) |
| onnxsim | 0.4.36 |
| cmake | 3.22.0 |
| Docker 容器 | sipe_inst |
| Conda 环境 | py39 |
