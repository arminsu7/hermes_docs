# IGEV TRT 构建在 Orin NX 上的显存限制：dynamic 失败与 opt shape 瓶颈

> 技术总结 | 2026-08-02 | 环境：Orin NX 16GB（192.168.11.47），容器 deepac_batch_test，TensorRT 10.3.0，MAXN_SUPER
> 前置：IGEV convc1 修复见 `igev-trt10.3-dynamic-build-fix.md`；本文验证「修复后的模型在 Orin NX 上能否转换」

## 1. 背景

- **硬件**：Orin NX 16GB 统一内存（`free` 显示 15GB），MAXN_SUPER 电源模式
- **软件**：TRT 10.3.0（/usr/src/tensorrt），CUDA 12.6，driver 540.4.0
- **模型**：`model_fp32_optimized_simplified_fixed.onnx`（AGX 上修复 convc1 后的版本，用户传到 NX）
- **目标**：在 NX 上跑 trtexec 转换（dynamic），并监控转换过程最大显存占用
- **原始 test.sh**：min=32×32，opt=544×960，max=960×1024（fp16）

## 2. 现象

dynamic 构建 2 分钟失败：

```
[E] Could not find any implementation for node /cnet/layer1/layer1.0/conv2/Conv + .../relu_1/Relu + PWN(.../Add + .../relu_2/Relu)
```

日志关键 warning：

```
[W] Tactic Device request: 720MB Available: 650MB. Device memory is insufficient to use tactic.
```

**Available 只有 650MB**（Device Global Memory 检测正确 = 15655 MiB，非 stub 问题）。

## 3. 排查过程（排除项）

| 假设 | 验证 | 结论 |
|---|---|---|
| stub libcuda 显存误检 | trtexec 打印 Device Global Memory = 15655 MiB（正确）；cudaMemGetInfo 实测 free 9.1GB | ❌ 不是 stub（NX trtexec 检测正常）|
| workspace 限制 | --memPoolSize=workspace:1024MiB → Available 仅 614→646MB | ❌ 几乎无效（瓶颈不是 workspace）|
| min 区间宽度 | min=32×32 和 min=512×512 同样 Available ~650MB | ❌ min 无关 |
| 降 max | max=960→640×960 依然失败（Available 掉到 <1MB）| ❌ 降 max 不解决（见下）|

**关键观察**：Available 不是"一次性规划值"，而是**随构建推进持续下降**：

```
Available = 15655MB - 已分配的累积 buffer
IGEV 38997 层，构建逐层分配 → Available 一路走低
max=640×960 实测: 284MB → 273MB → 104MB（构建中）
→ 中后期节点（cnet/conv2、deconv32_16 的 ForeignNode 请求 0.9MB 都被拒）
```

## 4. 实验矩阵（NX 上实测）

| 配置 | 结果 | Activation Memory |
|---|---|---|
| dynamic min=32/512, opt=544×960, max=960×1024 | ❌ 失败（Available 650MB）| — |
| dynamic min=512, opt=544×960, max=640×960 | ❌ 失败（Available <1MB @ ForeignNode）| — |
| dynamic min=256, opt=544×960, max=640×960（窄区间）| ❌ 失败（Available 104MB）| — |
| **单 profile 544×960**（min=opt=max）| ✅ **PASSED** | 839 MB |
| **dynamic min=256, opt=512×512, max=512×512（方案 2）** | ✅ **PASSED** | **323 MB** |

## 5. 根因分析

**瓶颈是 opt shape（tactic 评估的 buffer 主体），不是 max。**

```
NX 可分配显存 = ~10.6GB（cudaMemGetInfo 实测，16GB 统一内存被系统占用部分）

IGEV 多 profile 构建：
  opt=544×960 时，构建期累积 buffer ≈ 10GB+（多 profile 的 buffer 不共享/重复分配）
  → 构建中后期 Available 耗尽（<1MB）→ 大 tactic 全被拒 → 无实现

单 profile / opt 降到 512×512：
  buffer 主体小（activation 323~839MB）→ Available 充足 → 构建成功
```

**为什么降 max 没用**：buffer 主体由 **opt** 决定（tactic 按 opt shape 评估），max 只影响上限预留。opt=544×960 不变时降 max，只是把失败点往后推（650MB→104MB），不解决。

**对比 AGX（64GB）**：同参数 dynamic 构建成功（activation 1.42GB）——61GB 可分配对 10GB+ 累积 buffer 容错大，NX 16GB 直接吃满。

## 6. 结论与建议

1. **NX 上 dynamic 构建的可行配置：opt/max ≤ 512×512**（已验证 min=256, opt=max=512×512 PASSED）
2. **单 profile（固定分辨率）完全可行**：544×960 PASSED（activation 839MB），512×512 PASSED（323MB）
3. **业务如果必须 544×960 且要 dynamic**：NX 不可行 → 用 AGX 构建 dynamic engine 再部署到 NX 运行（NX/AGX 同 Orin 架构 + 同 TRT 10.3，engine 理论兼容，需实测）；或换更大显存设备
4. 构建期 CPU 内存也是约束：单 profile 544×960 CPU 峰值 10.9GB（NX 15GB 剩 ~4GB 余量），dynamic 若推进到后期可能更高（AGX 上 dynamic CPU 峰值 43GB）

## 7. 可行配置的命令（方案 2，已验证）

```bash
docker exec -it deepac_batch_test bash
cd /root/workspace/sumingrui/ffs/dynamic_onnx/igev_test_1

# LD_LIBRARY_PATH 修正（trtexec 在 NX 上也建议加，与 AGX 一致）
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64

/usr/src/tensorrt/bin/trtexec \
--onnx=model_fp32_optimized_simplified_fixed.onnx \
--saveEngine=test_512dyn.engine \
--fp16 \
--minShapes=image1:1x3x256x256,image2:1x3x256x256 \
--optShapes=image1:1x3x512x512,image2:1x3x512x512 \
--maxShapes=image1:1x3x512x512,image2:1x3x512x512 \
> /tmp/dyn512.log 2>&1
```

**日志**：容器内 `/tmp/dyn512.log`（PASSED 确认）；产物 `test_512dyn.engine`（55.6MB）

**成功构建的内存数据**（方案 2）：
```
Total Activation Memory = 338892288 B = 323 MB
Total Device Persistent  = 9283584 B  = 9.3 MB
GPU allocator 峰值       = 309 MiB（构建期）
CPU 峰值（构建+序列化）  = 11017 MiB（10.8GB）
```

## 8. 显存监控方法（NX 上 nvidia-smi 显存 N/A 时的替代）

```python
# 容器内每 3 秒采样 cudaMemGetInfo（注意先 export LD_LIBRARY_PATH）
import ctypes, time
c = ctypes.CDLL('libcudart.so.12')
while True:
    f = ctypes.c_size_t(); t = ctypes.c_size_t()
    r = c.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t))
    print(time.strftime('%H:%M:%S'), f.value // 2**20, 'MB free')
    time.sleep(3)
```
