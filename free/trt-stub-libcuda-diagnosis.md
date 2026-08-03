# 容器内 stub libcuda 导致 TRT 显存检测失败：诊断与修复

> 技术总结 | 2026-08-01 | 环境：AGX Orin 64GB（192.168.11.39），容器 smr_asr_nsys，TensorRT 10.3.0
> 背景：排查 IGEV 模型 TRT 构建失败时发现（完整 IGEV 排查见 `igev-trt10.3-dynamic-build-fix.md`）

## 1. 问题是什么

**现象**：python 调用 TensorRT 构建 engine 时，TRT 报告的可用显存只有 **257MB**（实际 61.37GB），导致所有需要 >257MB workspace 的 kernel 实现被静默拒绝，最终报 `Could not find any implementation for node ...`（Error 10）。

**本质**：容器默认 `LD_LIBRARY_PATH` 把 **stub 版 libcuda**（编译占位库）放在首位，python 加载 tensorrt 时解析到 stub 驱动 → CUDA driver API 全部失败 → TRT 无法确定显存 → **降级用保守默认值 257MB** 继续构建。

**关键点：这不是"获取到错误数值"，而是"检测到 stub → 无法确定显存 → fallback 保守值"。** TRT 日志有明确自述：

```
[TRT] [W] Unable to determine GPU memory usage: CUDA driver is a stub library
```

**影响范围**：
| 调用方式 | 是否受影响 | 原因 |
|---|---|---|
| python 加载 tensorrt | **是** | 依赖 LD_LIBRARY_PATH 解析 libcuda |
| trtexec（/usr/src/tensorrt/bin）| 否 | 编译时 RPATH 写死真实库路径 |

## 2. 怎么发现的（完整排查链）

### 2.1 现象起点

python 构建 83 秒失败，warning 显示：

```
Tactic Device request: 360MB  Available: 257MB. Device memory is insufficient to use tactic.
```

### 2.2 物理显存与 TRT 报告矛盾

```python
# CUDA runtime API（libcudart）空闲时实测 → free=47.08GB total=61.37GB
# 构建进行中（45 秒）再测 → free=44.24GB（几乎没掉！）
# cudaMalloc 连续分配 40 个 1GB chunk → 全部成功（非 carveout 限制）
```

**物理显存充足，但 TRT 报告 257MB → 排除显存真不足，怀疑驱动层。**

### 2.3 LD_LIBRARY_PATH 分析

```bash
echo $LD_LIBRARY_PATH
# /usr/local/cuda/lib64/stubs::/usr/local/cuda/compat:/usr/local/cuda/lib64:
#           ^^^^^^^^^^^^^^^^^^^^ stub 目录在首位！
```

系统里 libcuda 有多个版本：

```
/usr/local/cuda-12.6/targets/aarch64-linux/lib/stubs/libcuda.so   ← stub（编译用）
/usr/lib/aarch64-linux-gnu/stubs/libcuda.so                        ← stub
/usr/lib/aarch64-linux-gnu/libcuda.so                              ← 链接
/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1                     ← 真实驱动 ★
```

### 2.4 铁证一：LD_DEBUG 追踪库解析

```bash
# 默认环境：TRT 加载 stub
LD_DEBUG=libs python3 -c "import tensorrt" 2>&1 | grep -i libcuda
#   find library=libcuda.so [0]; searching
#   trying file=/usr/local/cuda/lib64/stubs/libcuda.so
#   calling init: /usr/local/cuda/lib64/stubs/libcuda.so     ← stub！

# 干净环境：TRT 加载真实驱动
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia LD_DEBUG=libs \
    python3 -c "import tensorrt" 2>&1 | grep -i libcuda
#   calling init: /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so  ← 真实！
```

### 2.5 铁证二：cuInit 错误码 34

```python
import ctypes
cu = ctypes.CDLL('libcuda.so')
print(cu.cuInit(0))
# 默认环境: 34 = CUDA_ERROR_STUB_LIBRARY（NVIDIA 官方给 stub 库的错误码，字面"驱动是 stub"）
# 干净环境: 0（成功）
```

### 2.6 铁证三：TRT 自己的警告

stub 环境下最小构建时 TRT 打印：

```
[TRT] [W] Unable to determine GPU memory usage: CUDA driver is a stub library
```

TRT **明确检测到** stub 驱动，然后显存检测失效 → fallback 257MB。

### 2.7 关键对照：trtexec 为什么不受影响

```bash
# 两种环境下 trtexec 打印的 Device Global Memory 都是正确的
trtexec --onnx=/nonexistent.onnx 2>&1 | grep "Device Global Memory"
#   Device Global Memory: 62841 MiB   （两种环境一致，RPATH 生效）
```

> 注意：trtexec 免疫是**这台 AGX 的特性**（deb 安装的 RPATH 写死了 /usr/src/tensorrt/lib 路径），不是普适规律。反例：sipe_inst 容器的 trtexec（TRT 8.6）连 `libnvinfer_plugin.so.8` 都要 export LD_LIBRARY_PATH 才能 dlopen。

## 3. 怎么解决的

```bash
# python 调用 TRT 前设置（或写入 test.sh / 启动脚本）
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64
```

要点：
- **真实驱动的路径**：`/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`（用 `find / -name "libcuda.so*"` 定位）
- **原则**：LD_LIBRARY_PATH 里**不要包含 stubs 目录**（或确保真实路径在 stubs 之前）
- 对 trtexec（这台 AGX）：不需要，但设了无害
- **不要用设置 workspace（memPoolSize）来"修复"**——workspace 是 tactic 过滤上限，改变不了显存检测值（实测 WS_MB=1024 后 Available 仍是 259MB）

## 4. 怎么验证的

| 验证项 | 方法 | 结果 |
|---|---|---|
| 行为验证 | 修复前后构建推进 | 83 秒（死在早期节点）→ 32 分钟（推进到深层节点）|
| 加载路径 | LD_DEBUG=libs | stub → 真实驱动 |
| driver API | cuInit | 34 → 0 |
| 显存查询 | cuDeviceTotalMem / cudaMemGetInfo | 0/错误 → 61.37GB |
| TRT 自述 | 最小构建捕获日志 | "Unable to determine GPU memory usage" → 消失 |
| 最小构建 | 64x64 add 网络 | 崩溃（Aborted）→ SUCCESS |

## 5. 验证脚本：diag_trt_stub.py

完整脚本（已部署在 `/home/workspace/armin_profile/ffs/igev_dynamic/diag_trt_stub.py`）：

```python
#!/usr/bin/env python3
"""
诊断：python 调用 TRT 时是否受 stub libcuda 影响（AGX Orin / TRT 10.3）

用法（对比两次输出即可看到 stub 影响）：
  # 1) 默认环境（容器原样，预期显示 stub 影响）
  python3 diag_trt_stub.py

  # 2) 修复环境（预期正常）
  LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64 \
      python3 diag_trt_stub.py

判断标准：
  [1] cuInit = 34  → CUDA_ERROR_STUB_LIBRARY，驱动是假的
  [4] TRT 日志出现 "Unable to determine GPU memory usage: CUDA driver is a stub library"
      → TRT 显存检测失效，构建时 Available 会 fallback 到 ~257MB
"""
import ctypes
import os


def section(title):
    print('\n' + '=' * 62)
    print(title)
    print('=' * 62)


def loaded_libcuda():
    """当前进程实际加载的 libcuda 路径（/proc/self/maps 铁证）"""
    paths = []
    for line in open('/proc/self/maps'):
        if 'libcuda' in line:
            p = line.split()[-1]
            if p and p not in paths:
                paths.append(p)
    return paths


def driver_cuinit():
    """CUDA driver API cuInit 错误码（34 = CUDA_ERROR_STUB_LIBRARY）"""
    try:
        cu = ctypes.CDLL('libcuda.so')
        r = cu.cuInit(0)
        return r
    except OSError as e:
        return 'load-fail: %s' % e


def runtime_meminfo():
    """CUDA runtime API 显存（libcudart；对照用）"""
    cuda = ctypes.CDLL('libcudart.so.12')
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    r = cuda.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return r, free.value / 2**30, total.value / 2**30


def trt_probe():
    """最小网络构建 + 自定义 logger 捕获 TRT 显存检测日志"""
    import tensorrt as trt
    capture = []

    class Cap(trt.ILogger):
        def log(self, severity, msg):
            capture.append(msg)
            print('  [TRT:%s] %s' % (str(severity).split('.')[-1], msg))

    builder = trt.Builder(Cap())
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    a = network.add_input('a', trt.float32, (1, 3, 64, 64))
    b = network.add_input('b', trt.float32, (1, 3, 64, 64))
    e = network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
    network.mark_output(e.get_output(0))
    config = builder.create_builder_config()
    try:
        ok = builder.build_serialized_network(network, config) is not None
        print('  最小构建(64x64 add):', 'SUCCESS' if ok else 'FAILED')
    except Exception as ex:
        print('  最小构建(64x64 add): EXCEPTION %s' % str(ex)[:80])

    print('  TRT 进程视角的 libcuda 加载:')
    for p in loaded_libcuda():
        print('    |', p)

    print('  TRT 显存相关日志:')
    keys = [m for m in capture
            if any(k in m for k in ('memory', 'Memory', 'stub', 'Available', 'Global'))]
    if not keys:
        print('    (无显存相关日志)')
    for m in keys[:10]:
        print('    |', m)


def main():
    print('LD_LIBRARY_PATH =', os.environ.get('LD_LIBRARY_PATH', '(empty)'))

    section('[1] CUDA driver API cuInit 错误码（34 = stub 驱动）')
    r = driver_cuinit()
    print('  cuInit =', r)
    if r == 34:
        print('  → 加载了 stub 驱动！driver API 显存查询全部不可信')
    elif r == 0:
        print('  → 驱动正常（真实 libcuda）')
    else:
        print('  → 其他错误码（查 CUDA 文档）')

    section('[2] 当前进程实际加载的 libcuda（/proc/self/maps）')
    for p in loaded_libcuda():
        print('  ', p)
    if not loaded_libcuda():
        print('   (none)')

    section('[3] CUDA runtime API 显存（libcudart，对照用）')
    r, free_gb, total_gb = runtime_meminfo()
    print('  cudaMemGetInfo ret=%d  free=%.2fGB  total=%.2fGB' % (r, free_gb, total_gb))
    if r == 34:
        print('  → runtime API 也被 stub 污染（stubs 目录拦截了 libcudart 的依赖）')

    section('[4] TRT 最小构建时的显存检测日志')
    trt_probe()

    section('[5] 判定')
    stub = any('stub' in p for p in loaded_libcuda())
    if stub:
        print('  ✗ 受 stub 影响：python 加载了 stub libcuda，TRT 显存检测不可信')
        print('    （构建时 Available 会 fallback 到 ~257MB，大 kernel 全被拒）')
        print('  修复：export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64')
        print('  然后重跑本脚本对比')
    else:
        print('  ✓ 驱动正常，TRT 显存检测可信')


if __name__ == '__main__':
    main()
```

**实测输出对比**（AGX 上验证过）：

| 检测项 | 默认环境（stub）| 修复环境 |
|---|---|---|
| cuInit 错误码 | 34（stub）| 0（正常）|
| 实际加载的 libcuda | `.../stubs/libcuda.so` | `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1.1` |
| cudaMemGetInfo | ret=34, free=0（runtime 也被污染）| ret=0, free=45.37GB |
| TRT 日志 | `Unable to determine GPU memory usage: CUDA driver is a stub library` | 无此警告 |
| TRT 最小构建 | 崩溃（terminate/Aborted）| SUCCESS |
| 判定 | ✗ | ✓ |

## 6. 关键命令速查

```bash
# ① 定位真实驱动
find / -name "libcuda.so*" -not -path "/proc/*" 2>/dev/null

# ② 确认 TRT 加载了哪个 libcuda（铁证）
LD_DEBUG=libs python3 -c "import tensorrt" 2>&1 | grep -i libcuda

# ③ 看进程实际加载（运行时）
grep libcuda /proc/self/maps

# ④ 修复
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/src/tensorrt/lib:/usr/local/cuda/lib64
```

## 7. 经验教训

1. **LD_LIBRARY_PATH 里的 stubs 是隐形杀手**：报错表现为"显存不足/无实现"，实际是驱动加载错误。遇到 `Available` 异常小的值先查这个。
2. **TRT 对显存检测失败是降级处理**（fallback 保守值 257MB 继续构建），不是报错退出——所以问题会藏在"找不到实现"这类后续错误里。
3. **判断环境是否受 stub 影响的标准**：`trtexec --onnx=不存在.onnx | grep "Device Global Memory"`——正确显示真实显存则 trtexec 免疫；python 环境用 `cuInit` 错误码判断。
4. **stub 环境连 runtime API（libcudart）也可能被污染**（cudaMemGetInfo 返回 34），所以"手动测显存正常"不能证明 TRT 正常。
5. workspace（memPoolSize）**不能**修复 stub 问题——两者是独立机制。
