# FFS 模型 TRT 8.6 动态引擎构建 OOM 分析：workspace / memPoolSize / 激活内存机制

> 讨论总结 | 2026-08-01 | 环境：sipe_inst 容器，TensorRT 8.6.1.6，RTX 3060 12GB（WSL）

## 1. 背景

- **模型**：Fast-FoundationStereo（FFS）单目/双目立体匹配，ViT-Large backbone，`max_disp=192`，`corr_levels=2`，`vit_size=vitl`
- **任务**：ONNX → TRT 8.6 FP16 动态 shape engine
- **shape 配置**：min=1x3x32x32，opt=1x3x544x960，max=1x3x1512x2048
- **硬件**：RTX 3060 12GB，Windows 桌面占用 ~3.6GB，实际可用 ~8.5GB
- **命令**：`trtexec --onnx=... --fp16 --minShapes/--optShapes/--maxShapes ...`

## 2. 现象

| 阶段 | 报错 | 原因 |
|---|---|---|
| 原始 test.sh（无 memPoolSize 限制） | `virtualMemoryBuffer::resizePhysical OutOfMemory`，大量 tactic 因 `Device memory is insufficient`（请求 4~9.3GB）被跳过 | 构建 profiling 阶段物理显存分配失败 |
| 加 `--memPoolSize=workspace:2048MiB` 后 | `Could not find any implementation for node PWN(/Div_2, PWN(/Div_1, /Mul_1)) due to insufficient workspace`（Tactic Device request **22560MB**） | workspace 配额过滤掉所有实现 |
| 宿主侧 | trtexec 构建吃 1.5~4GB 宿主内存，15GB 宿主 + 4GB swap 全满时被 Linux OOM killer 杀掉（oom_kill 累计 5 次） | 宿主内存不足，与显存无关的次生问题 |

## 3. 根因分析

### 3.1 定位 PWN 节点

`PWN(/Div_2, PWN(/Div_1, /Mul_1))` **不是** pos_embd 的位置编码（那是 3D 小张量），而是**根路径 `/Div_1`、`/Div_2`、`/Mul_1`** 构成的 6D 归一化模块（cost volume 分块归一化）：

```
 /Div_1:  a ÷ b → c    张量 [1, 8, 28, 48, d1, d2]
 /Div_2:  d ÷ e → f    张量 [1, 8, 28, 48, d1, d2]
 /Mul_1:  c × f → g    张量 [1, 8, 28, 48, d1, d2]
```

- 维度规律（ORT 实测）：`d1 = d2 = 输入分辨率 / 4`

| 输入 | 6D 张量 | 元素数 |
|---|---|---|
| 32×32 | [1,8,28,48,8,8] | 688K |
| 64×64 | [1,8,28,48,16,16] | 2.75M |
| 96×96 | [1,8,28,48,24,24] | 6.19M |
| **1512×2048** | **[1,8,28,48,378,512]** | **2.08G（=4.2GB fp16）** |

### 3.2 TRT 8.6 PWN workspace 放大效应

- PWN 融合把 Div_1+Div_2+Mul_1 合成一个 kernel，对 **6D + dynamic shape 不做流式处理**，而是**全物化中间 buffer**
- 同一时刻需同时占用约 5.4 份张量大小：**4.2GB × 5.4 ≈ 22.5GB**
- 12GB 卡（TRT 视角 Available=12287MB）物理装不下 → 该节点**所有**实现都被跳过 → 构建失败

**物化 vs 流式**（类比：搬仓库 vs 流水线）：

```
物化式（TRT 8.6 PWN）           流式（理想实现）
────────────────────           ────────────────────
所有箱子一次搬进仓库             传送带，一次处理 1 箱
空间 = 箱子数 × 每箱大小        空间 ≈ 1~2 箱大小
```

### 3.3 实测边界（python TRT API 简化网络，FP16）

| max 档位 | 6D 张量 (fp16) | workspace 需求 | 构建结果 |
|---|---|---|---|
| 544×960 | 0.5GB | ~2.7GB | 成功 45s |
| 720×1024 | 0.9GB | ~4.9GB | 成功 35s |
| 1024×1024 | 1.3GB | ~7.0GB | 成功 138s |
| 1080×1920 | 2.1GB | ~11.4GB | 成功 52s |
| **1512×2048** | **4.2GB** | **22.5GB** | **失败**（> 12GB）|

排除实验（均无效，确认是纯规模问题）：

| 尝试 | 结果 |
|---|---|
| 打断融合：恒等 Reshape | 被 TRT 消除，融合照旧 |
| 打断融合：全范围 Slice | 被 TRT 消除 |
| 打断融合：1x1 恒等 Conv | 被 TRT 消除（识别恒等卷积）|
| 打断融合：3x3 恒等 Conv | 解析后保留，但 builder 仍重新融合 PWN |
| 拆成独立 op（python 构造） | TRT 重新融合成 PWN，同样失败 |
| 6D reshape 成 1D | 同样失败（resizePhysical OOM）|

## 4. 关键概念：workspace / memPoolSize / 激活内存

### 4.1 workspace = 算子的临时工位

- **workspace（22.5GB）**：某个 kernel 执行时需要的临时显存，由 TRT 实现决定
- **4.2GB 是"一份数据"的大小**，22.5GB = 5.4 份同时占用（TRT 8.6 的 PWN 实现低效）

### 4.2 memPoolSize = 候选过滤器，不是预分配承诺

```
构建期:  memPoolSize=4GB ──过滤器──> 候选实现 = {workspace 需求 ≤ 4GB 的}
             │
             ↓ 选最快的
         选中实现实际需要 w* GB（例如 1.2GB）
             │
             ↓ 序列化进 engine
运行期:  engine.workspaceSize = max(所有层选中的 w*)   ← 这才是固定开销
```

- 设大：候选池大，可能选到"大 workspace 但更快"的实现；风险是 profiling 阶段物理分配 OOM、engine 运行时固定开销大
- 设小：候选池小，只剩保守实现（性能可能降 10~30%）
- **memPoolSize ≠ 运行时预分配**：运行时只分配选中实现的实际需求峰值

**两种报错的演变机制**（同一节点 22.5GB 需求，配额大小决定"死法"）：

```
┌─ memPoolSize 不设（默认=无限制）或设 >22.5GB
│    22.5GB ≤ 配额 → PWN 的实现入选 → 构建继续
│    → profiling 阶段实际分配显存 → 12GB 卡放不下
│    → 报错: resizePhysical OutOfMemory          ← 原始 test.sh 日志
│
├─ memPoolSize 设 2048MB（< 22.5GB）
│    22.5GB > 配额 → 所有实现被过滤
│    → 报错: Could not find any implementation
│             ...due to insufficient workspace    ← 加限制后的日志
│
└─ 结论：两种报错根因相同 —— 该节点需求 22.5GB 超卡容量，
         配额只是决定失败发生在"选实现"还是"分配显存"阶段
```

### 4.3 运行时显存构成

```
运行时总显存 ≈ 权重 + 输入输出 + 激活 + workspace + 引擎/context 开销
```

- **激活内存**（中间张量本体）由 **max shape** 决定，**不受 memPoolSize 管**
- memPoolSize 只约束 kernel 临时工位（workspace）一项

## 5. 结论与解决方案

**核心结论：max=1512×2048 下该 6D 中间张量 4.2GB，TRT 8.6 PWN 按 5.4 倍规划出 22.5GB，12GB 卡（可用 8.5GB）结构性装不下——构建和运行都会失败，非参数可解。**

| 方案 | 说明 | 适用 |
|---|---|---|
| ① max 降到 ≤1080×1920 | 实测可构建 | 业务输入 ≤1080p 时 |
| ② 换 16GB+ 卡（4090/3090 等 24GB） | 22.5GB 需求可容纳 | 必须支持 1512×2048 |
| ③ INT8 量化 | 激活减半，可能可行 | 需要 calibration，大工程 |

**正确决策链**（避免重复踩坑）：

```
1. 定 max shape      → 决定数据占位（激活内存），必须 ≤ 卡物理容量
2. 定 memPoolSize    → kernel 工位预算，12GB 卡建议 3~4GB
                      （可用显存 - 权重/激活/引擎开销 - 余量）
3. 构建后验证        → 看 --dumpProfile 实际峰值，留 10~20% 余量
```

## 6. 附：排障工具与命令

```bash
# 1. 中间张量真实 shape（ORT，把中间 tensor 挂到 graph.output）
python3 -c "
import onnx
m = onnx.load('model.onnx')
for t in ['/Div_1_output_0']:
    vi = onnx.helper.ValueInfoProto(); vi.name = t; m.graph.output.append(vi)
onnx.save(m, 'probe.onnx')"  # 然后 ORT run 取 shape

# 2. python TRT API 快速构建验证（避免 trtexec 全量构建）
#    builder.build_serialized_network(network, config) + set_memory_pool_limit

# 3. 符号 shape 推理
#    onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference

# 4. trtexec 必须 export 库路径（trtexec 是运行时 dlopen 加载 nvinfer）
export LD_LIBRARY_PATH=/root/trt_lib/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
```

## 7. 环境备忘

- TRT 8.6.1.6 路径：`/root/trt_lib/TensorRT-8.6.1.6`（sipe_inst 容器）
- py39 有 tensorrt python 8.6.1（import 前需 export LD_LIBRARY_PATH）
- 构建吃 1.5~4GB 宿主内存：15GB 宿主 + 4GB swap 满时会被 OOM killer 杀，构建前先腾内存
