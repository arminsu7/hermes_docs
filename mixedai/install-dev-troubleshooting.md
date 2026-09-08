# MixedAI `install_dev.sh` 安装失败排查与修复

> 环境：`gwc_cuda128` 容器（dexsdk:ubuntu22.04-cuda12.8.0-v0.2）
> Python：`py310`（torch 2.7.0+cu128, CUDA 12.8, TRT 10.11.0.33）
> 任务仓库：`/root/repos/dex_lib/mixedai`
> 参考 plugin 目录：`/root/repos/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/plugin`
> 日期：2026-08-17

---

## 一、结论（TL;DR）

`scripts/install_dev.sh` 原本会失败，根因有三层，从外到内：

1. **pip 依赖回溯卡死** —— `pymeshlab==2023.12` 与 `dexsim-engine 0.3.11` 要求的 `pymeshlab>=2023.12.post3` 冲突，pip 无限回溯反复下载大包，看起来「卡死/超时」。
2. **`setup.py develop` 崩溃** —— `setup.py` 顶层 `import torch`，而 `pip install -e . --use-pep517` 走 PEP517 隔离构建环境（无 torch），报 `ModuleNotFoundError: No module named 'torch'`，最终 `EXIT=1`。
3. **editable install 不被支持** —— requirements 强制 `setuptools==59.8.0`（2021 年老版），不支持 PEP 660 的 `build_editable` hook，`pip install -e .` 必挂。

**修复方案**：把 `scripts/install_dev.sh` 最后一步从 `python setup.py develop` 改为 `pip install -e . --no-build-isolation`（editable 安装 + 关掉 PEP517 隔离构建）；把两处 `requirements*.txt` 里的 `pymeshlab` 约束放宽（修 dexsim-engine 冲突）；把 `setuptools==59.8.0` 放宽为 `setuptools>=65.5.0`（支持 PEP 660 editable）。修复后 `pip install -e . --no-build-isolation` 编译 CUDA extension 成功（EXIT=0），`mixedai._C` 的 13 个 CUDA ops 全部可用，且为 editable（改纯 Python 即改即生效）。

---

## 二、脚本原始内容

```bash
# 开发者模式安装脚本
pip config set global.index-url http://192.168.3.43:8080/simple/
pip config set global.trusted-host 192.168.3.43
pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple
# 安装私有python包
pip install --no-cache-dir -r requirements-dev.txt --index-url http://192.168.3.43:8080/simple/ --trusted-host 192.168.3.43

# 最后一步：开发者模式安装
python setup.py develop
```

---

## 三、根因分析（现象 → 原因 → 修复）

### 问题 1：pip 依赖解析无限回溯（慢 / 卡死的元凶）

**现象**
- 日志反复打印：`INFO: pip is looking at multiple versions of dexsim-engine to determine which version is compatible with other requirements. This could take a while.`
- 同一批大包（numpy / open3d / opencv / warp-lang / vtk / pymeshlab…）被**反复重新下载**，每次轮回都从 0 开始，永远走不到「安装」阶段。
- 最终抛出：
  ```
  ERROR: Cannot install -r requirements-dev.txt (line 50) and pymeshlab==2023.12
  The conflict is caused by:
      The user requested pymeshlab==2023.12
      dexsim-engine 0.3.11 depends on pymeshlab>=2023.12.post3
  Additionally, some packages in these conflicts have no matching distributions available...
  ERROR: ResolutionImpossible
  ```

**原因**
- `requirements.txt` 和 `requirements-dev.txt` 第 43 行都写死 `pymeshlab==2023.12`。
- `dexsim-engine`（私有包，第 50 行）声明依赖 `pymeshlab>=2023.12.post3`。
- 两个约束互相矛盾（`==2023.12` 与 `>=2023.12.post3` 不相交），pip 解析器无解，只能不断回溯尝试不同版本组合，造成无限循环重下大包。
- 内网 pip 源（192.168.3.43）带宽约 1.5~2 MB/s，这些大 wheel（open3d 553MB、warp-lang 163MB、vtk 112MB、pymeshlab 105MB）本身就极慢，叠加回溯循环，看起来就像卡死/超时。

**为什么内网源慢**
- `open3d-0.19.0` 553.4 MB、`warp_lang-1.16.0` 163.1 MB、`vtk-9.5.2` 112.2 MB、`pymeshlab` 105.9 MB、`opencv_contrib` 73 MB… 全是几十上百 MB 的 wheel，走内网源下载耗时以分钟计。
- 加上 `pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple` 改变了候选源，pip 会重新评估各版本，进一步放大下载量。

**修复**
- 将两处 `pymeshlab==2023.12` 改为 `pymeshlab>=2023.12.post3`，满足 dexsim-engine 约束，pip 一次解析成功（最终解析到 `pymeshlab-2025.7.post1`），不再回溯。

---

### 问题 2：`setup.py develop` 崩溃（最终 EXIT=1 的直接原因）

**现象**
```
python setup.py develop
  → 内部 subprocess.check_call(['python', '-m', 'pip', 'install', '-e', '.', '--use-pep517'])
  → traceback 末尾：
      subprocess.CalledProcessError: Command '[... pip install -e . --use-pep517]' returned non-zero exit status 1.
```
更内层的错误（build 阶段）：
```
Getting requirements to build editable: finished with status 'error'
error: subprocess-exited-with-error
  ╰─> [19 lines of output]
      File "<string>", line 13, in <module>
      ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'file:///root/repos/dex_lib/mixedai' when getting requirements to build editable
```

**原因**
- `setup.py` 第 13–14 行顶层直接 `import torch`、`from torch.utils.cpp_extension import ...`（第 17–18 行还用 `torch.__version__` 做断言）。
- 项目**没有 `pyproject.toml`**，pip 默认用 PEP517 后端（`setuptools.build_meta`），在**隔离的 build 环境**（`/tmp/pip-build-env-*/overlay/`）里执行 `setup.py` 获取构建元数据。
- 隔离 build 环境是全新环境，**不含 torch** → `import torch` 直接崩。
- `setup.py develop` 这个 setuptools 命令内部又被转成了 `pip install -e . --use-pep517`，所以绕不开 PEP517 隔离构建。

**为什么是 PEP517 隔离构建**
- PEP 517 规定：无 `pyproject.toml` 时，pip 用默认 `setuptools.build_meta` 作为 build backend，并默认开启 `--use-pep517` 隔离构建。
- 隔离构建会先在一个临时的干净环境里安装 `build-system.requires`，再在其中运行 `setup.py`。由于项目没声明 torch 为构建依赖，隔离环境里自然没有 torch。

**修复（关键改动）**
- 不用 PEP517 隔离构建，改用**当前已激活环境**（py310 里已装 torch 2.7.0+cu128、CUDA 12.8、nvcc）直接构建：
  ```bash
  pip install . --no-build-isolation
  ```
- `--no-build-isolation` 让 pip 跳过隔离环境，直接在当前解释器环境执行 `setup.py`，torch 可用，CUDA 编译链齐全。

---

### 问题 3：editable install 不被支持（隐藏坑）

**现象**
在问题 2 用 `--no-build-isolation` 后，如果仍保留 `-e`（editable），会报：
```
ERROR: Project file:///root/repos/dex_lib/mixedai uses a build backend that is
missing the 'build_editable' hook, so it cannot be installed in editable mode.
Consider using a build backend that supports PEP 660.
```

**原因**
- editable 安装（`pip install -e .`）需要 PEP 660 定义的 `build_editable` hook。
- `requirements.txt` 第 18 行强制 `setuptools==59.8.0`（2021 年的老版本），**不支持 PEP 660**（PEP 660 需要 setuptools>=64）。
- 老 setuptools 只提供旧的 `setup.py develop` 路径，而 pip 现代版本对 `-e` 默认走 PEP 660。

**修复**
- 先放弃 editable，改用普通安装（非 `-e`）作为第一步绕开 PEP660：
  ```bash
  pip install . --no-build-isolation
  ```
- 普通安装同样会编译 CUDA extension 并把 `.so` 装进 site-packages，只是不做「源码目录 editable 链接」。
- **最终演进（见第六节）**：后续将 `setuptools==59.8.0` 放宽为 `>=65.5.0` 后已**恢复 editable**（`pip install -e . --no-build-isolation`），实现改纯 Python 即改即生效。

---

## 四、修复后的脚本

`scripts/install_dev.sh` 最终内容：

```bash
# 开发者模式安装脚本
pip config set global.index-url http://192.168.3.43:8080/simple/
pip config set global.trusted-host 192.168.3.43
pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple
# 安装私有python包
pip install --no-cache-dir -r requirements-dev.txt --index-url http://192.168.3.43:8080/simple/ --trusted-host 192.168.3.43

# editable 安装（-e）：绕开 PEP517 隔离构建 torch 缺失；setuptools>=65.5 支持 PEP660
pip install -e . --no-build-isolation
```

### 修改文件清单

| 文件 | 原始内容 | 修改后 |
|------|---------|--------|
| `scripts/install_dev.sh` | `python setup.py develop` | `pip install -e . --no-build-isolation` |
| `requirements.txt` 第 43 行 | `pymeshlab==2023.12` | `pymeshlab>=2023.12.post3` |
| `requirements-dev.txt` 第 43 行 | `pymeshlab==2023.12` | `pymeshlab>=2023.12.post3` |
| `requirements.txt` 第 18 行 | `setuptools==59.8.0` | `setuptools>=65.5.0` |
| `requirements-dev.txt` 第 18 行 | `setuptools==59.8.0` | `setuptools>=65.5.0` |

> 注意 1：`setup.py` 的 `install_requires=get_requirements(platform_arg)` 读取的是 `requirements.txt`（非 dev），所以 **pymeshlab 和 setuptools 两个文件都要改**，否则 `pip install .` / `pip install -e .` 会因 install_requires 里的旧约束再次触发冲突或降级。
>
> 注意 2：`setuptools==59.8.0` 是 2021 年老版，**不支持 PEP 660 的 `build_editable` hook**，是 editable 安装失败的直接原因。放宽为 `setuptools>=65.5.0`（PEP 660 需 setuptools>=64）后方可走 editable。

---

## 五、验证结果

### 1. pip 依赖解析成功
- 130 个包全部 `Successfully installed`，含 `pymeshlab-2025.7.post1`。
- 不再出现 `ResolutionImpossible` / `looking at multiple versions`。

### 2. CUDA extension 编译成功
```
Building wheel for mixedai: finished with status 'done'
Created wheel for mixedai: filename=mixedai-0.6.3-cp310-cp310-linux_x86_64.whl size=2112251
Successfully installed mixedai-0.6.3
EXIT=0
```

### 3. `mixedai._C` CUDA ops 可用
```bash
cd /tmp && LD_LIBRARY_PATH=/root/miniconda3/envs/py310/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH python -c "
import torch, mixedai, mixedai._C as C
print(torch.__version__, torch.cuda.is_available())   # 2.7.0+cu128 True
print(mixedai.__version__)                             # 0.6.3
print('ops:', [n for n in dir(C) if not n.startswith('_')][:8])
# box_iou_rotated, ms_deform_attn ... 13 个 CUDA ops
print(torch.randn(4,4,device='cuda').sum().item())     # CUDA 运算正常
"
```
输出：`ALL OK`。

---

## 六、遗留问题 / 注意事项

### 1. `import mixedai._C` 需要 `LD_LIBRARY_PATH` 含 torch/lib
- `mixedai/_C.cpython-310-...so` 链接了 libtorch 的 `libc10.so`。
- 当前容器 `LD_LIBRARY_PATH`（.bashrc 第 1 行只有 `glia/lib`）**没有** `torch/lib`，普通 shell 里 `import mixedai._C` 报：
  ```
  ImportError: libc10.so: cannot open shared object file
  ```
- 临时解决：`export LD_LIBRARY_PATH=/root/miniconda3/envs/py310/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH`
- **已修复（2026-08-17）**：已把 `torch/lib` 追加到容器 `.bashrc` 第 1 行（放在 `glia/lib` 之后，保持 glia 自带 ORT 优先级不变）。新 shell source 后自动生效，`import mixedai._C` 无需手动加路径即可成功：
  ```
  export LD_LIBRARY_PATH=/root/miniconda3/envs/py310/lib/python3.10/site-packages/glia/lib:/root/miniconda3/envs/py310/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
  ```

### 2. 关于 editable 的取舍（已实现）
- 最初为绕开 PEP660 问题采用了**普通安装**（非 `-e`），牺牲「源码即改即生效」。
- **已升级为 editable（2026-08-17）**：将两个 requirements 的 `setuptools==59.8.0` 放宽为 `setuptools>=65.5.0`，并把脚本改为 `pip install -e . --no-build-isolation`。验证通过：
  - `setuptools: 84.0.0`（全程不再降级）
  - `mixedai.__file__` 指向源码 `/root/repos/dex_lib/mixedai/mixedai/__init__.py`（即改即生效）
  - `mixedai._C` 可用
- **踩坑**：仅改 requirements 后重装 editable 仍会失败，因为当前环境 setuptools 已被降回 59.8.0（install_requires 里旧的 `==59.8.0` 造成的）。必须**先手动 `pip install "setuptools>=65.5.0"` 升回来**，再跑 editable——此时因 requirements 已放宽，不再被降级。
- **语义**：editable 只对**纯 Python 代码**「即改即生效」；改 `.cu`/`.cpp` 仍需重编译（`pip install -e . --no-build-isolation`，ninja 有缓存较快），这是编译型代码的固有行为。

### 3. 下载慢是内网带宽问题，非脚本 bug
- requirements 里大量几十上百 MB 的 wheel，内网源 1.5~2 MB/s，单次完整安装本就需 5~15 分钟，属正常现象。

---

## 七、关键命令速查

```bash
# 进入容器并激活 py310
docker exec -it gwc_cuda128 bash
conda activate py310

# 查看当前解释器（确认是 py310）
which python && python -V    # /root/miniconda3/envs/py310/bin/python, 3.10.12

# 查看 pip / CUDA 环境就绪情况
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"   # /usr/local/cuda
which nvcc && nvcc --version | tail -1                                          # cuda_12.8

# 跑修复后的安装脚本（editable 安装）
cd /root/repos/dex_lib/mixedai
bash scripts/install_dev.sh

# editable 安装需要新版 setuptools（>=64 支持 PEP660），若环境里是旧的 59.8.0 需先升级
pip install "setuptools>=65.5.0"

# 单独验证 editable 编译（关 PEP517 隔离）
pip install -e . --no-build-isolation

# 验证 CUDA extension
cd /tmp && LD_LIBRARY_PATH=/root/miniconda3/envs/py310/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
  python -c "import mixedai._C; print('_C OK')"
```

---

## 八、相关背景

- 本次 mixedai 安装是「在 mixedai 库中新增 `build_gwc_volume` 的 sipe plugin 算子」任务的**前置环境准备**。
- 参考 plugin 实现位于 `/root/repos/sipe-nocs/mixedai/hpc_quant/deploy_utils_hpc/plugin/`（含 `src/gwc_volume_plugin.cpp`、`src/depth_kernels.cu`、`include/ffs_gwc_plugin.hpp` 及多个 TRT 版本已编译 engine）。
- 任务仓库 `dex_lib/mixedai` 的 `csrc/` 目前只有 pytorch 扩展算子（deformable attention / ROIAlign / box_iou_rotated 等），**尚无 TRT plugin 集成**，需后续将 `build_gwc_volume` plugin 移植进来。
