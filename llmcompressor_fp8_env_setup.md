# llmcompressor FP8 量化环境搭建记录

## 目标

在容器 `smr2508` 中，为后续 bf16 -> fp8 量化工作创建独立虚拟环境：

- 虚拟环境路径：`/root/repos/hermes/llmcompressor_env`
- 文档路径：`/root/repos/hermes/docs`
- 量化工具：`llmcompressor`

说明：宿主机 `/home/armin/repos` 已 bind mount 到容器 `/root/repos`，因此容器内的：

- `/root/repos/hermes/llmcompressor_env`
- `/root/repos/hermes/docs`

分别对应宿主机上的：

- `/home/armin/repos/hermes/llmcompressor_env`
- `/home/armin/repos/hermes/docs`

## 实际环境信息

### 容器

- 容器名：`smr2508`
- 镜像：`nvcr.io/nvidia/pytorch:25.08-py3`
- 容器状态：运行中

### 容器内基础软件

- Python：`3.12.3`
- pip：`26.1.2`（容器系统环境）

### GPU 信息

实际探测结果：

- GPU：`NVIDIA GeForce RTX 3060`
- Driver：`596.49`
- 显存：`12288 MiB`
- Compute Capability：`8.6`

### 容器基础 PyTorch

容器系统自带 PyTorch：

- `torch 2.8.0a0+34c6371d24.nv25.08`
- CUDA：`13.0`

## 一个关键结论

官方 llmcompressor FP8 文档明确写到：

- `fp8` 计算支持的 NVIDIA GPU 需要 `compute capability > 8.9`
- 典型是 `Ada Lovelace / Hopper`

而当前机器实际 GPU 是：

- RTX 3060
- Compute Capability = `8.6`

所以结论是：

1. `llmcompressor` 的 FP8 量化环境已经搭好
2. 但当前这张卡不满足官方文档给出的 FP8 运行前提
3. 也就是说，这个环境适合先做工具链准备、脚本编写、依赖验证
4. 真正要稳定跑官方支持的 FP8 量化/推理，建议切到 Ada/Hopper 及以上平台

这点很重要。带宽像水管，算力像泵；你现在水管和泵都连好了，但阀门规格不匹配，官方这条 FP8 通路不会把 RTX 3060 当作“受支持硬件”。

## 完整执行过程

### 1. 确认容器和目标路径

执行：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

确认结果：

- `smr2508` 正在运行
- 镜像是 `nvcr.io/nvidia/pytorch:25.08-py3`

进入容器做基础检查：

```bash
docker exec smr2508 bash -lc 'pwd; whoami; python3 --version; python3 -m pip --version || true; ls -ld /root/repos/hermes /root/repos/hermes/docs 2>/dev/null || true'
```

确认：

- 当前目录：`/workspace`
- 用户：`root`
- `/root/repos/hermes` 存在
- `/root/repos/hermes/docs` 存在

### 2. 检查 GPU 与容器内 torch

执行：

```bash
docker exec smr2508 bash -lc 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader && python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO_CUDA\")"'
```

结果：

```text
NVIDIA GeForce RTX 3060, 596.49, 8.6, 12288 MiB
2.8.0a0+34c6371d24.nv25.08
13.0
True
NVIDIA GeForce RTX 3060
```

### 3. 检查 llmcompressor 版本来源

执行：

```bash
docker exec smr2508 bash -lc 'python3 -m pip index versions llmcompressor | sed -n "1,20p"'
```

结果：

- 可安装最新版：`llmcompressor 0.12.0`

### 4. 检查官方 FP8 示例

抓取到的关键信息有两类：

1. 官方要求：`compute capability > 8.9`
2. 推荐基础用法：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)
oneshot(model=model, recipe=recipe)
```

以及压缩保存形式：

```python
oneshot(
    model=model,
    recipe=recipe,
    tokenizer=tokenizer,
    output_dir=OUTPUT_DIR,
)
```

### 5. 检查目标虚拟环境目录

执行：

```bash
docker exec smr2508 bash -lc 'set -e
cd /root/repos/hermes
if [ -d llmcompressor_env ]; then echo VENV_EXISTS; else echo VENV_MISSING; fi
python3 -m venv --help >/dev/null && echo VENV_MODULE_OK
mkdir -p docs
'
```

发现：

- `llmcompressor_env` 目录已经存在
- 但后续检查发现它只是一个空目录，不是完整虚拟环境

继续检查：

```bash
docker exec smr2508 bash -lc 'set -e
if [ -d /root/repos/hermes/llmcompressor_env ]; then
  echo "== VENV TREE =="
  ls -la /root/repos/hermes/llmcompressor_env | sed -n "1,40p"
  if [ -f /root/repos/hermes/llmcompressor_env/pyvenv.cfg ]; then
    echo "== PYVENV =="
    cat /root/repos/hermes/llmcompressor_env/pyvenv.cfg
  fi
else
  echo "NO_VENV_DIR"
fi'
```

结果：

- 目录里只有 `.` 和 `..`
- 没有 `pyvenv.cfg`
- 说明需要重建

### 6. 重建虚拟环境并安装依赖

实际执行命令：

```bash
docker exec smr2508 bash -lc 'set -e
cd /root/repos/hermes
rm -rf llmcompressor_env
python3 -m venv llmcompressor_env
source llmcompressor_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install llmcompressor
python -m pip freeze | sort | tee /tmp/llmcompressor_env_freeze.txt >/dev/null
python - <<"PY"
import sys
mods=["llmcompressor","torch","transformers","compressed_tensors","accelerate"]
for m in mods:
    try:
        mod=__import__(m)
        print(m, getattr(mod, "__version__", "unknown"))
    except Exception as e:
        print(m, "IMPORT_FAIL", repr(e))
print("python", sys.version)
PY'
```

注意：我第一次执行验证脚本时，最后一行因为 shell 引号处理不严谨，触发了一个 `NameError`，后面已修正并重新验证，详见“踩坑与解决”。

## 实际安装的关键依赖

`pip install llmcompressor` 自动安装/拉取的关键包包括：

- `llmcompressor==0.12.0`
- `torch==2.12.0`
- `transformers==5.10.1`
- `compressed-tensors==0.17.1`
- `accelerate==1.13.0`
- `datasets==5.0.0`
- `auto-round==0.13.0`
- `numpy==2.4.6`
- `pydantic==2.13.4`
- `safetensors==0.8.0`
- `tokenizers==0.22.2`
- `cuda-toolkit==13.0.2`
- `nvidia-cublas==13.1.1.3`
- `nvidia-cudnn-cu13==9.20.0.48`

完整依赖快照已保存到：

- `/root/repos/hermes/docs/llmcompressor_fp8_env_requirements.txt`

宿主机对应路径：

- `/home/armin/repos/hermes/docs/llmcompressor_fp8_env_requirements.txt`

## 环境验证

### 1. 版本和 import 验证

实际执行：

```bash
docker exec smr2508 bash -lc 'set -e
source /root/repos/hermes/llmcompressor_env/bin/activate
python - <<'"'"'PY'"'"'
import sys, inspect, torch
import llmcompressor, transformers, compressed_tensors, accelerate
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
print("python", sys.version.split()[0])
print("llmcompressor", llmcompressor.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("compressed_tensors", compressed_tensors.__version__)
print("accelerate", accelerate.__version__)
print("oneshot_callable", callable(oneshot))
print("quant_modifier_ctor", str(inspect.signature(QuantizationModifier.__init__)))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("gpu_cc", f"{cc[0]}.{cc[1]}")
PY
python -m pip show llmcompressor | sed -n "1,40p"
python -m pip freeze | sort > /root/repos/hermes/docs/llmcompressor_fp8_env_requirements.txt
wc -l /root/repos/hermes/docs/llmcompressor_fp8_env_requirements.txt'
```

验证结果：

```text
python 3.12.3
llmcompressor 0.12.0
torch 2.12.0+cu130
transformers 5.10.1
compressed_tensors 0.17.1
accelerate 1.13.0
oneshot_callable True
quant_modifier_ctor (self, /, **data: 'Any') -> 'None'
cuda_available True
gpu_name NVIDIA GeForce RTX 3060
gpu_cc 8.6
```

并确认：

- `llmcompressor` 可正常 import
- `oneshot` 可调用
- `QuantizationModifier` 存在
- CUDA 可见
- 依赖快照文件已落盘，共 `86` 行

### 2. CLI 验证

执行：

```bash
docker exec smr2508 bash -lc 'source /root/repos/hermes/llmcompressor_env/bin/activate && llmcompressor.trace --help | sed -n "1,40p"'
```

结果：

- `llmcompressor.trace` 命令正常可用
- CLI 能正常输出参数帮助

## 踩坑与解决办法

### 坑 1：目标目录已经存在，但不是有效虚拟环境

现象：

- `/root/repos/hermes/llmcompressor_env` 已存在
- 但里面是空目录
- 没有 `pyvenv.cfg`

影响：

- 不能直接认为环境已经建好
- 如果在空目录上继续做不严格的判断，后续激活会失败

解决：

```bash
rm -rf /root/repos/hermes/llmcompressor_env
python3 -m venv /root/repos/hermes/llmcompressor_env
```

### 坑 2：验证脚本因为 shell 引号细节触发 `NameError`

现象：

第一次安装后追加的内嵌 Python 验证脚本最后报错：

```text
NameError: name 'python' is not defined
```

原因：

- shell heredoc / 引号拼接不够严谨
- 最后一行打印语句在传入 Python 时被错误展开

解决：

- 改为更安全的 heredoc 包装方式
- 重新执行独立验证命令
- 最终验证全部通过

### 坑 3：`pip install llmcompressor` 没有复用容器自带 torch，而是单独安装了一套新 torch

现象：

容器系统里原本有：

- `torch 2.8.0a0+34c6371d24.nv25.08`

但虚拟环境里实际安装的是：

- `torch 2.12.0+cu130`

影响：

- 虚拟环境体积会明显变大
- 会额外下载一批 CUDA 13 相关 wheel
- 建环境耗时和带宽占用都更高

这很正常，本质上是 Python 依赖解析在单独铺一条自己的 runtime 栈，相当于又拉了一套“专用管线”。

解决思路：

- 当前做法是接受这个隔离环境，优点是独立、可复现
- 如果后面你更想节省空间/复用容器基底，再考虑做依赖约束或尝试复用系统包
- 但那会提高环境耦合度，不适合作为第一版稳定方案

### 坑 4：当前 GPU 不满足官方 FP8 支持门槛

现象：

- 官方文档：`compute capability > 8.9`
- 当前 GPU：RTX 3060，`cc = 8.6`

影响：

- 环境能装
- Python 包能 import
- 但真正进入官方支持的 FP8 量化/推理路径时，硬件前提不满足

解决建议：

- 这个环境继续保留，作为 llmcompressor 开发/脚本准备环境
- 真正做 FP8 量化实验时，切到 Ada/Hopper 机器
- 如果你当前必须在 3060 上做压缩验证，建议考虑先做别的量化路线，例如更适配这代卡/生态的低比特权重量化或非官方路径

## 最终产物

### 已创建

- 虚拟环境：`/root/repos/hermes/llmcompressor_env`
- 文档目录：`/root/repos/hermes/docs`
- 依赖快照：`/root/repos/hermes/docs/llmcompressor_fp8_env_requirements.txt`

宿主机对应路径：

- `/home/armin/repos/hermes/llmcompressor_env`
- `/home/armin/repos/hermes/docs`
- `/home/armin/repos/hermes/docs/llmcompressor_fp8_env_requirements.txt`

### 建议的后续使用方式

进入容器并激活环境：

```bash
docker exec -it smr2508 bash
cd /root/repos/hermes
source llmcompressor_env/bin/activate
```

确认工具：

```bash
python -c "import llmcompressor, torch; print(llmcompressor.__version__, torch.__version__)"
llmcompressor.trace --help
```

## 参考的 FP8 示例骨架

后面你做 bf16 -> fp8 时，可以从这个骨架起步：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "your-bf16-model"
OUTPUT_DIR = "your-bf16-model-FP8"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    tokenizer=tokenizer,
    output_dir=OUTPUT_DIR,
)
```

但再强调一次：

- 这个脚本结构是对的
- 当前 RTX 3060 并不满足官方 FP8 硬件要求

## 关于“后续工作环境切到 smr2508”的说明

这次实际操作已经全部在 `smr2508` 容器内完成。

另外，我还把 Hermes 的默认 terminal backend 改成了 `docker`，镜像设为：

- `nvcr.io/nvidia/pytorch:25.08-py3`

但要注意：

- Hermes 的 docker backend 默认是“按镜像起容器”的工作模式
- 它不等价于“直接附着到已经存在的 `smr2508` 容器”
- 所以如果你要求的是“以后 Hermes 必须直接复用这个现成的 `smr2508` 容器实例”，还需要单独设计工作流（例如继续用 `docker exec smr2508 ...` 这一路）

也就是说：

- 本次任务已经在 `smr2508` 里完成
- Hermes 配置层面已经切到同镜像 docker backend
- 但“同镜像新容器”与“复用现有 smr2508 实例”不是一回事

这个区别后面别踩坑。
