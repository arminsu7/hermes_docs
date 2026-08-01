# Fast-FoundationStereo 复现文档

> 项目地址：https://github.com/NVlabs/Fast-FoundationStereo
> 复现环境：WSL2 (Ubuntu 24.04) + RTX 3060 12GB + Docker 29.5.1
> 目标：构建 `dockerfile_cpp` 镜像（含 TRT 10.11 + C++ plugin 编译支持）

---

## 第一章：Docker 环境构建

### 1.1 环境概览

| 项目 | 详情 |
|---|---|
| OS | WSL2 Ubuntu 24.04, kernel 6.6.114.1 |
| GPU | NVIDIA GeForce RTX 3060 (12GB, SM 8.6) |
| 驱动 | 596.49 |
| Docker | 29.5.1 (原生 WSL2, 非 Docker Desktop) |
| 代理 | HTTP 代理 127.0.0.1:7897 |

### 1.2 两个 Dockerfile 的区别

官方仓库 `docker/` 目录下有两个 Dockerfile：

| | `dockerfile`（官方推荐） | `dockerfile_cpp`（C++ 开发用） |
|---|---|---|
| PyTorch | 2.6.0 + cu124 | 2.9.1 + cu128 |
| xformers | ✅ | ❌ |
| libnvinfer-dev | ❌ | ✅（编译 TRT plugin 必需） |
| onnxscript / cuda-python | ❌ | ✅ |
| 用途 | 推理 + ONNX/TRT 转换 | 推理 + C++ plugin 开发 |

**选择 `dockerfile_cpp` 的原因**：后续需要做 TRT plugin 开发，需要 `libnvinfer-dev`、`libnvinfer-plugin-dev` 等 C++ 编译头文件。

### 1.3 原始 Dockerfile 的已知问题

在 build 之前，发现原始 `dockerfile_cpp` 存在以下问题：

#### 问题 1：nvidia-modelopt 会覆盖 PyTorch 版本

原始代码（第 39 行）：
```dockerfile
uv pip install onnxruntime-gpu onnx onnxscript pycuda cuda-python \
    tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12 && nvidia-modelopt[torch] &&\
```

**根因**：`nvidia-modelopt[torch]` 声明了对 torch 的传递依赖。如果 modelopt 要求的 torch 版本范围与已安装的 `torch==2.9.1` 不兼容，`uv pip install` 的依赖解析器会自动将已安装的 torch 替换为兼容版本（silently replace），导致后续 torchvision ABI 不匹配：
```
RuntimeError: operator torchvision::nms does not exist
```

Dockerfile 自身的注释也承认了这个问题（第 32-36 行），但代码中仍然安装了 modelopt——这是自相矛盾的。

**修复**：删除 `nvidia-modelopt[torch]` 的安装。后续如需量化工具，手动安装。

#### 问题 2：pip install 全部放在一个 RUN 命令中

原始代码将所有 Python 包安装塞在同一个 RUN 命令里（第 37-41 行）：
```dockerfile
RUN conda activate my &&\
    uv pip install -r /tmp/requirements.txt &&\
    uv pip install onnxruntime-gpu onnx onnxscript pycuda cuda-python \
        tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12 &&\
    conda install -y -c anaconda h5py &&\
    conda install -y -c conda-forge libstdcxx-ng
```

**问题**：一旦其中任何一个子命令失败，整个 layer 需要重建，所有包重新下载（即使前面的 `requirements.txt` 已安装成功）。

**修复**：拆分为多个独立 RUN 命令，让成功的步骤可以被 Docker layer cache 复用。

### 1.4 网络配置

#### 1.4.1 背景

基础镜像 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 约 4GB，Docker Hub 在国内直连超时。国内 Docker 镜像加速器（daocloud、1ms、xuanyuan 等）不缓存 nvidia/cuda 的大层 blob。

#### 1.4.2 Docker daemon 代理配置

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf << 'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7897"
Environment="HTTPS_PROXY=http://127.0.0.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

验证代理是否生效：
```bash
sudo systemctl show docker --property=Environment
# 应输出 HTTP_PROXY、HTTPS_PROXY、NO_PROXY
```

#### 1.4.3 build 容器代理传递

Docker daemon 的代理环境变量**不会**自动传递给 `docker build` 容器内的进程（如 apt-get、pip）。必须通过 `--build-arg` 显式传递：

```bash
docker build \
    --network host \
    --build-arg HTTP_PROXY=http://127.0.0.1:7897 \
    --build-arg HTTPS_PROXY=http://127.0.0.1:7897 \
    --build-arg http_proxy=http://127.0.0.1:7897 \
    --build-arg https_proxy=http://127.0.0.1:7897 \
    -t ffs-cpp -f docker/dockerfile_cpp .
```

注意：`--network host` 让容器共享宿主机网络（不是必须，但可以减少一层 NAT）。

#### 1.4.4 部分站点须直连

经测试发现，以下站点**直连**比走代理更快/更稳定：

| 站点 | 代理 | 直连 | 最终选择 |
|---|---|---|---|
| `pypi.nvidia.com` | 大文件 SHA256 损坏 | ✅ 稳定 | 直连 |
| `developer.download.nvidia.com` → `nvidia.cn` | 250 KB/s | 100+ MB/s | 直连 |
| `repo.anaconda.com` | 较慢但可用 | ❌ 超时 | 代理 |
| `pypi.org` | 不稳定 | ✅ 可用 | 视情况 |

因此在 Dockerfile 中对需要直连的 RUN 命令加了 `unset HTTP_PROXY ...` 前缀。

### 1.5 最终 Dockerfile

以下是修改后的 `docker/dockerfile_cpp`（修改处标注 `# <-- CHANGED`）：

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils build-essential ca-certificates cmake curl ffmpeg git \
        libturbojpeg-dev pkg-config wget \
        libnvinfer10 libnvinfer-dev libnvinfer-plugin10 libnvinfer-plugin-dev \
        libnvonnxparsers10 libnvonnxparsers-dev libnvinfer-dispatch10 libnvinfer-bin \
        zstd libyaml-cpp-dev libopencv-dev \
        libx11-xcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-render-util0 \
        libxcb-shape0 libxcb-keysyms1 libxcb-image0 libxkbcommon-x11-0 \
        libxcb-cursor0 libxcb-xkb1 libxcb-render0 libxcb-shm0 libxcb-sync1 \
        libxcb-xfixes0 libxcb-randr0 libxcb-xtest0 libsm6 libxext6 libxkbcommon0 && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "--login", "-c"]

# Step 3: Miniconda
RUN cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    /bin/bash -c "source ~/.bashrc" && \
    /opt/conda/bin/conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main &&\
    /opt/conda/bin/conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r &&\
    /opt/conda/bin/conda update -n base -c defaults conda -y &&\
    /opt/conda/bin/conda create -n my python=3.12

ENV PATH=$PATH:/opt/conda/envs/my/bin
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Step 4: PyTorch (conda 环境 my)
RUN conda init bash &&\
    echo "conda activate my" >> ~/.bashrc &&\
    conda activate my &&\
    pip install uv &&\
    uv pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt

# Step 5: Python 基础依赖（requirements.txt）
# 注意：nvidia-modelopt 会导致 torch 被静默替换，已移除 # <-- CHANGED
RUN conda activate my && \
    uv pip install -r /tmp/requirements.txt

# Step 6: NVIDIA Python 包（直连，不走代理）# <-- CHANGED
RUN conda activate my && \
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    for i in 1 2 3; do \
        pip install --extra-index-url https://pypi.nvidia.com/ \
            --retries 10 --timeout 300 \
            onnxruntime-gpu onnx onnxscript pycuda cuda-python \
            tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12 && break; \
        echo "Attempt $i failed, retrying..."; sleep 5; \
    done

# Step 7: h5py + libstdcxx-ng # <-- CHANGED
RUN conda activate my && \
    for i in 1 2 3; do \
        pip install h5py && break; \
        echo "Attempt $i failed, retrying..."; sleep 5; \
    done && \
    conda install -y -c conda-forge libstdcxx-ng

# Step 8: TensorRT 10.11 SDK（直连国内 CDN）# <-- CHANGED
RUN unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    cd / && wget https://developer.download.nvidia.com/compute/tensorrt/10.11.0/\
local_installers/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb &&\
    apt install -y \
        ./nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb &&\
    echo 'alias trtexec="/usr/src/tensorrt/bin/trtexec"' >> ~/.bashrc &&\
    rm -f /nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb

SHELL ["/bin/bash", "-c", "source ~/.bashrc && conda activate my"]
```

**修改总结**（对比原始 `dockerfile_cpp`）：

| 修改 | 原因 |
|---|---|
| 删除 `nvidia-modelopt[torch]` | 会静默替换 PyTorch 版本，导致 torchvision ABI 不匹配 |
| 拆分 RUN 命令（requirements.txt / tensorrt / h5py 独立） | 失败时不影响已成功的步骤，利用 Docker layer cache |
| tensorrt pip install 去代理（`unset HTTP_PROXY ...`） | 代理不稳定，pypi.nvidia.com 直连 SHA256 校验通过 |
| tensorrt pip install 加重试循环（3 次） | 应对偶发网络波动 |
| h5py 改为 pip 安装 + 重试 | conda 走代理也偶发断连，pip 更稳定 |
| TRT .deb 下载去代理 | 直连自动跳转国内 CDN (`nvidia.cn`)，速度 100+ MB/s vs 代理 250 KB/s |

### 1.6 完整构建步骤

#### Step 1：克隆仓库

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
cd Fast-FoundationStereo
```

#### Step 2：修改 Dockerfile（按 1.5 节）

编辑 `docker/dockerfile_cpp`，应用上述修改。

#### Step 3：配置 Docker 代理（如需要）

```bash
# 创建 systemd override
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf << 'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7897"
Environment="HTTPS_PROXY=http://127.0.0.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
EOF

# 重载并重启
sudo systemctl daemon-reload
sudo systemctl restart docker

# 验证
sudo systemctl show docker --property=Environment
```

#### Step 4：构建镜像

```bash
docker build \
    --network host \
    --build-arg HTTP_PROXY=http://127.0.0.1:7897 \
    --build-arg HTTPS_PROXY=http://127.0.0.1:7897 \
    --build-arg http_proxy=http://127.0.0.1:7897 \
    --build-arg https_proxy=http://127.0.0.1:7897 \
    -t ffs-cpp \
    -f docker/dockerfile_cpp .
```

**构建时间估算**：

| 步骤 | 内容 | 耗时 |
|---|---|---|
| 基础镜像 pull | nvidia/cuda:12.4.1 (~4GB) | ~40 min |
| apt install | 系统包 | ~3 min |
| Miniconda | 安装 + 创建环境 | ~2 min |
| PyTorch | torch 2.9.1 + cu128 | ~5 min |
| pip (requirements.txt) | open3d 427MB 等 | ~2 min |
| pip (tensorrt 等) | pycuda 编译 + 下载 | ~13 min |
| pip (h5py) + conda | 小包 | ~2 min |
| TRT .deb | 5GB 下载 + 安装 | ~23 min |
| 导出镜像 | layer 打包 | ~6 min |

#### Step 5：验证镜像

```bash
docker images ffs-cpp
# 输出示例：
# IMAGE            ID             DISK USAGE   CONTENT SIZE
# ffs-cpp:latest   4425ba665af7   70.9GB       28.8GB
```

#### Step 6：启动容器

```bash
# 使用官方 run_container.sh（注意修改镜像名和挂载路径）
docker run --gpus all --runtime nvidia \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it --network=host \
    --name ffs-cpp \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $(pwd)/..:/workspace \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    ffs-cpp bash
```

### 1.7 踩坑记录

#### 坑 1：国内 Docker 镜像源不缓存 nvidia/cuda 大层

**现象**：`docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 小层（几十 MB）能拉，大层（670MB + 2.65GB）始终停留在 "Pulling fs layer"，永不完成。

**根因**：阿里云、DaoCloud、1ms 等国内 Docker 镜像加速器**不缓存 nvidia/cuda 仓库的大层 blob**。这些镜像源主要缓存 library/ 下的常用镜像（如 alpine、ubuntu），而 nvidia/cuda 属于第三方仓库，只有元数据被代理，实际 blob 需要回源到 Docker Hub。在国内网络环境下，回源连接超时或极慢。

**解决**：配置 Docker daemon 走 HTTP 代理直连 Docker Hub。验证命令：
```bash
# 先验证代理是否可达
curl -s --connect-timeout 5 -x http://127.0.0.1:7897 \
    https://registry-1.docker.io/v2/ -o /dev/null -w "%{http_code}"
# 应返回 401（需要认证，说明连接成功）
```

#### 坑 2：build 容器不继承 daemon 代理环境变量

**现象**：配置了 systemd 代理后 `docker pull` 能正常工作，但 `docker build` 内的 `apt-get`、`pip` 仍然超时。

**根因**：systemd 的 `Environment=` 只影响 Docker daemon 进程本身。`docker build` 启动的容器是独立进程，不继承 daemon 的环境变量。需要通过 `--build-arg` 显式传入。

**验证与解决**：
```bash
# 验证 daemon 有代理
cat /proc/$(pgrep dockerd)/environ | tr '\0' '\n' | grep -i proxy

# 构建时显式传入（大小写都要）
docker build --build-arg HTTP_PROXY=... --build-arg http_proxy=... ...
```

#### 坑 3：代理对大文件下载不稳定（SHA256 不匹配）

**现象**：
```
Downloaded wheel and sha256 don't match!
tensorrt-cu12-libs==11.1.0.106: IncompleteRead(294912 bytes read, 9882698 more expected)
```

**根因**：`tensorrt-cu12-libs` 是占位包（wheel stub），安装时从 `pypi.nvidia.com` 下载真实 wheel（数百 MB）。代理服务器在处理大文件 HTTP 响应时偶发 TCP 连接中断，导致文件不完整，SHA256 校验失败。此问题在多次重试后仍复现（连续 3 次失败），说明不是瞬时波动，而是代理对特定站点的大文件传输有系统性缺陷。

**解决**：对 `pypi.nvidia.com` 的包安装前 `unset` 代理环境变量，直连下载。直连下载完整且 SHA256 校验通过。

```dockerfile
RUN conda activate my && \
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy && \
    pip install --extra-index-url https://pypi.nvidia.com/ ...
```

#### 坑 4：conda 走代理也偶发断连

**现象**：conda install 下载 `repo.anaconda.com` 的包时出现 `Connection broken: IncompleteRead`。

**根因**：与坑 3 相同——代理对大文件 HTTP 响应偶发断连。

**解决**：h5py 改为 pip 安装（走 PyPI，更稳定），conda 只用于 libstdcxx-ng（小包，断连概率低）。

#### 坑 5：TRT .deb 走代理极慢

**现象**：`nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb`（~5GB）通过代理下载速度仅 250 KB/s，预计 4 小时。

**根因**：代理服务器带宽有限或对大文件限速。

**解决**：`unset` 代理后 wget 直连。NVIDIA 的 `developer.download.nvidia.com` 会自动 301 重定向到 `developer.download.nvidia.cn`（国内 CDN，IP: 125.64.2.195），直连速度 100+ MB/s，5GB 约 50 秒完成。

验证直连可达性：
```bash
curl -s --connect-timeout 10 -o /dev/null -w "%{http_code}" \
    -L "https://developer.download.nvidia.com/compute/tensorrt/10.11.0/\
local_installers/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb"
# 应返回 200
```

#### 坑 6：Dockerfile 修改导致 layer cache 失效

**现象**：拆分 RUN 命令后，之前已成功的 `requirements.txt` 安装又被重新执行（package 全部重新下载）。

**根因**：Docker 的 layer cache 基于 RUN 命令的**完整文本**计算 hash。即使只改了一行，整个 RUN 命令的 hash 变了，cache 就失效。拆分后每个 RUN 独立缓存，后续修改只影响对应 layer。

**教训**：将可能失败的重试步骤拆分为独立 RUN，避免"一颗老鼠屎坏一锅粥"。一旦某步成功缓存，后续 build 调试只需重试失败的那一步。

### 1.8 最终镜像

```
IMAGE            ID             DISK USAGE   CONTENT SIZE
ffs-cpp:latest   4425ba665af7   70.9GB       28.8GB
```

容器内包含：
- Ubuntu 22.04 + CUDA 12.4.1 + cuDNN
- Miniconda (Python 3.12, 环境名 `my`)
- PyTorch 2.9.1 + torchvision 0.24.1 (CUDA 12.8)
- TensorRT 10.11.0 SDK + Python bindings
- libnvinfer-dev / libnvinfer-plugin-dev（C++ plugin 编译头文件）
- onnx / onnxruntime-gpu / onnxscript
- pycuda / cuda-python
- open3d / opencv-contrib-python / scipy / scikit-image / timm / h5py

---

## 后续章节（待补充）

- 第二章：模型权重下载
- 第三章：PyTorch 推理（run_demo.py）
- 第四章：ONNX 导出
- 第五章：TRT Engine 构建与推理
- 第六章：C++ Plugin 开发与集成
