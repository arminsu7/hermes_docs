# Qwen3-ASR-1.7B-Int4 在 AGX Orin 上执行 `python main.py` 的报错分析

## 1. 问题概述

目标是在 AGX Orin 的 `smr_asr_nsys` 容器中运行：

```bash
cd /home/workspace/armin_profile/tts/code/dexe_agent/src/local_model_deploy/asr_server
python3 main.py
```

使用模型：

`/home/workspace/armin_profile/tts/code/dexe_agent/src/local_model_deploy/asr_server/app/models/Qwen/Qwen3-ASR-1.7B-Int4`

实际结果：服务未启动成功，vLLM 在模型权重加载阶段报错退出。

核心结论：
- 这不是 `main.py` 业务逻辑错误。
- 这不是显存不足或 tokenizer 主故障。
- 根因是：`llmcompressor` 导出的 `compressed-tensors` INT4 权重格式，与当前 `qwen_asr + vLLM` 的权重加载逻辑不兼容。

## 2. 复现环境

| 项目 | 实际值 |
|---|---|
| 设备 | NVIDIA Jetson AGX Orin |
| 容器 | `smr_asr_nsys` |
| 工作目录 | `/home/workspace/armin_profile/tts/code/dexe_agent/src/local_model_deploy/asr_server` |
| Python | `python3` |
| vLLM | `0.14.0` |
| PyTorch | `2.9.1` |
| 模型 | `Qwen3-ASR-1.7B-Int4` |

## 3. 实际报错现象

### 3.1 启动阶段日志

程序启动后，vLLM 成功识别了模型目录，并开始初始化引擎：

```text
INFO ... model='/home/workspace/.../Qwen3-ASR-1.7B-Int4'
INFO ... Resolved architecture: Qwen3ASRForConditionalGeneration
INFO ... Using max model len 65536
INFO ... Starting to load model ...
```

随后在权重加载阶段报错：

```text
KeyError: 'layers.0.self_attn.qkv.weight_packed'
```

最外层异常为：

```text
RuntimeError: Engine core initialization failed. See root cause above.
```

### 3.2 完整故障位置

关键调用链如下：

```text
main.py
 -> app/qwen3_asr.py::_create_vllm_engine
 -> qwen_asr.inference.qwen3_asr.py::LLM
 -> vllm LLMEngine
 -> qwen_asr/core/vllm_backend/qwen3_asr.py::load_weights
 -> KeyError: 'layers.0.self_attn.qkv.weight_packed'
```

## 4. 已验证证据

### 4.1 量化配置文件证据

模型目录内存在：

```text
quantize_config.json
```

内容为：

```json
{
  "quant_method": "compressed-tensors",
  "format": "int4-quantized",
  "bits": 4,
  "group_size": 128,
  "symmetric": true,
  "strategy": "group"
}
```

这说明该模型不是普通 BF16/FP16 权重，也不是 GPTQ/AWQ 命名体系，而是 `llmcompressor` 导出的 `compressed-tensors` INT4 格式。

### 4.2 safetensors 内真实权重名证据

实际检查 `model.safetensors` 后，发现存在的键是分离的投影权重，例如：

```text
thinker.model.layers.0.self_attn.q_proj.weight_packed
thinker.model.layers.0.self_attn.k_proj.weight_packed
thinker.model.layers.0.self_attn.v_proj.weight_packed
```

而不存在：

```text
thinker.model.layers.0.self_attn.qkv.weight_packed
```

这说明量化产物按 `q_proj / k_proj / v_proj` 三个独立张量存储，而不是预先融合成 `qkv.weight_packed`。

### 4.3 qwen_asr vLLM backend 的加载逻辑证据

实际检查容器内：

```text
/usr/local/lib/python3.10/dist-packages/qwen_asr/core/vllm_backend/qwen3_asr.py
```

其 `load_weights()` 中存在如下映射逻辑：

```python
stacked_params_mapping = [
    ("self_attn.qkv.", "self_attn.q_proj.", "q"),
    ("self_attn.qkv.", "self_attn.k_proj.", "k"),
    ("self_attn.qkv.", "self_attn.v_proj.", "v"),
]
```

随后它会执行类似逻辑：

```python
name = name.replace(weight_name, param_name)
param = params_dict[name]
```

也就是说：
- 当 loader 读到 `q_proj.weight_packed` 时
- 它会把名字改写成 `qkv.weight_packed`
- 然后去 `params_dict` 里查找该参数

最终查找失败，触发：

```text
KeyError: 'layers.0.self_attn.qkv.weight_packed'
```

## 5. 根因分析

### 5.1 现象是什么

现象是：
- vLLM 引擎可以启动到模型加载阶段
- 但在加载第 0 层 attention 权重时崩溃
- 崩溃点是 `params_dict[name]`，即“预期参数名”和“模型实际注册参数名”不一致

### 5.2 为什么会发生

根因是“权重存储协议”和“backend 参数加载协议”不匹配。

具体来说：

1. `llmcompressor` 导出的 `compressed-tensors` INT4 产物，把量化权重按独立线性层保存：
   - `q_proj.weight_packed`
   - `k_proj.weight_packed`
   - `v_proj.weight_packed`
   - 以及对应的 `weight_scale`、`weight_shape`

2. 当前 `qwen_asr` 的 vLLM backend，在加载 attention 权重时，假定内部存在一个融合参数：
   - `self_attn.qkv.*`

3. backend 试图把输入权重从 `q_proj/k_proj/v_proj` 映射到 `qkv`，但当前实际模型参数注册方式没有提供对应的 `qkv.weight_packed` 条目，因此 `params_dict` 查找失败。

本质上，这是“命名与结构协议不一致”问题，不是单纯的路径问题，也不是量化文件损坏。

### 5.3 从代码/设计层面看为什么会错

这里的冲突分两层：

#### 第一层：attention 参数组织方式不同

量化产物按三个独立线性层保存；当前 backend 却尝试按“融合 QKV 参数”加载。

这两种组织方式在 FP16/BF16 常常可以通过 shard loader 兼容，但在 `compressed-tensors` 下不一定成立，因为 packed 权重不只是数值精度变化，还带有额外元信息和特殊内存布局。

#### 第二层：packed quantized tensor 不是普通 dense tensor

`weight_packed` 不是普通的 `weight`：
- 它可能带有压缩布局
- 需要匹配 `weight_scale`
- 需要匹配 `weight_shape`
- backend/kernel 可能还要求特定排列顺序

因此即便把名字机械替换成功，也不代表后续 kernel 一定能正确消费。当前报错发生在更早的“参数名查找”阶段，说明兼容性问题甚至还没走到算子执行层。

## 6. 为什么旧版本或其他模型可能不报错

这个问题必须拆开看。

### 6.1 为什么未量化模型不报错

未量化模型通常保存的是：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

这类普通 dense 权重，backend 更容易在加载时做 shard 合并或内部转换，因此不会触发当前这种 `weight_packed` 级别的命名冲突。

### 6.2 为什么其他量化格式可能不报错

如果量化格式本身就是 vLLM 更成熟支持的路线，例如 GPTQ / AWQ / Marlin 相关路径，vLLM 内部已有较完整的：
- 参数注册
- 命名映射
- kernel 对接
- packed layout 处理

所以同样是 INT4，也不一定会触发这个问题。

### 6.3 为什么 vLLM 声称支持 compressed-tensors，但这里仍然报错

因为“vLLM 支持 compressed-tensors”是框架层面的能力，不等于“所有自定义模型 backend 都自动兼容”。

当前路径不是 vLLM 原生直接加载一个标准 decoder-only 文本模型，而是：

```text
Qwen3-ASR 自定义结构
+ qwen_asr 自定义 vllm_backend
+ compressed-tensors INT4 权重
+ Jetson Orin 环境
```

中间多了自定义 backend 这一层，只要它的权重注册逻辑和 compressed-tensors 的命名协议没完全打通，就会在加载阶段出错。

## 7. 旁证验证

为进一步确认不是单点偶发错误，还做了两个旁证检查。

### 7.1 Transformer 引擎可“继续加载”，但并未真正正确吃进量化权重

将同一模型改走 `EngineType.TRANSFORMER` 路径时，没有立即出现同样的 KeyError，但日志中出现大量警告：

- `... weight_packed / weight_scale / weight_shape` 未被使用
- 对应的浮点 `weight` 被新初始化

这说明：
- Transformers 路径也没有真正理解当前量化产物
- 它只是“没有在相同位置崩掉”，并不代表模型能正确推理

### 7.2 tokenizer warning 不是主因

日志里有：

```text
fix_mistral_regex=True
```

这是 tokenizer 兼容警告，不会导致本次引擎初始化失败。

### 7.3 Hugging Face repo id warning 不是主因

日志里还有：

```text
Error retrieving safetensors: Repo id must be in the form ...
```

这不是最终致命错误，因为后续程序仍然继续尝试从本地目录加载模型。真正终止启动的是后续的 `KeyError`。

## 8. 解决方案建议（只给方案，不直接修）

### 8.1 方案 A：不要用当前这份 compressed-tensors INT4 模型走 vLLM

这是最稳妥的方案。

建议：
- 先换回未量化模型 `Qwen3-ASR-1.7B`
- 或改用当前 `qwen_asr + vLLM` 更成熟支持的量化格式，例如 AWQ / GPTQ

适用场景：
- 当前目标是先把 ASR 服务跑起来
- 优先验证接口、流式状态、识别效果

### 8.2 方案 B：保留 INT4，但改用真正支持该量化格式的推理路径

如果业务必须使用 `llmcompressor` 产物，需要让“权重格式”和“加载器”完全匹配。

理论上有两条方向：

1. 使用能正确消费 `compressed-tensors` 的推理引擎/加载路径
2. 确认 Qwen3-ASR 这类 ASR 自定义结构在该引擎上也支持

难点在于：
- 不是所有支持 `compressed-tensors` 的框架都支持 Qwen3-ASR
- 不是所有支持 Qwen3-ASR 的路径都支持 packed INT4

### 8.3 方案 C：修改 qwen_asr 的 vLLM backend 以适配 compressed-tensors

这属于开发修复路线。

核心改动点会在：

```text
/usr/local/lib/python3.10/dist-packages/qwen_asr/core/vllm_backend/qwen3_asr.py
```

可能需要做的事情包括：

1. 针对 `compressed-tensors` 分支，不再强制把：
   - `q_proj.weight_packed`
   - `k_proj.weight_packed`
   - `v_proj.weight_packed`
   映射成：
   - `qkv.weight_packed`

2. 要么在模型参数注册阶段真的提供 `qkv.weight_packed`
3. 要么调整 `load_weights()`，允许分别加载 q/k/v 的 packed 权重
4. 同时补齐：
   - `weight_scale`
   - `weight_shape`
   - 可能存在的 group-wise packed layout 处理

这不是简单字符串替换，因为 packed tensor 的内存语义和后端 kernel 预期需要一致。

## 9. 最终判断

一句话总结：

当前这份 `llmcompressor` 导出的 `compressed-tensors` INT4 模型，实际保存的是“分离的 packed Q/K/V 权重”；而当前 `qwen_asr` 的 vLLM backend 却按“融合的 qkv packed 参数”去接，两者权重命名和加载协议没有对齐，所以在模型加载阶段直接报 `KeyError`。

因此，这个问题的定位应当是：
- 不是部署参数没调好
- 不是 `main.py` 代码错误
- 而是一个明确的 backend 兼容性问题

## 10. 推荐行动

如果目标是尽快恢复服务：

1. 先切回未量化模型跑通 ASR 服务
2. 如果需要量化，优先评估 AWQ / GPTQ 这类 vLLM 更成熟的量化格式
3. 把 `compressed-tensors` 路线降级为后续专项适配任务
