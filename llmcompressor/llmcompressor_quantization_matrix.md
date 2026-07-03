# llmcompressor 量化方法 × 输出格式 × 推理框架对照表

## 目的

这份表是给后续选型用的：

- 你要做什么量化
- llmcompressor 能不能做
- 产物更适合接到哪个推理框架
- 哪些路线不要混用

结论先说：

- 如果目标是 vLLM，llmcompressor 很合适
- 如果目标是 llama.cpp / GGUF，llmcompressor 不是主通路
- AWQ / GPTQ / FP8 是“量化方法”
- GGUF 是“模型文件格式”

这两个维度不要混淆。

---

## 一张总表

| 方法 / 方案 | llmcompressor 是否支持 | 典型位宽 / 形式 | 输出主格式 | 更适合的推理框架 | 备注 |
|---|---|---:|---|---|---|
| FP8_DYNAMIC | 支持 | FP8（权重/激活） | compressed-tensors / HF save_pretrained | vLLM | 官方主推路线之一，硬件要求高 |
| GPTQ | 支持 | 常见 W4 / group-wise | compressed-tensors / HF save_pretrained | vLLM | 有 GPTQModifier，支持 actorder / block_size / group_size |
| AWQ | 支持 | 常见 W4A16 / 非对称组量化 | compressed-tensors / HF save_pretrained | vLLM | 有 AWQModifier / AWQ transform 流程 |
| AutoRound | 支持 | 常见 4-bit 权重量化 | compressed-tensors / HF save_pretrained | vLLM | 更偏自动调优权重量化 |
| SmoothQuant | 支持 | 量化前变换/平滑，不是单独部署格式 | 仍走 HF / compressed-tensors | vLLM / 后续 INT8 路线 | 更像前处理/增强步骤 |
| Log Equalization | 支持 | 量化前变换 | 仍走 HF / compressed-tensors | vLLM / 后续量化链 | 本质上并到 SmoothQuantModifier |
| 通用 W8A8 | 支持 | INT8 权重+激活 | compressed-tensors / HF save_pretrained | vLLM | 通过 QuantizationModifier/preset scheme 配置 |
| 通用 W4A16 | 支持 | INT4 权重 + FP16/BF16 激活 | compressed-tensors / HF save_pretrained | vLLM | AWQ/GPTQ/AutoRound 常落在这类部署形态 |
| KV Cache Quant | 支持配置入口 | KV 单独量化 | 仍随 HF / compressed-tensors 保存 | vLLM | 通过 kv_cache_scheme 配置 |
| GGUF 导出 | 不建议视为支持 | GGUF 文件格式 | 非 llmcompressor 主输出 | llama.cpp | llmcompressor 主保存链不是 GGUF |

---

## 分项说明

### 1. FP8

适合场景：

- Hopper / Ada / 更高代支持较好的 NVIDIA 平台
- 你要追求更高吞吐、更低显存压力
- 部署目标是 vLLM

llmcompressor 情况：

- 明确支持
- 常见写法：`QuantizationModifier(scheme="FP8_DYNAMIC")`
- 主输出路线：`save_pretrained` + compressed-tensors

推理框架建议：

- 首选：vLLM
- 不建议目标直接设成 GGUF / llama.cpp

注意：

- 你的当前 RTX 3060（cc 8.6）不在官方 FP8 支持主路径内

---

### 2. GPTQ

适合场景：

- 想做低比特权重量化
- 希望在精度与压缩率之间取得相对稳的平衡
- 部署目标偏 vLLM / HF 生态

llmcompressor 情况：

- 明确支持
- 有 `GPTQModifier`
- 代码里能看到典型参数：
  - `block_size`
  - `dampening_frac`
  - `actorder`
  - `group_size`

推理框架建议：

- 首选：vLLM
- 也可视具体生态接其他 HF 兼容链路
- 不建议把 llmcompressor 当 GGUF 导出器

---

### 3. AWQ

适合场景：

- 你要做激活感知的权重量化
- 常见是 W4A16 这一类部署形态
- 想在低比特下尽量保留质量

llmcompressor 情况：

- 明确支持
- 可见 `AWQModifier`
- 新接口更偏：
  - `transform.awq.AWQModifier`
  - 再配 `QuantizationModifier`
- 代码里还能看到多种模型架构 mapping，说明不是空壳支持

推理框架建议：

- 首选：vLLM
- 如果你最终目标是 GGUF / llama.cpp，建议直接走 GGUF 工具链，而不是先 llmcompressor 再硬转

---

### 4. AutoRound

适合场景：

- 想自动调优低比特权重量化参数
- 常见是 4-bit / group quant
- 偏“自动找更优 rounding”

llmcompressor 情况：

- 明确支持
- 有 `AutoRoundModifier`
- 代码里能看到：
  - `num_bits: 4`
  - `group_size: 128`
  - `symmetric: true`

推理框架建议：

- 更适合 vLLM / HF 路线
- 不把它当 GGUF 主链

---

### 5. SmoothQuant / Log Equalization

适合场景：

- 不是只看最终 bit 数，而是先做激活/权重分布整形
- 给后续量化创造更稳的统计条件

llmcompressor 情况：

- 明确支持 `SmoothQuantModifier`
- `LogarithmicEqualizationModifier` 实际已经并到：
  - `SmoothQuantModifier(algorithm="log_equalization")`

理解方式：

- 这更像“量化前变换”
- 不是单独的最终部署文件格式

推理框架建议：

- 更适合继续接 vLLM / HF 压缩保存链

---

## 关于 GGUF 的位置

### GGUF 是什么

GGUF 是 llama.cpp 生态里的模型文件格式。

它关注的是：

- 文件如何组织
- 权重如何存放
- llama.cpp 如何直接加载

所以它和 AWQ / GPTQ / FP8 不是同一层概念。

- AWQ / GPTQ / FP8：是“怎么压”
- GGUF：是“压完装进什么箱子里”

### llmcompressor 对 GGUF 的关系

从当前安装环境里实际查到的情况看：

- 主保存链是 `save_pretrained` + `compressed-tensors`
- 配置识别也偏 `quant_method = compressed-tensors`
- 没看到完整的原生 GGUF 导出主链
- 只在 AutoRound 附近看到一个 `enable_gguf_official_mixed=False` 的内部配置痕迹

所以实际工程建议是：

- 要 vLLM：用 llmcompressor，顺着 compressed-tensors 走
- 要 llama.cpp：优先直接走 GGUF/llama.cpp 工具链

不要把 llmcompressor 当作“通用 GGUF 导出器”。

---

## 选型建议

### 场景 A：目标是 vLLM 在线推理

优先顺序：

1. FP8（如果硬件支持）
2. AWQ / GPTQ
3. AutoRound
4. W8A8 / SmoothQuant 组合路线

建议：

- 输出保持在 HF / compressed-tensors
- 不要中途转 GGUF

### 场景 B：目标是 llama.cpp / 本地 CPU/边缘设备

建议：

- 直接走 GGUF 工具链
- 不要把 llmcompressor 作为核心导出工具

原因：

- llmcompressor 的主生态是 vLLM / HF
- llama.cpp 的主生态是 GGUF
- 两条链像两条总线，强接能接，但往往不优雅，还容易出兼容坑

### 场景 C：你要做研究对比

建议这样分组：

- 组 1：BF16 baseline
- 组 2：FP8_DYNAMIC
- 组 3：AWQ W4A16
- 组 4：GPTQ W4
- 组 5：AutoRound 4bit

统一：

- 同一校准集
- 同一评测集
- 同一推理框架（最好先统一在 vLLM）

这样对比才公平。

---

## 实操建议：你现在这台机器怎么选

你当前环境：

- GPU：RTX 3060
- Compute Capability：8.6

所以建议是：

1. 现在可以先完成 llmcompressor 工具链和脚本准备
2. 真要跑官方 FP8 主线，换到 Ada / Hopper 更合适
3. 如果当前必须继续做压缩实验：
   - 优先尝试 AWQ / GPTQ / AutoRound
   - 它们比 FP8 更符合你当前卡的实际情况

---

## 一句话总结

- llmcompressor 支持 FP8、GPTQ、AWQ、AutoRound、SmoothQuant 等多种方法
- AWQ 是支持的，而且是正式支持
- GGUF 不是 llmcompressor 的主输出格式
- 目标 vLLM 就走 llmcompressor
- 目标 llama.cpp 就走 GGUF 工具链

---

## 你后面可以怎么用这张表

如果下一步你要我继续做，我建议两条路线二选一：

1. 我帮你在 `scripts/` 下再生成一个 AWQ 或 GPTQ 的最小量化脚本模板
2. 我帮你做一份“bf16 -> AWQ / GPTQ / FP8”三路线实验计划表，方便你直接开跑
