# 真机（3060）TRT 化部署调试记录

> 记录时间：2026-09-03
> 服务器：192.168.8.40（用户 dex），容器 build_embodichain（RTX 3060 12GB）
> 环境：TRT 10.11.0.33、torch 2.6.0+cu124、transformers 5.5.4（打桩）
> 分支：smr/sortbook-deploy @ 31744ec29 + 未提交改动
> 目标：把 sortbook-opt 的 TRT engine（vit/llm/cerebellum/refine）挂到真机 policy server，跑通真实推理
> 结构：每阶段 = 现象 → 实验 → 结论

---

## 一、engine 按真机 shape 重建

### 现象
TRT policy 启动后，cerebellum/refine/llm engine 均报 shape 不匹配：
- cb 旧 engine `Set dimensions [1,50,512]. Expected [1,34,512]`
- lang_c `Valid range [1,9,512]..[1,19,512]`（实际 20）
- refine `setInputShape x [1,48,512] Expected [1,32,512]`
- llm `Set dimension [1,317,2048]` 超 profile max 316

### 实验与结论
| 项 | 真机实际 | 旧 engine | 修复 |
|----|---------|----------|------|
| cb x | (1,50,512) | 34 | 5090 重导 x=50 静态 |
| cb lang_c | (1,20,512) | 19 | build profile 固定 20 |
| refine x | (1,48,512) | 32 | 5090 重导 x=48 静态 |
| llm seq | 317 | max 316 | 重 build min300/opt320/max360 |

- **x=50 来源**：`action_horizon=48`（客户端更新 server_config 覆盖 32）+ t(1) + state_history(1)
- **lang_c=20**：4 句真机指令 max 19 token + 1 BOS
- llm engine 重建后 profile：**min=300 / opt=320 / max=360**（覆盖 seq 317）

---

## 二、显存 OOM 与 engine 加载

### 现象
policy 加载模型后（~9.7GB），llm engine（2.7GB）deserialize 失败 `Cuda Runtime (out of memory)`，vit/llm 回退 torch，仅 cerebellum TRT 生效。

### 实验
- 纯 torch policy 显存 9.7GB（模型 + DexSim），剩 ~2.1GB
- 评测链路（infer_trt.sh）4 engine 全装成功——差异疑似 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
- patch start_policy_trt.sh 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 结论
- `expandable_segments` 让 vit engine 装上；llm 2.7GB 仍装不下（差 ~0.6GB）
- **3060 加载不了「完整模型 + 全部 4 engine」**——llm engine 2.7GB 是瓶颈
- 但 llm TRT 在 3060（Ampere）无加速（33ms ≈ torch 32ms）→ **llm 保持 torch 无性能损失**

---

## 三、engine 自由开关（TRT_ENABLE_*）

### 需求
支持自由开关各 engine 加载，用于组合测试与定位。

### 改动
- `trt_infer_shim.py`：新增 `TRT_ENABLE_VIT / TRT_ENABLE_LLM / TRT_ENABLE_CEREBELLUM / TRT_ENABLE_REFINE`（默认全 1），兼容旧 `TRT_SKIP_*`
- `start_policy_trt.sh`：默认全开，env 可覆盖（如 `TRT_ENABLE_CEREBELLUM=0` 关 cb）

### 实测组合
| 组合 | 显存 | Step | 状态 |
|------|------|------|------|
| 纯 torch | 9.7GB | 276ms | ✅ |
| vit+cb TRT + llm torch | 6.9GB | 162-181ms | ✅ |
| 全 TRT（初版）| 11GB | - | ❌ llm OOM |
| vit+llm TRT + cb torch | - | 189ms | ✅ |
| 全 TRT（cb 修复后）| - | ~194ms | ✅ |

### 控制方式（TRT_ENABLE_* 环境变量）

| 环境变量 | 作用 | 默认 |
|---------|------|------|
| `TRT_ENABLE_VIT` | 是否装 ViT engine | 1（开）|
| `TRT_ENABLE_LLM` | 是否装 LLM engine | 1（开）|
| `TRT_ENABLE_CEREBELLUM` | 是否装 cerebellum engine | 1（开）|
| `TRT_ENABLE_REFINE` | 是否装 refine engine | 1（开）|

**用法**（启动时指定，关闭的模块回退 torch，无 hook 注入）：
```bash
# 全 TRT（默认）
bash scripts/sortbook-opt/start_policy_trt.sh --background

# 关 cb/refine（回退 torch）
TRT_ENABLE_CEREBELLUM=0 TRT_ENABLE_REFINE=0 bash ... --background

# 只开 vit+llm
TRT_ENABLE_CEREBELLUM=0 TRT_ENABLE_REFINE=0 bash ... --background

# 只开 cerebellum
TRT_ENABLE_VIT=0 TRT_ENABLE_LLM=0 bash ... --background
```

**控制逻辑**（`trt_infer_shim.py`）：
- `TRT_ENABLE_VIT` / `TRT_ENABLE_LLM` → `_install_vlm_hooks`（装 ViT/LLM）
- `TRT_ENABLE_CEREBELLUM` → `install_cerebellum_trt_hook`（装 cb）
- `TRT_ENABLE_REFINE` → `install_refine_trt_hook`（装 refine）
- 兼容旧 `TRT_SKIP_*`（反向语义）

---

## 四、cb TRT 输出 NaN（根因与修复）

> 详细消融见 cerebellum_trt_plan.md §9.5。此处记录真机现场。

### 现象
全 TRT 后客户端报 **`SVD did not converge`**（FK/IK 求解崩）。server 日志无错误，但 cb 输出全 NaN（model_output NaN=3744 / intermediate NaN=24576）。

### 定位过程（逐步隔离）
1. cb engine_in 诊断：**输入 x NaN=0（正常）**，但**输出全 NaN**
2. dexdp 诊断：第一次 predict_noise 正常（_actions NaN=0），**第二次起全 NaN**（第一步 cb 输出 NaN → 污染采样循环）
3. 独立测 engine（随机 + 放大输入）：**输出正常 NaN=0**——engine 本身没问题
4. **结论：cb engine（fp16_rmsnorm norm-only）对「真实输入」输出 NaN，独立测正常**

### 根因
- `fp16_rmsnorm_fp32`（norm fp32 + attn fp16）在**真实输入（x max~8、img_c max~138）**下，**attn 层 fp16 溢出 → NaN**
- 离线 verify 用 32/33 假数据（pad），未触发真实值域 → 掩盖问题
- **「只需 norm fp32」的消融结论在随机/小数据下成立，真实值域下不成立**

### 修复
| 方案 | engine | NaN | Step | 说明 |
|------|--------|-----|------|------|
| 全 fp32 | cerebellum_core_fp32.plan | 0 | 172-218ms | IO fp32，需 TrtEngine 输入 dtype 自适应 |
| **fp16 + attn/norm fp32** | fp16_rmsnorm_fp32.plan（779 层）| 0 | 194-210ms | 保留 IO fp16，attn/norm 强制 fp32 |

**采用 fp16 + attn/norm fp32（779 层）**：engine 小（100MB）+ IO fp16 hook 不改 + Step ~194ms。

配套改动：
- `build_cerebellum_engine.py` extract 筛选 → `ATTR_KEYS`（norm/attn/mha/q_/k_/v_/out_proj/qkv/matmul/softmax），443 层 → 779 层
- `trt_hooks.py TrtEngine.__call__` 输入按 engine IO dtype 自适应（fp32 → .float()，fp16 → .half()）
- `config.py CB_ENGINE` 指向 fp16+attn+norm plan

---

## 五、离线验证 vs 真机推理的 shape 差异

### 现象
离线 verify（verify_refine_real.py）TRT vs torch max_diff 需在 reference 上跑，但报维度不匹配（engine 48 vs reference 32）。

### 结论（关键）
**离线验证链路和真机推理链路的 cb/refine 输入 shape 不一样**：
| 输入 | 离线 reference（cerebellum_ref.safetensors）| 真机推理 |
|------|------------------------------------------|---------|
| cb x (state_action_traj) | (1,33,512) | (1,50,512) |
| lang_c | (1,19,512) | (1,20,512) |
| refine full_x | (1,32,512) | (1,48,512) |

- 离线 reference 是**旧录制数据**（action_horizon=32、lang 19 token）
- 真机现在 action_horizon=48、lang 20 token
- **离线 verify 的精度数值（cb 0.0031 / refine 0.0297）是 32/33 shape 下测得，不代表 48/50 真实输入**——refine 真机 x=48 时 max_diff 0.0297（合格），但 cb 的 32/33 假数据验证掩盖了真实值域的 fp16 NaN
- 验证脚本已支持 reference pad 到 48（`verify_refine_real.py`）

### 来源确认（2026-09-04，脚本 verify_ref_offline.py）
**这些 safetensors 是 5090 上传的、适配离线评测链路的 reference**（不是真机生成）：
- 证据：文件属主 1000:1000（容器内生成是 root）+ 时间 09-03 12:25/13:14（传输时间）；5090 上已无同款
- 由**离线评测链路**（test_eval_models...v2.py 跑 capture 截取）生成，从 5090 传输到真机

**确认脚本**：`scripts/sortbook-opt/capture_verify/verify_ref_offline.py`
```
对比基准：离线 action_horizon=32 / lang=19 / seq=316 / img=288
[vit]      ✅ ds0_fp32 (288,2048) + 2 图
[llm]      ✅ input_ids/ds_full seq=316
[cerebellum] ✅ x=33（32 action+1 state）/ refine=32 / lang=19
[gihs]     ✅ final_hidden seq=316 + 288 img token
总体结论：✅ 全部适配离线评测
```
- **4 个 reference 全部适配离线评测链路**（action_horizon=32 时代）
- **非真机（action_horizon=48 / seq=317）**——这是离线 vs 真机不对齐的根本原因

---

## 六、refine 精度验证

真机 x=48 下，refine TRT engine（refine_core_fp16.plan）：
```
torch baseline:  orig out (1,48,128)
wrapper vs orig: max_diff=0.00000000   ← onnx wrapper 完全一致
TRT engine vs orig: max_diff=0.029669, mean_diff=0.000578   ← 合格（≤0.05）
engine avg 1.260 ms/iter
```
- **refine 精度合格**（max_diff 0.0297 ≤ 0.05），且 refine 是纯 fp16 engine（不需要 fp32 层）

---

## 七、最终状态（真机全 TRT 跑通）

| 模块 | engine | 精度/状态 |
|------|--------|----------|
| vit | fp16（vit_qwen3vl_384_fp16.plan）| ✅ 跑通 |
| llm | fp16+fp32_layers（max-seq 360）| ✅ seq 317 覆盖 |
| cerebellum | **fp16 + attn/norm fp32（779 层）** | ✅ NaN=0 |
| refine | fp16（refine_core_fp16.plan）| ✅ max_diff 0.0297，1.26ms |
| **整体** | 全 TRT | ✅ Step ~194ms，多帧稳定 |

### 关键改动文件（真机）
- `config.py`：CB_ENGINE 指向 fp16+attn+norm plan
- `build_cerebellum_engine.py`：extract 筛选含 attn（ATTR_KEYS）
- `trt_infer_shim.py`：TRT_ENABLE_* 自由开关
- `trt_hooks.py`：TrtEngine 输入 dtype 自适应 + deserialize 前 empty_cache
- `start_policy_trt.sh`：PYTORCH_CUDA_ALLOC_CONF + TRT_ENABLE 默认
- `verify_refine_real.py`：reference pad 到 48

### 启动方式
```bash
bash scripts/sortbook-opt/start_policy_trt.sh --background
# 单个 engine 开关：TRT_ENABLE_CEREBELLUM=0 TRT_ENABLE_REFINE=0 bash ... --background
```

---

## 八、日志清理与耗时统计（2026-09-03 收尾）

### 1. 删除 debug 打印
定位 NaN 时加的诊断打印已清理：
- `cerebellumWrapper.py`：删除 `[cerebellum-trt]` engine_in shapes / 值诊断（NaN/mean/max/inf）/ 输出统计
- `dexdp.py`：删除 `[dexdp-diag]` state_cond/action_cond/_actions 诊断

清理后 policy_server_trt.log 只保留正常日志（Step 时间 / Actions ready / hook 装载）。

### 2. 平均耗时丢弃前 5 帧
**原逻辑**（问题）：`self.step_time.append(finish_predict)` + `np.mean()`——**平均包含所有帧**，首次帧（模型加载 + 4 engine 装载 15-26s）严重拉高平均值。

**改后**（`policy_server_pure_inference.py`）：
```python
self.step_count = 0  # 初始化
...
self.step_count += 1
if self.step_count >= 5:  # 丢弃前 5 帧（初始加载/预热慢帧），从第 6 帧起统计
    self.step_time.append(finish_predict)
avg_time = np.mean(self.step_time) if self.step_time else 0.0
```
- 前 5 帧（预热）不纳入平均，从第 6 帧起统计
- `step_time` 仍是 deque(maxlen=50)（滚动窗口）

---

## 九、install 日志 + 全优化验证（2026-09-03）

### 1. 给所有 install 加安装日志
定位优化是否生效，给每个 TRT engine install + 优化 install 加统一前缀日志：

**TRT engine**（`trt_hooks.py`，`[trt-hook]` 前缀）：
- `install_vit_trt_hook` → `[trt-hook] ViT installed: xxx.plan`
- `install_llm_trt_hook` → `[trt-hook] LLM installed: xxx.plan`
- `install_trt_hooks` → `[trt-hook] ViT+LLM installed`
- `install_cerebellum_trt_hook` / `install_refine_trt_hook`：已有 `[cerebellum-hook] installed` / `[refine-hook] installed`

**优化**（`[optim]` 前缀）：
- `gpu_get_image_hs.py install_get_image_hs_hook` → `[optim] get_image_hs installed（预计算 index 优化）`
- `prepare_inputs_gpu_hook.py install_prepare_inputs_gpu_hook` → `[optim] prepare_inputs GPU 化 installed（image_processor 代理 + prepare_inputs 替换）`

### 2. 全部优化生效验证（真实 policy 日志）
重启 policy 触发推理后，policy_server_trt.log 完整 install 序列：
```
[trt-hook] ViT installed: vit_qwen3vl_384_fp16.plan
[trt-hook] LLM installed: llm_add_ds_fp16_fp32_layers.plan
[trt-hook] ViT+LLM installed: vit... / llm...
[trt-shim] ViT+LLM TRT engines 已装
[optim] prepare_inputs GPU 化 installed（image_processor 代理 + prepare_inputs 替换）
[optim] get_image_hs installed（预计算 index 优化）
[cerebellum-hook] installed: cerebellum_core_fp16_rmsnorm_fp32.plan
[trt-shim] cerebellum TRT engine 已装
[refine-hook] installed: refine_core_fp16.plan
[trt-shim] refine TRT engine 已装
```

**全部 6 项确认生效**：
| 项 | 日志证据 |
|----|---------|
| ViT TRT | `[trt-hook] ViT installed` |
| LLM TRT | `[trt-hook] LLM installed` |
| cerebellum TRT（779 层 NaN 修复版，20:37 build）| `[cerebellum-hook] installed` |
| refine TRT | `[refine-hook] installed` |
| prepare_inputs GPU 化 | `[optim] prepare_inputs GPU 化 installed` |
| get_image_hs 预计算 | `[optim] get_image_hs installed` |

**推理性能**（丢弃前 5 帧后）：
- Step 172-196ms（多帧稳定）
- 平均推理时间 182-184ms
- 客户端正常（无报错、动作正常）
