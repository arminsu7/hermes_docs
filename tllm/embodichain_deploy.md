# Embodichain 部署移植记录（外部文件夹 → deploy 分支）

> 记录时间：2026-09-02
> 服务器：192.168.11.51（用户 oem）
> 容器：whh_vla_trt2（conda py310）
> 工作目录：/root/workspace/sumingrui/embodichain
> （2026-09-02 由 deploy 重命名而来；重命名不影响 git 仓库，仍在 deploy 分支）
> 目标：把**另外的文件夹**中的内容移植/移动到 **deploy 分支**上，逐步记录过程
> （注：最初误记为「→ master」，用户已纠正为「→ deploy」）

---

## 一、当前状态

- [x] 步骤 1：新建本 deploy 文档
- [x] 步骤 2：登录服务器，确认在 deploy 分支且代码最新
- [x] 步骤 3：基于 deploy 创建 `smr/sortbook-deploy` 分支并推送远端
- [x] 步骤 4：精简 `scripts/sortbook-opt/infer_realdata.sh`（参考原脚本，保留 cfg_path/model_path/test_data，默认 torch 后端跑通 qwen-vla）
- [x] 步骤 5：实机验证脚本默认跑通 qwen-vla（torch）✅
- [x] 步骤 6：py310 环境重新指向当前目录（pip install -e --no-deps）
- [x] 步骤 7：修复 HF hub 兼容性 bug（utils.py 加 = None 默认值）并验证真·当前分支代码跑通 ✅
- [x] 步骤 8：新建 `test_eval_models_by_realdata_profile_sortbooks_v2.py`，完美复刻参考链路打桩，脚本入口已切换 ✅
- [ ] 步骤 9：commit / push（待用户确认）

---

## 二、执行记录

### 2.1 环境确认（2026-09-02）

**路径说明**：工作目录 `/root/workspace/sumingrui/embodichain` 原名为 `/root/workspace/sumingrui/deploy`，
2026-09-02 由用户重命名（deploy 目录本身就是 embodiments 仓库根，含 .git、dexechain、setup.py 等，
重命名不影响 git 仓库）。`/root/workspace/sumingrui/` 下当前仅此一个 embodiments 目录。

**git 状态**（`git fetch` 后）：

| 项 | 值 |
|----|----|
| 当前分支 | `deploy`（已跟踪 `origin/deploy`，无 ahead/behind，工作区干净） |
| 本地 HEAD | `106adc95b18a6a2d234542341db3ea78250804d7` |
| origin/deploy | `106adc95b18a6a2d234542341db3ea78250804d7`（完全同步） |
| 最新 commit | `106adc95b ENH: add runtime module & onnx trt export` |
| remote | `http://sumingrui@dexforce.top:...@192.168.3.16/Engine/embodichain.git` |
| 其他本地分支 | `master`（待对比） |

**仓库顶层结构**：`.gitlab-ci.yml` / `README(.zh).md` / `VERSION` / `configs` / `dexechain` /
`docker_run.sh` / `docs` / `examples` / `scripts` / `setup.py` / `test_configs` / `tests` /
`tools` / `web_server`

---

## 三、移植步骤

### 3.1 deploy vs master 差异（2026-09-02 分析）

- 分支关系：`merge-base = d9f0c0c1b`，`origin/master..deploy` 仅 **1 个 commit**
  （`106adc95b ENH: add runtime module & onnx trt export`）
- 差异规模：**40 文件，+3438 / -163**

**新增核心能力（runtime 模块）**：

| 文件 | 作用 |
|------|------|
| `dexechain/agents/dexforce_vla/runtime/*`（11 文件） | VLA 推理运行时：`vla_inference.py` 主入口、`factory.py` 后端工厂、`onnx_backend.py` / `trt_backend.py` / `torch_backend.py` 三后端、`config.py` / `base.py` / `export_modules.py` / `utils.py` |
| `scripts/build_vla_trt.py` / `.sh` | TRT engine 构建（421 行） |
| `scripts/export_vla_onnx_trt.py` / `.sh` | ONNX 导出 |
| `scripts/infer_vla_exported.py` / `.sh` | 用导出产物推理 |
| `scripts/compare_vla_backends.sh` / `compare_vla_predictions.py` | 多后端输出对比 |
| `scripts/profile_vla_trt.sh` / `inspect_vla_artifact_dtypes.py` | 性能/精度分析 |
| `dexechain/agents/dexforce_vla/models/policy/inference_core.py` | 推理核心（+54） |
| `dexechain/agents/dexforce_vla/train/build_model.py` | 构建模型调整（+22） |
| `dexechain/utils/profile.py` | profiling 工具扩展（+201） |
| `tests/models/test_vla_runtime_config.py` | runtime 配置单测（+109） |
| 若干 `*_by_realdata_profile.py` / `*_profile` | 真实数据 profile 相关 |

**改动较大的既有文件**：`dexforcevla_runner.py`（+307，接入 runtime）、
`dexdp.py`（+114/-）、`dexforcevla_sim_by_realdata_profile.py`（+455）

### 3.2 移植候选方式（待定）

- [ ] 方式 A：直接 cherry-pick `106adc95b` 到 master（推荐，若 master 无冲突）
- [ ] 方式 B：deploy → master 开 MR 合入（遵循用户 git 工作流）
- [ ] 方式 C：手动挑选文件移植

### 3.3 精简 `scripts/sortbook-opt/infer_realdata.sh`（2026-09-02）

**目标**：参考 `/root/workspace/embodichain/scripts/infer_realdata.sh`（原始 torch 推理脚本），
精简目标脚本；保留 `cfg_path/model_path/test_data`，只兼容 sortbook qwen-vla 单任务，
默认 `bash scripts/sortbook-opt/infer_realdata.sh` 直接跑通。

**参数核对结论**（基于 `dexforcevla_sim_by_realdata_profile.py` + `build_model` + `runner` 代码证据）：

| 参数 | 作用 | 处理 |
|------|------|------|
| `cfg_path` / `model_path` / `test_data` | 任务核心三路径 | **保留**（用户要求） |
| `MODE` / `BACKEND` / `DEVICE` / `PRECISION` / `SEED` | 进 `runtime_overrides` → `RuntimeConfig`，驱动 torch/onnx/trt 后端 | **保留** |
| `ARTIFACT_DIR` | TRT/ONNX 导出产物目录，后端切换必需 | **保留**（默认空，torch 不需要） |
| `SAMPLE_NUM` / `WARMUP` / `PROFILE_ITERS` / `PROFILE_*` | 采样与 profiling | **保留**（原脚本同款） |
| `NUM_BATCHES` | 数据批数，python 侧有默认 | **删除** |
| `SAVE_PATH_DIR` | sortbook 多后端对比目录 | **删除**（python 侧默认 tmp_record） |
| `PREDICTION_OUTPUT` + `PREDICTION_FLAGS` | 预测保存（可选增强） | **删除** |

**关键修正**：原默认 `BACKEND=trt` 且 `ARTIFACT_DIR` 指向 `/root/workspace/embodichain/exported_models/bloody_hell_v1_4`（该目录不存在）→ 默认跑不通。
改为 **`BACKEND=torch` 默认**（无需 artifact 直接从 model_path 加载权重），需要时 `BACKEND=trt ARTIFACT_DIR=...` 覆盖。

**改动方式**：本地写好脚本 → base64 经 ssh+docker exec 管道推送 → `bash -n` 语法检查通过。

### 3.4 实机验证（2026-09-02）✅

**默认 torch 后端完整跑通 sortbook qwen-vla 任务**：

```
07:18:30 [DexEChain INFO]: [sync_full_metrics][episode=0] | right_armqpos    | 154 | mae 0.051959 | mse 0.004952 | rmse 0.070372
07:18:30 [DexEChain INFO]: [sync_full_metrics][episode=0] | right_eefgripper | 154 | mae 0.021363 | mse 0.012903 | rmse 0.113589
07:18:31 [DexEChain INFO]: Inference test completed!
```

- 命令：`cd /root/workspace/sumingrui/embodichain && bash scripts/sortbook-opt/infer_realdata.sh`
- 产物：`logs/infer_torch_20260902_071802.log` + `tmp_record/record_sync_full_0_*_gt_vs_pred.png`
- 模型加载正常（image_hidden_state [1,288,2048]），评测指标正常输出

**踩坑记录（根因分析）**：

| 坑 | 现象 | 根因 | 修复 |
|----|------|------|------|
| cfg_path 坏路径 | `FileNotFoundError: .../gy_...-checkpoint-24000/config.json` | 脚本 cfg_path 是相对路径指向不存在的 `gy_.../config.json` | 改为真实 yaml 绝对路径 `qwen3vl-sortbookv26-...-data-2cams.yaml`（已确认与 checkpoint 的 dexforcevla_config.json 完全对齐） |
| dexechain 双份代码 | 后台 traceback 曾指向 `/root/workspace/embodichain` | editable 安装 egg-link 指向参考目录；cwd 下 import 优先用本地副本 | 在仓库根 cwd 执行即可用当前分支代码（已验证） |

**git 状态**：`smr/sortbook-deploy` 分支，`scripts/sortbook-opt/` 未跟踪（待 commit）。

### 3.5 环境重新指向 + HF 兼容性修复（2026-09-02）✅

**背景**：用户要求把 py310 环境的 dexechain 重新指向当前目录，让所有脚本都用当前分支代码。

**3.5.1 重新指向 editable 安装**

- 原状态：`dexechain` editable 安装指向参考目录 `/root/workspace/embodichain`（版本 0.1.4，分支 smr/vla-zhuashu）
- 操作：`pip install -e /root/workspace/sumingrui/embodichain --no-deps --no-build-isolation`（避开 setup.py 的内网 git+ 依赖和 torchsdf 编译风险）
- 结果：egg-link 指向 `/root/workspace/sumingrui/embodichain`，Version 0.1.7，任意 cwd import 都用当前分支代码 ✅

**3.5.2 发现 HF 兼容性 bug 并修复**

重新指向后跑脚本报错：
```
TypeError: CompatiblePyTorchModelHubMixin._from_pretrained() missing 2 required
keyword-only arguments: 'proxies' and 'resume_download'
```

**根因分析**（git blame + 双目录对比）：
- 当前分支 `utils.py` 的 `CompatiblePyTorchModelHubMixin._from_pretrained` 里
  `proxies`/`resume_download` **无默认值**（必填）——来自 commit `1cb23030b1`（2025-10-30）
- 环境 HF hub 1.28.0 已废弃这两个参数，调用 `_from_pretrained` 时不传 → TypeError
- 参考目录 `smr/vla-zhuashu` 有**未提交本地修复**（加了 `= None` 默认值）→ 所以之前能跑
- 之前第一次"跑通"其实是蹭了参考目录代码（egg-link 当时还指向它），并非当前分支代码真的能跑

**修复**：给 `dexechain/agents/dexforce_vla/models/utils.py` 137-138 行加 `= None` 默认值：
```python
proxies: Optional[Dict] = None,
resume_download: Optional[bool] = None,
```
editable 安装是文件系统实时链接，改完立即生效，无需重装。

**3.5.3 修复后实机验证** ✅（真·当前分支代码）

```
07:27:25 [DexEChain INFO]: VLM forward pass time: 0.04 seconds
07:27:25 [DexEChain INFO]: Extracted image_hidden_state mean: 0.33984375, shape: torch.Size([1, 288, 2048])
07:27:25 [DexEChain INFO]: [sync_full_metrics][episode=0] | right_armqpos    | 154 | mae 0.045470 | mse 0.003990 | rmse 0.063170
07:27:25 [DexEChain INFO]: [sync_full_metrics][episode=0] | right_eefgripper | 154 | mae 0.012392 | mse 0.003060 | rmse 0.055315
07:27:26 [DexEChain INFO]: Inference test completed!
```

- traceback/日志路径全部指向 `/root/workspace/sumingrui/embodichain/`（当前分支）✅
- 修复后指标更优（armqpos mae 0.046 vs 之前的 0.052；eefgripper 0.012 vs 0.021）
- 副产品：`dexechain.toolkits.graspkit` import 报 `No module named 'embodichain.toolkits'` —— 参考/当前两目录**原本就一样**，不影响 VLA 主流程，可忽略

### 3.6 新建 test_profile 文件：完美复刻参考链路打桩（2026-09-02）✅

**背景**：待对齐链路跑 profile 时，最后的耗时报告缺少参考链路的**细粒度 VLM 计时**
（vlm.prepare_inputs/forward/stack_mean/get_image_hs/get_text_hs + Qwen3VLModel 内部锚点
vlm_model.*/vlm_cgm.*）。参考链路能打全这些桩，待对齐链路打不全。

**根因**（双链路对比）：
- 参考链路 `qwen2_5_vl.py:forward_for_dexforcevla` 有 5 处 `self.profile_speed["vlm.*"]` 记录
- 待对齐链路 `qwen2_5_vl.py:forward_for_dexforcevla` **没有**这些记录（只有 2 个粗粒度 log）
- 参考链路 runner `prepare_images`（706 行）采集 `images_encoder.profile_speed` 的 vlm.* 键
- 待对齐链路 runner `prepare_images` **没有**采集逻辑

**方案**：新建 `tests/models/test_eval_models_by_realdata_profile_sortbooks_v2.py`，
monkey-patch 三处（不改原始 qwen2_5_vl.py / runner）：
1. patch `Qwen25VLEncoder.prepare_inputs` → `_timed_prepare_inputs`
   （复刻参考的 5 个细粒度桩：vlm.prepare_to_pil / prepare_template /
   prepare_vision_info / prepare_processor / prepare_to_gpu）
2. patch `Qwen25VLEncoder.forward_for_dexforcevla` → `_timed_forward_for_dexforcevla`
   （复刻参考的 5 处 vlm.* 计时 + 采集 vlm_model.*/vlm_cgm.* 锚点）
3. patch `DexForceVLA.prepare_images` → `_prepare_images_with_vlm_collect`
   （返回前把 encoder profile_speed 的 vlm.* 键采集进 runner.profile_speed）

**踩坑**：
- `run_evaluation` 需要 `argparse.Namespace`（传 dict 会 `AttributeError: timestep_id`）→ 用 `argparse.Namespace(**vars(args))`
- `self.encoders` 是 `nn.ModuleDict`（无 `.get()`）→ 用 `in` + 下标访问
- `DexImage`/`Lang` 在 `dexechain.data.enum`（`Image as DexImage`），`log_info` 在 `dexechain.utils.logger`
- `to_pil_image` 需从 `torchvision.transforms.functional` import

**验证结果**（`logs/infer_torch_20260902_080635.log`）✅ 完整跑通，vlm.* 打桩与参考链路完全对齐：

```
vlm.prepare_to_pil       0.0035   ← prepare_inputs 内部 5 桩
vlm.prepare_template     0.0002
vlm.prepare_vision_info  0.0002
vlm.prepare_processor    0.0061
vlm.prepare_to_gpu       0.0023
vlm.prepare_inputs       0.0128   ← forward 层 5 桩
vlm.forward              0.0452
vlm_cgm.model            0.0416   ← Qwen3VLModel 内部锚点
vlm_cgm.lm_head          2.8e-05
vlm_model.embed          9.4e-05
vlm_model.get_image_features 0.0187
vlm_model.mask_inject    0.0004
vlm_model.compute_position_ids 0.0012
vlm_model.language_model 0.0212
vlm.stack_mean           0.0001
vlm.get_image_hs         0.0008
vlm.get_text_hs          0.0004
```

**脚本入口已切换**：`scripts/sortbook-opt/infer_realdata.sh` 第 59 行
`test_eval_models_by_realdata_profile_sortbooks.py` → `_sortbooks_v2.py`

### 3.7 报告层对齐：Speed Report 表格显示 vlm.*（2026-09-02）✅

**问题**：即使 encoder 打桩成功，Speed Report **表格**里仍看不到 vlm.* 行（只在 profile dict 里）。

**根因**（对比两份 `_profile.py`）：
- 参考链路 `_print_brain_cere` 表格有完整 vlm.* 行（818-834 行），用下划线键名（`vlm_prepare_inputs_s` 等）
- 参考链路 `_run_one_sample` 有 vlm.* 键映射（914-923 行）
- 待对齐链路 `_print_brain_cere` **无 vlm.* 行**；`_run_one_sample` **无 vlm.* 映射**
- 待对齐 runner 还缺 from_batch/prepare_images 细分桩（from_batch.dtype_convert 等全为 0，参考有）

**修复**（新文件继续 monkey-patch，不改 `_profile.py`）：
1. patch `_run_one_sample` → `_run_one_sample_with_vlm`：调用原版后，把 17 个 vlm.* 键映射补进 speed_stats
2. patch `_print_brain_cere` → `_print_brain_cere_with_vlm`：复刻参考表格行（vlm.prepare_inputs → to_pil/template/vision_info/processor/to_gpu → vlm.forward → vlm_cgm.model → vlm.embed/vit/mask_inject/mrope/llm → vlm_cgm.lm_head → stack_mean/get_image_hs/get_text_hs）

**验证结果**（`logs/infer_torch_20260902_081502.log`）✅ Speed Report 表格完整显示 vlm.*：

```
prepare_images (DINO): 51.16 / 37.60 / 63.78 ms
    ├ vlm.prepare_inputs: 11.48 / 7.39 / 16.99 ms
    │   ├ to_pil:      2.92 / 1.74 / 3.53 ms
    │   ├ template:    0.17 / 0.12 / 0.77 ms
    │   ├ vision_info: 0.18 / 0.09 / 0.40 ms
    │   ├ processor:   6.40 / 4.36 / 12.10 ms
    │   └ to_gpu:      0.87 / 0.62 / 1.51 ms
    ├ vlm.forward: 37.66 / 27.27 / 48.88 ms
    │   ├ vlm_cgm.model: 37.37 / 27.12 / 48.60 ms
    │   │   ├ vlm.embed:     0.10 / 0.06 / 0.38 ms
    │   │   ├ vlm.vit:     16.19 / 12.20 / 20.20 ms
    │   │   ├ vlm.mask_inject: 0.36 / 0.23 / 0.75 ms
    │   │   ├ vlm.mrope:     0.91 / 0.55 / 1.87 ms
    │   │   └ vlm.llm:     19.75 / 13.49 / 27.61 ms
    │   └ vlm_cgm.lm_head: 0.04 / 0.02 / 0.10 ms
    ├ vlm.stack_mean: 0.11 / 0.06 / 0.28 ms
    ├ vlm.get_image_hs: 0.88 / 0.55 / 1.46 ms
    └ vlm.get_text_hs: 0.43 / 0.24 / 0.77 ms
```

与参考链路表格层级一一对应（仅分支字符 `│` 位置因 label 宽度有轻微偏移，数据正确）。

**已知差距**：待对齐 runner 仍缺 from_batch/prepare_images 细分桩（from_batch.dtype_convert、prepare_images.preprocess/to_device/encoder/postprocess、brain entry/exit overhead、gap 等），这些表格键会显示 0。如需完全对齐需补 runner 打桩（后续可做）。

### 3.8 cerebellum 耗时差异根因：配置文件（2026-09-02）✅

**问题**：参考链路 cerebellum ~30ms，待对齐链路 ~70ms。

**根因**：**两条链路跑的是不同任务/模型，配置文件（yaml）差异导致 cerebellum 计算量不同**，不是代码 bug。

**配置对比**：

| 参数 | 参考 (grasp_book_qwen3vl2b) | 待对齐 (sortbookv26) | 影响 |
|------|------|------|------|
| `model.hidden_size` | 384 | 512 | 扩散网络 hidden 大 1.33 倍 |
| `cerebellum.depth` | 8 | 12 | 扩散 transformer 深 1.5 倍 |
| `cerebellum.num_heads` | 4 | 8 | 注意力头翻倍 |
| `cerebellum.use_refiner` | **False** | **True** | **待对齐额外跑 refiner 前向** |
| `noise_scheduler.num_inference_timesteps` | 10 | 10 | 相同 |
| `loss` refined 相关权重 | 0（refined 分支全 0） | 非 0 | 待对齐模型带 refiner 训练 |

**结论**：待对齐 sortbook 模型（hidden=512, depth=12, heads=8, refiner=on）比参考 grasp_book
模型（hidden=384, depth=8, heads=4, refiner=off）**计算量大得多**：
1. `use_refiner=True` → 扩散采样后多一次 refiner transformer 前向（参考完全跳过）
2. hidden 512 + depth 12 + heads 8 → 10 步扩散循环每步都更重

叠加起来 cerebellum 从 ~30ms 涨到 ~70ms（约 2 倍）合理。**代码路径（runner 打桩、dexdp.inference 算法）两者等价**，差异纯在配置/模型规模。

### 3.9 报告表格布局修复：宽松对齐 + 树形字符（2026-09-02）✅

**问题**：Speed Report 表格太紧凑，深层 vlm.* 行耗时数字错位（列宽随深度递减到 10，长 label 挤进数字列）。

**根因**：待对齐 `_profile.py` 的 `_srow` 用动态列宽 `w = max(34 - 4*depth, 10)`，
深度大时 label 宽度只有 10，`vlm.mask_inject` 等长 label 放不下导致数字错位。

**修复**（新文件 monkey-patch `_srow`/`_ssep`）：
- `_srow_wide`：固定 `_COL_LABEL=60` + `_COL_AVG/MIN/MAX=12`，数字列右对齐
- `_ssep_wide`：`-` 分隔线按新列宽
- 浅缩进 `"  " + "  "*depth`（树形字符 `│/├/└` 由 label 自带，不再双重缩进）

**验证**（构造假数据直接测 `_print_brain_cere`）：数字列完全右对齐，`│` 树形字符连贯，
vlm 子树（depth 5-7）在 prepare_images.encoder 下正确缩进。真实运行日志
`logs/infer_torch_20260902_082517.log` 同样对齐。

### 3.10 双容器对齐验证：打桩对齐 + 配置差异确认 + 耗时对比（2026-09-02）✅

**背景**：参考标准链路移到 whh_vla_trt 容器（`cd /root/workspace/embodichain && bash scripts/infer_realdata.sh`），
待对齐链路仍在 whh_vla_trt2（`cd /root/workspace/sumingrui/embodichain && bash scripts/sortbook-opt/infer_realdata.sh`）。
两容器独立环境，可同跑对比。

**3.10.1 打桩位置对齐（改待对齐 runner）**

待对齐 runner 原缺打桩（相对参考）：
- `prepare_images.preprocess / to_device / encoder / postprocess` 4 段
- `from_batch.dtype_convert / augment` + `brain_infer.prepare_*` → 改名 `from_batch.prepare_*`
- `brain_infer.entry_overhead / exit_overhead`（含 sub_sum 键）
- `cerebellum_infer.forward_action / refine`（dexdp.py 加 cerebellum_profile 计时 + runner 采集）

改后对比（`grep profile_push/pop`）：**待对齐打桩与参考完全一致**，唯一多 `predict_action.total`
（待对齐最外层总耗时桩，参考通过 brain+cerebellum 相加）。已备份 runner `.bak_sortbook` / dexdp `.bak_sortbook`。

**3.10.2 配置文件差异确认**

| 参数 | 参考 grasp_book | 待对齐 sortbook | 影响 |
|------|------|------|------|
| `hidden_size` | 384 | 512 | 扩散网络更大 |
| `cerebellum.depth` | 8 | 12 | 更深 |
| `num_heads` | 4 | 8 | 翻倍 |
| `use_refiner` | False | True | 待对齐多 refiner |
| `num_inference_timesteps` | 10 | 10 | **相同** |
| `masking_gripper` | True | False | gripper 处理不同 |

**forward_action 循环次数验证**：两条链路 cfg 都是 `num_inference_timesteps: 10`，
DDPMScheduler.set_timesteps(10) → **实际循环 10 次**（非记忆中的 5 次）。
从日志数 `cerebellum_infer.forward_action` 出现次数：**两链路都是 102**（100 iters + warmup），
即每次 inference 1 次 forward_action 调用、内部 10 步扩散循环。

**3.10.3 耗时对比（同跑，100 iters avg）**

| 环节 | 参考 (grasp_book) | 待对齐 (sortbook) | 比值 |
|------|------|------|------|
| `cerebellum_infer.inference` | 33.10ms | 49.92ms（dict 最后次） / 表格 avg ~70ms | ~1.5-2x |
| `cerebellum_infer.forward_action` | 27.28ms | 45.58ms | ~1.7x |
| `cerebellum_infer.refine` | 4.70ms | 4.10ms | ~1x |
| `predict_action.cerebellum_infer` | 34.09ms | 53.01ms | ~1.6x |
| `prepare_images.encoder`（VLM） | 74.73ms | 45.91ms | 待对齐反而快 |
| `vlm.forward` | 49.95ms | 37.99ms | 待对齐快 |

**结论**：
1. cerebellum 耗时差（~30ms vs ~70ms）**由配置差异解释**：待对齐 hidden=512/depth=12/heads=8
   → forward_action 每步扩散 transformer 更重（~1.7x），非代码 bug。
2. refine 两链路几乎相同（refiner 规模接近）。
3. VLM 部分待对齐反而更快（vlm.forward 38 vs 50ms），因 grasp_book 参考模型不同/任务图像不同。
4. 打桩已对齐，后续可在此双容器环境持续对比调优。

### 3.11 VLM 部分耗时差异分析（2026-09-02）✅

**VLM 模型/输入/环境完全一致**（已逐项确认）：
- VLM encoder 配置：两链路都是 `Qwen3VLEncoder, num_patches=144, image_size=384, token_dim=2048, dtype=bf16`
- 实测 input_ids：**两链路都是 `seq_len=316, image_tokens=288`**（DBG 打点）
- 环境：torch 2.7.0+cu128 / transformers 5.5.4 完全一致
- 两容器共享同一块 RTX 5090（GPU-be500d6a），但串行跑无并发竞争

**VLM 细分耗时对比**（100 iters avg）：

| 环节 | 参考 (grasp_book) | 待对齐 (sortbook) | 结论 |
|------|------|------|------|
| vlm.prepare_inputs | 10.58ms | 11.68ms | 接近 |
| vlm.forward | 49.95ms | 37.77ms | 待对齐快 12ms |
| vlm_cgm.model / vlm_model.* | 全 0 | 有值（vit 16 + llm 20） | 参考无锚点 |
| vlm.get_image_hs | 13.22ms | 0.88ms | **参考 profile 假高** |
| vlm.get_text_hs | 0.50ms | 0.40ms | 接近 |
| prepare_images.encoder 总 | 74.73ms | 42.22ms | 待对齐快 32ms |

**get_image_hs 差异根因（实测证实）**：参考 profile 显示 13.22ms vs 待对齐 0.88ms，
但用 `torch.cuda.Event` 精确测 GPU 耗时：**参考 0.09-0.46ms，待对齐 0.09-0.44ms，几乎一样**。
→ 参考的 13ms 是 **`time.time()` CPU 异步计时伪影**（前面 vlm.forward 大 kernel 未 flush，
CPU 侧量到排队等待墙钟时间），**非真实性能差异**。

**vlm.forward 差异**：待对齐 37.77ms 且内部锚点自洽（vlm_cgm.model 37.49 ≈ vlm.vit 16 + vlm.llm 20）。
参考 49.95ms 但**无内部锚点**（Qwen3VL 未打 `_profile_speed`），无法内部分解，可能同样含异步伪影或任务输入差异。

**待排查（已解决）**：待对齐 `vlm_cgm.model`/`vlm_model.*` 锚点来源已定位——
**whh_vla_trt2 容器的 transformers 安装被手工修改**：`modeling_qwen3_vl.py` 被注入了
`_profile_speed` 打桩（13 处）：
- `Qwen3VLModel.forward` → `vlm_model.embed / get_image_features / mask_inject / compute_position_ids / language_model`
- `Qwen3VLForConditionalGeneration.forward` → `vlm_cgm.model`

对比：whh_vla_trt（参考）transformers 原版（`_profile_speed` 0 次）→ 参考 vlm_model.* 全 0；
whh_vla_trt2（待对齐）打桩版（`_profile_speed` 13 次）→ 待对齐有完整锚点。

**影响**：待对齐能分解 vlm.forward 内部（vit 16 + llm 20 ≈ 37ms），参考不能。若需参考也输出
vlm_model.* 锚点，需在 whh_vla_trt 容器同样修改 transformers（不推荐污染标准库）或用
monkey-patch 方式（类似新文件的打桩）替代。

### 3.12 替换 whh_vla_trt transformers：参考链路锚点对齐（2026-09-02）✅

**操作**：把 whh_vla_trt2 的整个 transformers 包替换进 whh_vla_trt。
- 备份：whh_vla_trt 原版 → `/root/transformers_bak_whh_vla_trt_20260902.tar.gz`（19MB）
- 打包 whh_vla_trt2 的 transformers → tar → 经宿主机中转 → 解包替换到 whh_vla_trt
- 验证：替换后 `modeling_qwen3_vl.py` `_profile_speed` 从 0 → 13 次；transformers 5.5.4 import 正常；参考链路跑通无报错

**替换后参考链路 VLM 内部锚点出现**（vs 待对齐）：

| VLM 内部 | 参考 (grasp_book) | 待对齐 (sortbook) | 差异 |
|------|------|------|------|
| vlm.forward | 49.68ms | 37.77ms | 参考慢 12ms |
| vlm_cgm.model | 49.42ms | 37.49ms | 参考慢 12ms |
| **vlm.vit** | **29.60ms** | **16.04ms** | **参考慢 13.5ms（差异主体）** |
| vlm.llm | 18.50ms | 19.99ms | 待对齐略快 |
| vlm.mask_inject / mrope | 0.37 / 0.79 | 0.37 / 0.91 | 接近 |

**结论**：VLM 差异（~12ms）几乎全部来自 **`vlm.vit`（29.6 vs 16.0ms）**。两条链路 ViT 结构相同
（Qwen3VL, num_patches=144, image_size=384），差异根源是**任务输入图像内容不同**
（grasp_book 抓书场景 vs sortbook 整理书场景）——不同图像的 ViT kernel 执行耗时波动，
非代码/结构差异。`vlm.llm` 两链路接近（LLM 部分无差异）。

### 3.14 vlm.vit 耗时差异根因：ViT 运行精度不同（2026-09-02）✅

**问题**：vlm.vit 参考 ~29ms vs 待对齐 ~15ms（2 倍）。之前 3.12 误判为「图像内容差异」，
实测后修正为**精度差异**。

**排查过程（逐项排除）**：
| 变量 | 结果 |
|------|------|
| ViT 结构（depth/hidden/heads/patch） | ✅ 两链路一样（24/1024/16/16） |
| 权重 | ✅ md5 相同（36d589f11ed...） |
| 输入 shape | ✅ 一样（pv 1152×1536, grid 2×3） |
| 环境（torch/cudnn/cuda） | ✅ 一样（2.7.0+cu128/90701/12.8） |
| GPU 时钟（跑时采样） | ✅ 都满频 2902MHz / 385W |

**实测（torch.cuda.Event 精确测 GPU 耗时）**：
| 链路 | pv_dtype / visual_dtype | vit GPU 耗时 |
|------|------|------|
| 参考 (whh_vla_trt) | **torch.float32** | **~28ms** |
| 待对齐 (whh_vla_trt2) | **torch.bfloat16** | **~13ms** |

**根因**：**两条链路 ViT 运行精度不同**——参考 FP32（慢 2 倍），待对齐 BF16（快 2 倍）。
相同结构/权重/输入下，FP32 计算量是 BF16 的 2 倍 → 耗时 2 倍，完全吻合。

**为什么精度不同**：
- 两链路 cfg 都是 `vision_language.dtype: bf16` + `train.mixed_precision: "no"`
- `MappingTorchDtype["no"] = torch.float32`（注意！"no" 映射到 fp32）
- `build_model`：`model_dtype = config["train"]["mixed_precision"]` = "no" → **fp32**
- 参考链路按 fp32 加载/运行 ViT → 保持 fp32
- 待对齐链路 ViT 运行时为 bf16 → **某个环节被转回 bf16（待排查：qwen2_5_vl.to() 差异——参考只记录 `_torch_dtype` 不改 vlm；待对齐显式 `self.vlm.to(dtype=self._torch_dtype)`）**

**待排查**：待对齐链路 ViT 为何从 fp32（model_dtype 推导）变回 bf16——疑为
`Qwen25VLEncoder.to()` 差异（待对齐 807 行显式 `self.vlm.to(dtype=self._torch_dtype=bf16)`，
参考 651 行只 `self._torch_dtype = next(self.vlm.parameters()).dtype` 不改 vlm）。

### 3.15 待对齐 ViT 保持 bf16 的根因确认（2026-09-02）✅

**待排查已解决**：待对齐 ViT 保持 bf16 的根因是 **`Qwen25VLEncoder.to()` 实现差异**。

**对比**：
| | 参考 (whh_vla_trt) | 待对齐 (whh_vla_trt2) |
|------|------|------|
| `to()` 实现 | `super().to()` 递归转换 + `_torch_dtype = next(vlm.params).dtype`（跟随） | **不调 super**，强制 `self.vlm.to(dtype=self._torch_dtype)` |
| 外层 `model.to(fp32)` 后 ViT | **fp32**（被递归转换） | **bf16**（被强制转回 `_torch_dtype`=bf16） |
| vit 耗时 | ~28ms | ~13ms |

**根因**：待对齐 `Qwen25VLEncoder.to()` 重写了默认行为——不调用 `super().to()`，
而是**强制 `self.vlm.to(dtype=self._torch_dtype)`**。`_torch_dtype` 初始为 cfg `dtype: bf16`，
所以无论外层传什么 dtype，ViT 都被强制转回 bf16。

**定性**：这是**待对齐链路故意的 BF16 加速**（vit 快 2 倍），非随机 bug。
但需注意精度：若待对齐模型训练/量化为 bf16 则无损；若本应 fp32（参考的无损精度偏好），
bf16 推理会有精度损失（需验证 max_diff 是否可接受）。

**建议**：若需待对齐 ViT 也用 fp32（对齐参考精度），改待对齐 `to()` 为参考的实现
（调 super + 跟随 vlm dtype）；若接受 bf16 加速，则保持现状并确认精度 OK。

### 3.16 待对齐 ViT 改 fp32，对齐参考链路（2026-09-02）✅

**改动**（用户决定走 fp32 对齐参考）：改待对齐 `qwen2_5_vl.py` 的 `Qwen25VLEncoder.to()`，
从 bf16 强制版改成参考实现：
- 改前（bf16 强制）：`def to(self, device, *args, **kwargs)` 不调 super，
  `self.vlm.to(dtype=self._torch_dtype=bf16)` → ViT 恒 bf16
- 改后（参考版）：`def to(self, *args, **kwargs)` 调 `super().to()` 递归转换 + 
  `self._torch_dtype = next(self.vlm.parameters()).dtype`（跟随）→ ViT 跟随外层 fp32

备份：`qwen2_5_vl.py.bak_bf16`

**验证**（CUDA Event 实测 + Speed Report）：
- 改前：ViT bf16 ~13-15ms
- 改后：ViT **fp32 ~29-31ms**（跟参考 28-30ms 对齐）

**改后待对齐 vs 参考 Speed Report**（`logs/infer_torch_20260902_094412.log`）：

| 环节 | 参考 | 待对齐（改后） |
|------|------|------|
| vlm.vit | 29.71ms | 30.39ms ✅ |
| vlm.forward | 49.70ms | 49.67ms ✅ |
| vlm.llm | 18.32ms | 17.59ms ✅ |
| prepare_images | 76.40ms | 76.77ms ✅ |

链路完整跑通无报错，VLM 部分耗时与参考完全对齐。

## 五、hpc_opt/trtllm 优化项移植（sortbook 版）✅

### 5.1 移植总览（2026-09-02，全部落到 scripts/sortbook-opt/，未 commit 未改 dexechain 源码）

| 模块 | 移植产物（sortbook-opt/ 下） | 验证结果 |
|------|------|------|
| vit（3 步流程） | vit/（vit_wrapper + export_fp16 + build_engine + verify + capture/） | wrapper **max_diff=0**；fp16 onnx 775MB；fp16 engine 787MB（61.8s）；engine vs PT max_diff<2.0；**4.27ms/iter** |
| llm | llm/（llm_wrapper add_ds + export + build + gen_fp32_layers + capture） | fp16 onnx(5.4GB) → fp16+RMSNorm-fp32 engine 2.6GB；verify **ULP=1.2**；5.08ms/iter |
| prepare_inputs | prepare_inputs/（gpu_prepare_inputs + hook + verify） | 6 帧 bf16 量化级对齐（raw 1.2e-7）；**2.7x 加速** |
| get_image_hidden_state | get_image_hidden_state/（gpu_get_image_hs + verify + capture） | 4 帧 **max_diff=0**；1.6x |
| cerebellum | cerebellum/（wrapper + refine_wrapper + export×2 + build×2 + capture） | sortbook 512/12/8/split_arm_cond=False 适配；wrapper vs 原始 **max_diff=0**；engine max_diff=0.0018；**0.705ms/iter** |
| vitllm_infer | vitllm_infer/（infer_trt.sh + sitecustomize + trt_hooks + shim 等 8 文件） | **端到端跑通**，见 5.2 |

### 5.2 端到端验证：infer_trt.sh（TRT 代理）vs infer_realdata.sh（torch 基线）✅

真实数据 checkpoint-24000 + test_speed_data，100 iters avg：

**更新（2026-09-02 清理时修复 infer_trt.sh LLM engine 路径 bug 后）**：
初版 64.85ms 的 LLM 部分实际是 torch fallback（infer_trt.sh 的 DEFAULT_LLM_ENGINE 指向不存在的
`fp32onnx_fp16engine_fp32RMSNorm/`），修复为 `fp16_fp32_layers/` 后 LLM TRT 真正接入：

| 环节 | torch | TRT（修复后） | 加速 |
|------|------|------|------|
| **total_inference** | **138.27ms** | **47.71ms** | **2.9x** |
| vlm.prepare_inputs | 9.36ms | 4.25ms | 2.2x |
| vlm.forward | 48.39ms | 13.18ms | 3.67x |
| **vlm.vit** | 29.62ms | **4.23ms** | **7.0x** 🚀 |
| vlm.llm | 17.04ms | **7.29ms** | **2.34x**（修复前 torch 19.29ms）|
| vlm.get_image_hs | 14.37ms | 12.40ms | 1.16x |
| prepare_images.encoder | 72.97ms | 43.13ms | 1.69x |
| **cerebellum forward_action** | 52.43ms | **15.81ms** | **3.32x** |
| cerebellum refine | 4.96ms | 2.04ms | 2.43x |
| cerebellum_total | 61.83ms | 20.91ms | 2.96x |

**结论**：vlm.vit（TRT engine 7x）、cerebellum（3x）、LLM（修复后 2.34x）为主要加速来源，
端到端 **2.9x**。TRT 链路 hooks（vit/llm/cerebellum/refine）全装上（trt-shim 确认 3 engine 就绪）、无 crash、Speed Report 完整。

**LLM engine 路径 bug 修复**（清理 sortbook-opt 时发现）：infer_trt.sh 的 DEFAULT_LLM_ENGINE
原指向 `llm_engine/fp32onnx_fp16engine_fp32RMSNorm/llm_add_ds_fp16_fp32_layers.plan`（不存在），
导致端到端时 LLM 静默降级 torch（trt_hooks.py 路径正确但 shim 用了 infer_trt.sh 的环境变量）。
改为 `llm_engine/fp16_fp32_layers/llm_add_ds_fp16_fp32_layers.plan` 后 LLM TRT 生效：
vlm.llm 19.29→7.29ms，端到端 64.99→47.71ms。

### 5.3 关键实现要点
- engine 精度策略：vit fp16onnx_fp16engine（fp32 IO）；llm fp16 + RMSNorm 层 fp32（108 子图×6 节点 layerPrecisions）；cerebellum forward fp16 + norm fp32、refine 纯 fp16（bit 级）
- sortbook cerebellum 差异适配：512/12/8、split_arm_cond=False（intermediate=x 本身，非参考 cat 版）、refiner 12 层真实训练需导出
- 所有真实输入截取（checkpoint-24000 + test_speed_data 过评测入口）替代 sortbook 不存在的 llm_pt_reference_with_ds.safetensors
- vitllm_infer engine 默认路径指向 sortbook 产物（vit/llm/cerebellum/refine 4 个 plan）

### 5.4 对比 2 非 0 根因：attention 后端不一致（sdpa vs eager）✅（2026-09-02）

**问题**：verify_vit_wrapper 对比 2（standalone vs 真实链路截取）有 ds2=1.85e-4 微小差异，
用户质疑「应该为 0」。经多轮排查（dtype/权重/显存/CPU 对照等全排除）后用**逐环节对比脚本**
锁定根因。

**排查链（逐环节记录 dtype+数值）**：
- patch_embed_out / b0：0 差异
- **b0.attn_out：首个分叉 9.5e-7** → b1.norm2 放大 3.8e-6 → 逐层放大到 ds2 1.44e-4
- monkey-patch `F.scaled_dot_product_attention` 计数：**v1(standalone)=48 次，v2(DexForceVLA)=0 次**

**根因**：两个 visual 的 `config._attn_implementation` 不同：
| | _attn_implementation | attention 实现 |
|---|---|---|
| standalone（from_pretrained 不指定） | **sdpa** | `F.scaled_dot_product_attention` |
| DexForceVLA 链路（qwen3_vl.py 默认 eager） | **eager** | `eager_attention_forward`（手动 matmul） |

sdpa 与 eager 数学等价但浮点归约顺序不同 → block0 attn 差 9.5e-7 → 24 层放大到 1e-4。
这也解释了 deepcopy 0 差异（deepcopy 保留 sdpa config）、CPU/GPU 都有微差（两后端在两硬件都不同）。

**修复**：vit_wrapper.py 的 `from_pretrained` 加 `attn_implementation="eager"`（与链路一致）。

**验证**：修复后 verify_vit_wrapper 重跑——**对比 1 全 0 + 对比 2 全 0（bit 级复现链路）**，
pooler/ds0/ds1/ds2 max_diff 全部 0.000000。彻底闭环。

### 5.5 重新导出 eager onnx/engine + 新旧误差对比 ✅（2026-09-02）

vit_wrapper 改 eager 后，已存在的 sdpa 语义 onnx（10:27）/engine（10:30）与新 wrapper 不一致，
按方案 A 重新导出+build，并与旧 sdpa 版对比。

**重新导出链**（eager，覆盖旧 sdpa 产物）：
- export fp16 onnx：774.9MB，**节点 1672**（旧 sdpa 是 1624，确认 eager 语义新图）
- build fp16 engine：786.3MB，`&&&& PASSED`
- verify_vit_engine：OK（max_diff<2.0），engine 4.19ms/iter
- 端到端 infer_trt.sh：跑通，total_inference 47.71ms（LLM engine 路径 bug 修复后，见 §5.2），vs torch 基线 ~2.9x

**新旧 engine 误差对比**（verify_vit_engine，真实权重 checkpoint-24000 + 真实链路输入，FP16 加载同精度基准）：

| 输出 | sdpa（旧）max/mean | eager（新）max/mean | 变化 |
|---|---|---|---|
| pooler_output | 0.1445 / 0.0036 | 0.1381 / 0.0037 | max ↓ 4.4% |
| deepstack_0 | 0.0313 / 0.0013 | 0.0313 / 0.0013 | 持平 |
| deepstack_1 | 0.2246 / 0.0027 | 0.1060 / 0.0026 | max ↓ 52.8% |
| deepstack_2 | 1.1719 / 0.0079 | 1.2031 / 0.0080 | max ↑ 2.7% |

engine vs PyTorch FP32（跨精度参考）：pooler 0.1477、ds0 0.0275、ds1 0.1113、ds2 1.0639（sdpa 版 0.1323/0.0342/0.2427/0.8848）。

**结论**：新旧误差量级相当（fp16 量化正常范围，阈值 2.0 内实测 <1.3）；deepstack_1 明显改善
（0.225→0.106），pooler 略降，ds2 微升（+0.03 可忽略），mean_diff 全一致（整体分布不变仅个别峰值点差异）；
耗时持平（4.19~4.27ms）。eager 与真实链路语义一致 → 新 engine 更正确。


### 3.13 待对齐 report 对齐参考标准链路（2026-09-02）✅

**目标**：待对齐链路 Speed Report 完全对齐参考标准链路的打桩方式/统计点/打印格式。

**改动**（新文件 monkey-patch，不改 `_profile.py` 主流程）：
1. `_run_one_sample_with_vlm` → 完全复刻参考 `_run_one_sample` 的 speed_stats 映射（45+ 键）：
   brain entry/exit overhead、from_batch.prepare_*（dtype_convert/augment/lang/images/geomap/state/
   affordance/vqa）、prepare_images.preprocess/to_device/encoder/postprocess、vlm.* 全套、
   cerebellum forward_action/refine 等
2. `_print_brain_cere_with_vlm` → 参考完整树（predict_action.brain_infer → from_batch → prepare_images
   内部 → vlm → postprocess → cerebellum forward_action/refine）
3. `_srow`/`_ssep`/`_sheader` → 参考版（`_COL_LABEL=46` + `"  "+" "*depth` + 中文表头）

**验证**（`logs/infer_torch_20260902_092043.log`）：待对齐 Speed Report 统计点/树形/列宽
**与参考完全一致**：
- `predict_action.brain_infer` → `brain_sub_sum` → `from_batch` → `dtype_convert/augment/prepare_lang/
  prepare_images` → `├ preprocess/to_device/encoder` → `vlm.*` → `└ postprocess` → `prepare_geomap/...`
- cerebellum：`forward_action`（57.68ms）/`refine`（5.28ms）显示（之前待对齐缺失）
- prepare_lang/prepare_images 等**有值**（之前全 0）

**已知差异**：Speed Report 表头行仍为待对齐 `_profile.py` 硬编码的
`metric avg / min / max (ms)`；参考为「统计点/平均(ms)/最小(ms)/最大(ms)」+ 单位说明行。
表头在 run_evaluation 主流程（非 monkey-patch 范围），如需对齐需直接改 `_profile.py`。

---

## 四、验证结果

（待补充：关键验证命令与输出）
