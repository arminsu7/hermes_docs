# VLA 推理链路详细分析

> 仓库：embodichain (DexForce VLA)
> 远程环境：ssh oem@192.168.11.51 → docker exec whh_vla_trt2 → conda activate py310
> 仓库根目录：/root/workspace/embodichain/
> 模型 checkpoint：/root/workspace/models/checkpoint-46000/
> 配置文件：grasp_book_qwen3vl2b_0601_bce_master.yaml
> 硬件：NVIDIA RTX 5090 D v2
> 分析日期：2026-08-21

---

## 一、总体架构：Brain-Cerebellum 双层结构

DexForceVLA 采用 "大脑-小脑" 分层架构：

```
                        ┌─────────────────────────────────────────┐
                        │         DexForceVLA (Full Model)         │
                        │  dexforcevla_runner.py:DexForceVLA      │
                        │                                         │
   batch ──► predict_action ──┬──► brain_infer    (Brain=VLM编码)  │
                               │   - from_batch: 图像/语言/状态编码 │
                               │   - _compute_adaptors (第1轮)      │
                               │   - _compute_priviliges             │
                               │   - _compute_adaptors (第2轮)      │
                               │                                     │
                               ├──► cerebellum_infer (小脑=扩散策略) │
                               │   - inference: 去噪采样            │
                               │   - post_process: 反归一化/相对→绝对 │
                               └──► action tensor [1, T, action_dim] │
                        └─────────────────────────────────────────┘
```

- **Brain（大脑）**：Qwen3-VL-2B 视觉语言模型，负责图像编码 + 语言理解，输出 multimodal embedding tokens
- **Cerebellum（小脑）**：DexFlowMatchingPolicy (基于 Diffusion/Flow Matching)，负责动作轨迹生成
- **Adaptors**：Brain 和 Cerebellum 之间的特征适配 MLP

---

## 二、入口脚本到推理函数的完整调用链

### 2.1 启动链路

```
scripts/infer_realdata.sh
  │
  ├─ 设置环境变量：CUDA_VISIBLE_DEVICES=0, TORCH_COMPILE=1, EVAL_MODE=sync_full
  ├─ 设置 profiling 开关：PROFILE_SPEED=0, PROFILE_MEM=0, PROFILE_NVTX=0
  ├─ 设置 WARMUP=1, PROFILE_ITERS=100
  │
  └─ python tests/models/test_eval_models_by_realdata_profile.py
       │  --config_path    /root/workspace/models/checkpoint-46000/grasp_book_qwen3vl2b_0601_bce_master.yaml
       │  --pretrained_model_name_or_path /root/workspace/models/checkpoint-46000
       │  --dataset_path   /root/workspace/datas/test_speedup
       │  --sample_num 1
       │  --warmup 1 --profile_iters 100
       │
       └─ eval_models_by_realdata(vars(args))
            │
            └─ run_evaluation(args)                          ← 主入口函数
                 │  文件: dexechain/evaluation/vla/dexforcevla_sim_by_realdata_profile.py
                 │
                 ├─ [Stage 0] 初始化
                 │   ├─ SimulationManager (headless, cpu)
                 │   ├─ ProfileConfig (speed/mem/nvtx 开关)
                 │   └─ deterministic_seed(42)
                 │
                 ├─ [Stage 1] build_model(config)            ← 模型构建
                 │   │  文件: dexechain/agents/dexforce_vla/train/build_model.py
                 │   ├─ load_vla_config(yaml)                 ← 配置加载 (_BASE_ 继承)
                 │   │  文件: dexechain/agents/dexforce_vla/config.py
                 │   ├─ load_model_from_pretrained(DexForceVLA, checkpoint_path)
                 │   ├─ (可选) LoRA
                 │   ├─ model.set_profile_config(deploy_compile, is_nvtx, ...)
                 │   └─ compile_model(model)                   ← torch.compile (可选)
                 │      文件: dexechain/agents/dexforce_vla/models/dexforcevla_runner.py
                 │
                 ├─ [Stage 2] VLAConsumerDataset + DataLoader ← 数据加载
                 │   │  文件: dexechain/agents/dexforce_vla/train/vla_dataset.py
                 │   ├─ VLAConsumerDataset(config, data_path, ...)
                 │   └─ DataCollatorForVLAConsumerDataset()
                 │
                 ├─ [Stage 3A] 评估路径 (EVAL_MODE=sync_full)
                 │   └─ run_sync_full_evaluation(dataset, model, ...)
                 │      ├─ 遍历 episode (step_id 滑窗)
                 │      ├─ predict_action_chunk(model, batch, ...)
                 │      │   └─ model.predict_action(batch, ...)  ← 核心推理入口
                 │      ├─ build_gt_target(batch, model, ...)
                 │      ├─ collect_scope_metric_rows (MAE/MSE/RMSE)
                 │      └─ plot_action_from_tensor (GT vs Pred 绘图)
                 │
                 └─ [Stage 3B] Profile 路径 (PROFILE_SPEED=1 时)
                    ├─ data_prep (预取 batch → GPU)
                    ├─ warmup (pfCfg.warmup 轮)
                    ├─ profile_iters (pfCfg.profile_iters 轮)
                    │   └─ _run_one_sample(model, batch, chunk_size, pfCfg)
                    │       └─ model.predict_action(batch, action_only=True, ...)
                    └─ 报告生成 (speed_report.json, mem_report.json)
```

### 2.2 核心推理函数：predict_action

**文件**：`dexechain/agents/dexforce_vla/models/dexforcevla_runner.py`
**方法**：`DexForceVLA.predict_action()`

```
predict_action(batch, action_only=True, inference_horizon=32, **kwargs)
  │
  ├─ profile_push("predict_action.brain_infer")
  ├─ data = brain_infer(batch, **kwargs)         ← Brain 推理
  ├─ profile_pop("predict_action.brain_infer")
  │
  ├─ [可选] SAVE_BRAIN_DATA: 保存 brain 输出 → brain_data.pt (用于 ONNX/TRT 导出)
  │
  ├─ profile_push("predict_action.cerebellum_infer")
  ├─ result = cerebellum_infer(data, ...)        ← Cerebellum 推理
  ├─ profile_pop("predict_action.cerebellum_infer")
  │
  └─ return result  (action tensor 或 tuple)
```

---

## 三、Brain 推理链路：brain_infer → from_batch

### 3.1 brain_infer

**文件**：`dexechain/agents/dexforce_vla/models/dexforcevla_runner.py`

```
brain_infer(batch, **kwargs)
  │
  ├─ [可选] predict_sparse_affordance (VLM 生成 affordance, 仅 reasoner 模式)
  │
  ├─ profile_push("brain_infer.from_batch")
  ├─ data = from_batch(batch, use_fix_aug, debug)     ← 核心编码
  ├─ profile_pop("brain_infer.from_batch")
  │
  ├─ profile_push("brain_infer.compute_adaptors1")
  ├─ data = _compute_adaptors(data, COMPUTE_BEFORE_PRIVILEGE)  ← 第1轮适配器
  ├─ profile_pop("brain_infer.compute_adaptors1")
  │
  ├─ profile_push("brain_infer.compute_priviliges")
  ├─ data = _compute_priviliges(data)                 ← 特权估计器
  ├─ profile_pop("brain_infer.compute_priviliges")
  │
  ├─ profile_push("brain_infer.compute_adaptors2")
  ├─ data = _compute_adaptors(data, COMPUTE_AFTER_PRIVILEGE)   ← 第2轮适配器
  ├─ profile_pop("brain_infer.compute_adaptors2")
  │
  └─ return data  (Dict[Modality/PrivilegeType → ModalInput/Privilege])
```

### 3.2 from_batch — 多模态编码核心

**文件**：`dexechain/agents/dexforce_vla/models/dexforcevla_runner.py`
**行号**：886

```
from_batch(batch, use_fix_aug=True, debug=False)
  │
  ├─ batch dtype 转换
  │
  ├─ [推理模式] VLAAugmentor.augment(batch)       ← 图像增广(pad_to_square)
  │  文件: dexechain/agents/dexforce_vla/data/model_processor.py
  │
  ├─ prepare_lang(batch)                           ← 语言准备
  │  ├─ 如果有 VLM encoder → 返回 raw instruction (稍后在 prepare_images 中编码)
  │  └─ 如果有独立 lang encoder → 直接编码
  │  返回: Lang(data, mask)
  │
  ├─ prepare_images(batch, "images", lang_data)    ← 图像+VLM 编码 (最耗时)
  │  │
  │  ├─ images_processor.preprocess(images)        ← 图像预处理(resize/normalize)
  │  ├─ images_encoder(images, task_instructions)  ← VLM forward
  │  │   └─ Qwen3VLEncoder.forward(images, instructions)
  │  │       └─ forward_for_dexforcevla(images, instructions)
  │  │           ├─ prepare_inputs(images, instructions)    ← processor + tokenize
  │  │           │   日志: "Input preparation time: X seconds"
  │  │           ├─ self.vlm(**inputs, output_hidden_states=True)  ← Qwen3-VL 前向
  │  │           │   日志: "VLM forward pass time: X seconds"
  │  │           ├─ _get_image_hidden_state(final_hs, inputs)    ← 提取 image token 位置的 hidden state
  │  │           └─ _get_text_hidden_state(final_hs, inputs)     ← 提取 text token 位置的 hidden state
  │  │
  │  └─ 返回: ret_image(Image, shape=[bs, hs, num_cam, patch_len, hidden]),
  │           ret_lang(Lang, shape=[bs, text_len, hidden])
  │
  ├─ prepare_images(batch, "geomap", lang_data)    ← GeoMap 编码 (可选, 通常为 None)
  │
  ├─ prepare_state(batch)                          ← 状态编码
  │  ├─ [可选] state_moving_avg 归一化 (use_bn)
  │  ├─ state_encoder(states[:, :, :action_dim], state_indicator)  ← 状态 MLP
  │  └─ _prepare_state(states, state_indicator, ...)  ← 拼接 proprioception
  │     文件: dexechain/agents/dexforce_vla/models/utils.py
  │
  ├─ prepare_affordance(batch)                     ← Affordance 编码 (可选)
  │  └─ AffordanceParser.aggregate_affordance_by_cam → affordance_encoder
  │
  ├─ prepare_vqa(batch)                            ← VQA 准备 (可选, 仅 reasoner)
  │
  └─ return data  (Dict[Modality → ModalInput])
      包含: state_indicator, action_indicator, lang, lang_indicator,
            states, images, geomap, sparse_affordance, dense_affordance
```

### 3.3 Brain 子模块速查

| 子方法 | 文件 | 行号 | 功能 |
|--------|------|------|------|
| `from_batch` | dexforcevla_runner.py | 886 | 多模态编码主入口 |
| `prepare_lang` | dexforcevla_runner.py | 747 | 语言准备 |
| `prepare_images` | dexforcevla_runner.py | 579 | 图像+VLM 编码 |
| `prepare_state` | dexforcevla_runner.py | 787 | 状态编码+归一化 |
| `prepare_affordance` | dexforcevla_runner.py | 841 | Affordance 编码 |
| `prepare_vqa` | dexforcevla_runner.py | 858 | VQA 输入准备 |
| `_compute_adaptors` | dexforcevla_runner.py | ~1050 | 适配器 MLP 前向 |
| `_compute_priviliges` | dexforcevla_runner.py | ~1020 | 特权估计器前向 |
| `Qwen3VLEncoder.forward` | vlms/qwen3_vl.py | (继承) | VLM 前向入口 |
| `Qwen25VLEncoder.forward_for_dexforcevla` | vlms/qwen2_5_vl.py | 400 | VLM 实际前向 |
| `Qwen25VLEncoder.forward` | vlms/qwen2_5_vl.py | 500 | VLM forward 调度 |

---

## 四、Cerebellum 推理链路：cerebellum_infer → diffusion sampling

### 4.1 cerebellum_infer

**文件**：`dexechain/agents/dexforce_vla/models/dexforcevla_runner.py`

```
cerebellum_infer(data, action_only=True, inference_horizon=32, **kwargs)
  │
  ├─ profile_push("cerebellum_infer.inference")
  ├─ data = cerebellum.inference(data, inference_horizon, adaptors)  ← 扩散采样
  ├─ profile_pop("cerebellum_infer.inference")
  │
  ├─ profile_push("cerebellum_infer.post_process")
  ├─ [可选] action_moving_avg 反归一化 (use_bn)
  ├─ post_process(data, is_training, ...)    ← 相对→绝对动作转换
  │  文件: dexechain/agents/dexforce_vla/models/utils.py:215
  ├─ [可选] visualization (attention 可视化)
  ├─ profile_pop("cerebellum_infer.post_process")
  │
  └─ return data[Modality.ACTIONS.value]  或  tuple(action, exp, vis, imgs, mask)
```

### 4.2 cerebellum.inference — 扩散策略采样

**文件**：`dexechain/agents/dexforce_vla/models/policy/dexdp.py` (DexDiffusionPolicy)
**继承链**：DexFlowMatchingPolicy → DexDiffusionPolicy → DiTPolicyWithAdditionalModality → DiTPolicy

```
inference(data, inference_horizon=32, **kwargs)
  │
  ├─ 提取条件: state_cond, img_cond, lang_cond, disp_cond, mask, exteroception_cond, ...
  │
  ├─ noisy_action, intermediate_output = _inference_noisy_action(...)  ← 去噪循环
  │  │  文件: dexdp.py:475
  │  │
  │  ├─ noisy_action = torch.randn(bs, inference_horizon, output_dim)  ← 初始噪声
  │  ├─ noise_scheduler_sample.set_timesteps(num_inference_timesteps)
  │  │
  │  └─ for t in timesteps:                    ← 去噪迭代 (默认10步)
  │      ├─ action_traj = cat([noisy_action * mask, mask], dim=2)
  │      ├─ action_cond = action_adaptor(action_traj)
  │      ├─ [可选] exp_action_binding (exteroception 绑定)
  │      ├─ state_action_traj = cat([state_cond, action_cond], dim=1)
  │      ├─ model_output, inter = forward_action(...)   ← DiT 前向 (最耗时)
  │      │  文件: dexdp_blocks.py:491
  │      │  ├─ t_embedder(t)                        ← 时间步嵌入
  │      │  ├─ [可选] progress_embeder(progress)
  │      │  ├─ cat([t, progress, sparse_aff, x])   ← 拼接条件 token
  │      │  ├─ x_pos_embed 位置编码
  │      │  ├─ _parse_img_c(img_c, mask_pred, shape) ← 图像条件解析
  │      │  ├─ [可选] dense_aff 加到 img_c
  │      │  └─ DiT blocks (self.blocks) 前向       ← Transformer 主干
  │      │     文件: dexdp_blocks.py (DiTBlock / DiTXBlock)
  │      │
  │      └─ noisy_action = noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
  │
  ├─ [可选] forward_exp (exteroception 精化)
  │
  ├─ init_action = noisy_action[:, :, :output_dim] * action_mask
  │
  ├─ [use_refiner=True 时] 精化阶段:
  │  ├─ full_x = action_adaptor(cat([sigmoid(init_action), mask]))
  │  ├─ [可选] exp_action_binding
  │  ├─ refined_action = refine(full_x, intermediate_output, img_cond, ...)  ← Refiner 前向
  │  │  文件: dexdp_blocks.py:931
  │  │  ├─ img_cond + img_cond_pos_embed
  │  │  ├─ x + refine_cond_pos_embed
  │  │  ├─ _parse_img_c(img_c, mask_pred, shape)
  │  │  ├─ for block in self.refiner: x = block(x, ada_ln_cond, c, m)
  │  │  └─ refiner_final_layer(cat([init_x, x])) → action
  │  │
  │  └─ ret_action = init_action + refined_action * action_mask
  │
  └─ ret_action = _apply_sigmoid_to_gripper(ret_action) * action_mask
```

### 4.3 Cerebellum 子模块速查

| 子方法 | 文件 | 行号 | 功能 |
|--------|------|------|------|
| `inference` | policy/dexdp.py | 567 | 扩散采样主入口 |
| `_inference_noisy_action` | policy/dexdp.py | 475 | 去噪循环 |
| `forward_action` | policy/dexdp_blocks.py | 491 | DiT 前向（含 cross-attn） |
| `refine` | policy/dexdp_blocks.py | 931 | Refiner 精化 |
| `forward_exp` | policy/dexdp_blocks.py | 238 | Exteroception 前向 |
| `post_process` | models/utils.py | 215 | 相对→绝对转换 |
| `DiTPolicyWithAdditionalModality` | policy/dexdp_blocks.py | 196 | DiT 策略基类 |

---

## 五、完整调用链路一图总览

```
infer_realdata.sh
└─ test_eval_models_by_realdata_profile.py
   └─ run_evaluation(args)                              [eval/vla/dexforcevla_sim_by_realdata_profile.py]
      ├─ build_model(config)                           [train/build_model.py]
      │  ├─ load_vla_config(yaml)                     [config.py]
      │  ├─ load_model_from_pretrained(DexForceVLA)   [models/utils.py]
      │  └─ compile_model(model)                      [dexforcevla_runner.py:compile_model]
      │
      ├─ VLAConsumerDataset(config, data_path)         [train/vla_dataset.py]
      ├─ DataCollatorForVLAConsumerDataset()
      │
      └─ [sync_full] run_sync_full_evaluation(...)
         └─ predict_action_chunk(model, batch, ...)    [eval/vla/dexforcevla_sim_by_realdata_profile.py]
            └─ model.predict_action(batch, ...)        [dexforcevla_runner.py]
               │
               ├─ brain_infer(batch)                   [dexforcevla_runner.py]
               │  ├─ from_batch(batch)                 [dexforcevla_runner.py:886]
               │  │  ├─ prepare_lang(batch)            [dexforcevla_runner.py:747]
               │  │  ├─ prepare_images(batch,"images") [dexforcevla_runner.py:579]
               │  │  │  └─ Qwen3VLEncoder.forward()    [vlms/qwen3_vl.py → qwen2_5_vl.py:500]
               │  │  │     └─ forward_for_dexforcevla() [vlms/qwen2_5_vl.py:400]
               │  │  │        ├─ prepare_inputs(images, instructions)
               │  │  │        ├─ self.vlm(**inputs, output_hidden_states=True)
               │  │  │        ├─ _get_image_hidden_state()
               │  │  │        └─ _get_text_hidden_state()
               │  │  ├─ prepare_images(batch,"geomap")  [dexforcevla_runner.py:579]
               │  │  ├─ prepare_state(batch)           [dexforcevla_runner.py:787]
               │  │  │  └─ state_encoder(states, indicator)
               │  │  ├─ prepare_affordance(batch)     [dexforcevla_runner.py:841]
               │  │  └─ prepare_vqa(batch)            [dexforcevla_runner.py:858]
               │  ├─ _compute_adaptors(data, BEFORE)   [dexforcevla_runner.py]
               │  ├─ _compute_priviliges(data)          [dexforcevla_runner.py]
               │  └─ _compute_adaptors(data, AFTER)    [dexforcevla_runner.py]
               │
               └─ cerebellum_infer(data)               [dexforcevla_runner.py]
                  ├─ cerebellum.inference(data)        [policy/dexdp.py:567]
                  │  ├─ _inference_noisy_action(...)   [policy/dexdp.py:475]
                  │  │  └─ for t in timesteps:
                  │  │     └─ forward_action(...)      [policy/dexdp_blocks.py:491]
                  │  │        └─ DiT blocks 前向
                  │  ├─ [可选] forward_exp(...)         [policy/dexdp_blocks.py:238]
                  │  └─ [可选] refine(...)             [policy/dexdp_blocks.py:931]
                  │     └─ refiner blocks 前向
                  │
                  └─ post_process(data)                [models/utils.py:215]
                     └─ 相对→绝对 qpos 转换
```

---

## 六、实际运行参数与代码路径判定

### 6.1 Shell 脚本环境变量 (infer_realdata.sh)

| 变量 | 实际值 | 代码路径影响 |
|------|--------|-------------|
| TORCH_COMPILE | 1 | compile_model() 执行：cerebellum+adaptors 被 torch.compile(mode=reduce-overhead/default) |
| USE_TENSORRT | (空,未设) | 不走 TRT 路径，纯 PyTorch 推理 |
| BRAIN_COMPILE | (空,未设) | Brain VLM 不额外 compile（仅 TORCH_COMPILE 控制 cerebellum+adaptors） |
| BRAIN_FP8 | (空,未设) | 不做 FP8 量化 |
| AUTO_FP8 | (空,未设) | 不做自动 FP8 |
| BRAIN_VLLM | (空,未设) | 不用 vLLM 后端，走标准 transformers VLM forward |
| EVAL_MODE | sync_full | 走 run_sync_full_evaluation()（非 chunk_debug） |
| PROFILE_SPEED | 0 | 不走 profile 路径，走 sync_full 评估 |
| PROFILE_MEM | 0 | 不统计显存 |
| PROFILE_NVTX | 0 | 不加 NVTX 标记 |
| WARMUP | 1 | (仅 profile 路径生效) |
| PROFILE_ITERS | 100 | (仅 profile 路径生效) |
| SAMPLE_NUM | 1 | 每个视频取 1 帧采样 |
| SAVE_BRAIN_DATA | /root/workspace/embodichain/brain_data.pt | 保存第一次 brain 输出（用于 TRT 导出） |

### 6.2 模型配置 — YAML vs Checkpoint config.json

重要：由于 `pretrained_model_name_or_path` 指向 checkpoint-46000 目录，
build_model() 走 `load_model_from_pretrained` 路径，模型结构用的是
**checkpoint 内保存的 config.json**，而非 yaml。两者有 3 处关键差异：

| 参数 | YAML 值 | Checkpoint config.json 值 | 实际生效 |
|------|---------|--------------------------|---------|
| use_refiner | False | **True** | True → 走 refine() 精化路径 |
| num_inference_timesteps | 10 | **5** | 5 → 去噪循环只跑 5 步 |
| llm_trainable_idx | 12 | **0** | 0 → VLM 完全冻结，不训练 |
| vision_tower_trainable_idx | 12 | **0** | 0 → ViT 完全冻结 |

### 6.3 模型结构实际值 (checkpoint config.json)

**顶层**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| level | **full** | predict_action() 中走 brain_infer + cerebellum_infer 双阶段 |
| hidden_size | 384 | Cerebellum/Adaptors 的统一隐藏维度 |
| use_bn | **False** | 不走 state_moving_avg / action_moving_avg 归一化路径 |
| use_affordance | **False** | 不走 prepare_affordance / affordance_encoder |
| img_history_size | 1 | 单帧图像输入 |
| state_history_len | 1 | 单步状态 |
| pred_horizon | 64 | 最大 action chunk 长度 |
| test_action_chunk_size | 32 | 推理时 inference_horizon=32 |
| num_cameras | 2 | head + right_wrist |
| image_size | 384 | 图像 384×384 |
| arm_dofs | 14 | 单臂 7 DoF × 2 |
| state_dim | 128 | 状态/动作向量维度 |
| misc.pad_to_square | False | VLAAugmentor 不做 pad_to_square |

**Brain — Vision-Language Encoder**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| encoder name | **Qwen3VLEncoder** | 用 Qwen3-VL-2B 做 dual-modal 编码 |
| vlm_name_or_path | {}/extract/Qwen3VL/Qwen3VL | VLM 权重路径 (data_root 拼接) |
| output_hs_mode | **dual-modal** | 返回独立的 image_hidden_state + text_hidden_state |
| num_patches | 144 | (384/32)^2 = 144 patches/camera/frame |
| token_dim | 2048 | VLM hidden_size = 2048 |
| patch_size | 32 (默认) | Qwen3VLEncoder.__init__ 默认值 |
| llm_trainable_idx | 0 | VLM LLM 层全部冻结 |
| vision_tower_trainable_idx | 0 | ViT 全部冻结 |
| is_detached_for_actuator | False (默认) | 不 detach hidden state |
| final_layer_norm | False (默认) | Identity()，不加额外 LayerNorm |
| attn_implementation | eager (默认) | VLM 用 eager attention（非 sdpa/flash） |

VLM 输出: image_hidden_state [bs, 288, 2048]
  - 288 = img_history_size(1) × num_cameras(2) × num_patches(144)
  - 2048 = VLM hidden_size

**Brain — State Encoder**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| name | IdentityStateEncoder | 状态直接透传（不做编码），仅拼接 indicator |
| state_token_dim | 128 | 状态 token 维度 = state_dim |

**Brain — Adaptors (全部 SimpleAdaptor, mlp2x_gelu)**

| Adaptor | 输入 | in_features | out_features | compute_order | 说明 |
|---------|------|-------------|-------------|---------------|------|
| lang_adaptor | lang | 2048 | 384 | BEFORE_PRIVILEGE | 语言 2048→384 |
| images_adaptor | images | 2048 | 384 | BEFORE_PRIVILEGE | 图像 2048→384 |
| states_adaptor | states+state_indicator | 256 | 384 | BEFORE_PRIVILEGE | 状态 256→384 (128×2) |
| actions_adaptor | actions+action_indicator | 256 | 384 | AFTER_PRIVILEGE | 动作 256→384 (128×2) |

  - compute_order: lang/images/states 在特权估计前执行；actions 在特权估计后执行
  - 但 privileges=[] 为空，所以 _compute_priviliges() 实际不执行任何 estimator

**Brain — Privileges**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| privileges | **[]** (空) | _compute_priviliges() 遍历空 dict，什么都不做 |

**Cerebellum — DexDiffusionPolicy**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| name | DexDiffusionPolicy | 实际类: DexFlowMatchingPolicy(继承DexDiffusionPolicy) |
| depth | 8 | DiT 主干 8 层 block |
| num_heads | 4 | 4 头注意力 |
| hidden_size | 384 | Cerebellum 内部维度 |
| output_dim | 128 | 动作维度 = state_dim |
| use_refiner | **True** | 走 refine() 精化路径 |
| independent_action | True | 独立动作处理 |
| masking_gripper | True | gripper 状态被 mask 为 0 |
| block_type | DiTBlock (默认) | 使用 DiTBlock（非 DiTXBlock） |

**Cerebellum — Noise Scheduler (DDPM)**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| type | ddpm | DDPM 调度器 |
| num_train_timesteps | 500 | 训练时 500 步 |
| num_inference_timesteps | **5** | 推理时 5 步去噪 |
| beta_schedule | squaredcos_cap_v2 | 余弦调度 |
| prediction_type | **sample** | 预测样本本身（非 epsilon/v_prediction） |
| clip_sample | False | 不裁剪样本 |

**Cerebellum — Loss (推理时不走 loss，但影响 eef 处理)**

| 参数 | 值 | 代码路径影响 |
|------|-----|-------------|
| eef_loss_type | bce | _apply_sigmoid_to_gripper 对 gripper 做 sigmoid |
| loss_weight.eef_pose_action_loss | 10 | EEF pose 损失权重 |

**推导出的条件维度**

| 维度 | 值 | 计算 |
|------|-----|------|
| img_cond_len | 288 | img_history_size(1) × num_cameras(2) × num_patches(144) |
| geo_cond_len | 144 | img_history_size(1) × num_patches(144) |
| lang_cond_len | 500 | tokenizer_max_length |
| state_token_dim | 128 | = state_dim |

### 6.4 推理时的实际代码路径判定

基于以上参数，代码实际走以下路径：

```
predict_action()
  ├─ level="full" → 走 brain_infer + cerebellum_infer
  │
  ├─ brain_infer()
  │  ├─ VISION_LANGUAGE encoder 存在 + has predict_sparse_affordance
  │  │   → 检查 aff_cameras/aff_objects（推理时通常为 None）
  │  │   → 如果为 None → "Don't predict affordance in brain_infer" 跳过
  │  │
  │  ├─ from_batch()
  │  │  ├─ use_bn=False → 不走 moving_avg 归一化
  │  │  ├─ prepare_lang() → VISION_LANGUAGE encoder 存在 → 返回 raw instruction
  │  │  │   → 日志: "Using vision-language model later in prepare_images()"
  │  │  ├─ prepare_images(batch, "images")
  │  │  │  ├─ Modality.IMAGES in basic_modality → 走图像分支
  │  │  │  ├─ image_processor=None (Qwen25VLEncoder property) → 不走 preprocess
  │  │  │  │   → images.permute(0,1,2,5,3,4) 直接用
  │  │  │  ├─ expects_cpu_image_input=True → 图像留在 CPU
  │  │  │  ├─ lang_data.data is List[str] → 走 VLM 联合编码分支
  │  │  │  │   → images_encoder(images, task_instructions)
  │  │  │  │   → Qwen3VLEncoder.forward() → forward_for_dexforcevla()
  │  │  │  │      ├─ prepare_inputs() → processor+tokenize (日志: "Input prep X seconds")
  │  │  │  │      ├─ self.vlm(**inputs, output_hidden_states=True) (日志: "VLM forward X seconds")
  │  │  │  │      ├─ _get_image_hidden_state() → [bs, 288, 2048]
  │  │  │  │      └─ _get_text_hidden_state() → [bs, text_len, 2048]
  │  │  │  └─ ret_image reshape → [bs, hs=1, num_cam=2, 144, 2048]
  │  │  │
  │  │  ├─ prepare_images(batch, "geomap") → GEOMAP not in additional_modality → return None
  │  │  ├─ prepare_state()
  │  │  │  ├─ use_bn=False → 不走 moving_avg
  │  │  │  ├─ masking_gripper=True → gripper 状态位置被置 0
  │  │  │  └─ state_encoder (IdentityStateEncoder) → 直接透传
  │  │  ├─ prepare_affordance() → use_affordance=False → return None, None
  │  │  └─ prepare_vqa() → 无 VQA data → return None
  │  │
  │  ├─ _compute_adaptors(BEFORE_PRIVILEGE)
  │  │  ├─ lang_adaptor: 2048→384
  │  │  ├─ images_adaptor: 2048→384
  │  │  └─ states_adaptor: 256→384
  │  │
  │  ├─ _compute_priviliges() → privileges=[] → 空循环，什么都不做
  │  │
  │  └─ _compute_adaptors(AFTER_PRIVILEGE)
  │     └─ actions_adaptor: 被跳过！
  │        → 日志: "Input type actions for adaptor actions_adaptor not in data keys"
  │        → 因为推理时 data 中没有 actions（actions 是训练时的 GT，推理时不存在）
  │
  ├─ [SAVE_BRAIN_DATA] → 第一次调用时保存 brain_data.pt
  │
  └─ cerebellum_infer()
     ├─ cerebellum.inference()
     │  ├─ 提取条件: state_cond, img_cond, lang_cond, disp_cond=None, mask=None
     │  │  (disp_cond=None 因为 GEOMAP 没编码; mask=None 因为无 privilege)
     │  ├─ _inference_noisy_action()
     │  │  ├─ noisy_action = randn(bs, 32, 128)  ← inference_horizon=32
     │  │  ├─ noise_scheduler.set_timesteps(5)    ← 5 步去噪
     │  │  └─ for t in 5 timesteps:
     │  │     ├─ action_adaptor(action_traj)     ← 256→384
     │  │     ├─ forward_action(...)             ← DiT 8层 前向
     │  │     │   ├─ block_type=DiTBlock → 时间嵌入拼接到输入
     │  │     │   ├─ use_progress=False → 不拼 progress
     │  │     │   ├─ use_affordance=False → 不拼 affordance
     │  │     │   ├─ use_lang=True → lang_cond + pos_embed
     │  │     │   ├─ img_cond + pos_embed
     │  │     │   ├─ use_geomap=False → 不用 disp_cond
     │  │     │   ├─ split_arm_cond=True → 左右臂分别处理
     │  │     │   └─ self.blocks (8层 DiTBlock) 前向
     │  │     └─ noise_scheduler.step() → x_t → x_{t-1}
     │  │
     │  ├─ use_refiner=True → 走精化路径
     │  │  ├─ init_action = noisy_action * mask
     │  │  ├─ init_action_ = cat([sigmoid_gripper(init_action), mask])
     │  │  ├─ full_x = action_adaptor(init_action_)  ← 256→384
     │  │  ├─ has_exp=False → 不走 exp_action_binding
     │  │  └─ refine(full_x, intermediate_output, img_cond, ...)
     │  │     ├─ img_cond + pos_embed
     │  │     ├─ split_arm_cond=True → 左右臂分别处理
     │  │     ├─ for block in self.refiner: x = block(x, ada_ln_cond, c, m)
     │  │     └─ refiner_final_layer → action
     │  │
     │  └─ ret_action = (init_action + refined_action * mask) * mask
     │     → _apply_sigmoid_to_gripper() (eef_loss_type=bce)
     │
     └─ post_process()
        ├─ use_bn=False → 不做 action_moving_avg 反归一化
        ├─ control_mode=ABSOLUTE (output 不含 "relativeqpos") → 不做相对→绝对转换
        └─ return data[ACTIONS]
```

### 6.5 推理时的控制流判定

| 判定点 | 条件 | 结果 | 代码位置 |
|--------|------|------|---------|
| level | config level="full" | 走 brain+cerebellum | dexforcevla_runner.py:predict_action |
| eval_mode | "sync_full" | 走 run_sync_full_evaluation | eval script |
| is_profile | PROFILE_SPEED=0+MEM=0+NVTX=0 | False → 走评估而非 profile | eval script |
| control_mode | output 不含 "relativeqpos" | ABSOLUTE | eval script |
| eef_mode | state_meta 含 "right_eefgripper" | GRIPPER | eval script:infer_eef_mode |
| use_bn | False | 不走 moving_avg | prepare_state, cerebellum_infer |
| use_refiner | True (checkpoint) | 走 refine() | dexdp.py:inference |
| use_affordance | False | 不走 affordance 分支 | from_batch, forward_action |
| use_progress | False (无 privilege) | 不拼 progress token | forward_action |
| use_lang | True (additional_modality 含 lang) | 拼接 lang_cond | forward_action |
| use_geomap | False (additional 不含 geomap) | 不用 disp_cond | forward_action |
| split_arm_cond | True (默认) | 左右臂分别 cross-attn | forward_action, refine |
| masking_gripper | True | gripper 状态置 0 | prepare_state |
| has_exp | False (privileges 空) | 不走 exp_action_binding | inference |
| TORCH_COMPILE | 1 | cerebellum+adaptors 被 compile | compile_model() |
| BRAIN_VLLM | (空) | VLM 走标准 transformers forward | qwen2_5_vl.py |
| expects_cpu_image_input | True (Qwen25VLEncoder) | 图像留 CPU 不预转 GPU | prepare_images |
| image_processor | None (Qwen25VLEncoder) | 不走 preprocess 分支 | prepare_images |

---

## 七、from_batch 逐步骤实测耗时

> 数据来源：容器内加细粒度 profile_push/pop 打点后实跑
> 测试条件：PROFILE_SPEED=1, WARMUP=1, PROFILE_ITERS=100
> 打点方式：torch.cuda.synchronize() + time.perf_counter()（profile_push/pop）；VLM 内部用 time() inline + profile_speed dict

### 7.0 最新状态（2026-08-31，FP32 + TORCH_COMPILE=0 + 深度打点）

> 相比 7.1（2026-08-21 bf16 + TORCH_COMPILE=1 时代），已把打点细化到 VLM 内部
> （prepare_inputs 5 段 + vlm.forward + hidden 提取 3 段），并修复了 memory 统计口径。

**Speed Report（100 iter avg/min/max，2026-09-01 最新实测）**：

```
predict_action.brain_infer : 79.63 / 72.40 / 94.85 ms
├ brain_sub_sum            : 77.72 ms
├ from_batch               : 77.72 ms
│   └ prepare_images       : 75.87 ms   ← 占 brain 95%
│       ├ preprocess       :  0.05 ms
│       ├ to_device        :  0.01 ms
│       ├ encoder          : 74.90 ms   ← VLM forward_for_dexforcevla
│       │   ├ vlm.prepare_inputs : 11.29 ms
│       │   │   ├ to_pil       :  2.89 ms  (CPU, tensor->PIL)
│       │   │   ├ template     :  0.16 ms
│       │   │   ├ vision_info  :  0.17 ms
│       │   │   ├ processor    :  6.45 ms  (CPU, PIL resize+tokenize)
│       │   │   └ to_gpu       :  0.79 ms
│       │   ├ vlm.forward        : 50.00 / 44.32 / 59.48 ms  (self.vlm(**inputs))
│       │   │   ├ vlm_cgm.model  : 49.72 / 44.10 / 59.21 ms  (Qwen3VLModel.forward 整体)
│       │   │   │   ├ vlm_model.embed              :  0.10 ms  (embed_tokens)
│       │   │   │   ├ vlm_model.get_image_features : 29.58 ms  (★ ViT 图像编码, 59%)
│       │   │   │   ├ vlm_model.mask_inject        :  0.41 ms  (masked_scatter 注入)
│       │   │   │   ├ vlm_model.compute_position_ids: 0.84 ms  (M-RoPE 3D)
│       │   │   │   └ vlm_model.language_model     : 18.73 ms  (LLM 前 27 层, 37%)
│       │   │   └ vlm_cgm.lm_head  :  0.03 ms  (lm_head logits)
│       │   ├ vlm.stack_mean     :  0.10 ms
│       │   ├ vlm.get_image_hs   : 13.20 ms  (boolean mask 提取 image token)
│       │   └ vlm.get_text_hs    :  0.51 ms
│       └ postprocess        :  0.02 ms
cerebellum_total            : 28.87 / 19.78 / 43.15 ms
total_inference             : 108.50 ms
```

**关键发现（vlm.forward 内部锚点，2026-09-01 新增）**：
1. **vlm.forward 50.00ms = ViT 29.58ms（59%）+ LLM 18.73ms（37%）+ torch 边界 ~1.4ms**
   - `vlm_model.get_image_features`（ViT）= 29.58ms → **会被 ViT TRT engine 替代**
   - `vlm_model.language_model`（LLM 前 27 层）= 18.73ms → **会被 LLM TRT engine 替代**
   - `embed(0.10) + mask_inject(0.41) + compute_position_ids(0.84) + lm_head(0.03) ≈ 1.38ms` → **engine 边界开销，始终 torch，不可消除**
2. **TRT 化后 vlm.forward 可省 ~48ms**（ViT+LLM engine 替代 48.3ms），剩下 ~1.4ms torch 边界 + engine 本身耗时（实测 TRT 链路 ~17ms）
3. **锚点层级**：vlm_cgm.model（Qwen3VLForCGM 外层）包住 vlm_model.*（Qwen3VLModel 内层），lm_head 在最后
4. **锚点实现**：transformers modeling_qwen3_vl.py 的 Qwen3VLModel.forward / Qwen3VLForConditionalGeneration.forward 加 time() 打点，写入 `_profile_speed`，qwen2_5_vl.py 采集进 profile_speed，runner 合并（vlm_model./vlm_cgm. 前缀）

> 相比 7.1（2026-08-21 bf16 + TORCH_COMPILE=1 时代），已把打点细化到 VLM 内部
> （prepare_inputs 5 段 + vlm.forward 5 段 + hidden 提取 3 段），并修复了 memory 统计口径。

### 7.1 from_batch 逐步骤耗时表（100 iter 均值，稳态）

> 旧版（2026-08-21，bf16 + TORCH_COMPILE=1），保留作对比。最新见 7.0。

| 序号 | 步骤 | 行号 | 耗时(ms) | 占比 | 设备 | 说明 |
|------|------|------|----------|------|------|------|
| 1 | dtype_convert | L896~900 | 0.01 | 0.02% | GPU | batch float tensor → bf16 |
| 2 | augment | L902~928 | ~1 | 1.6% | CPU | VLAAugmentor + to_tensor_for_data_dict |
| 3 | prepare_lang | L930~934 | 0.03 | 0.05% | CPU | 返回 raw instruction，不做编码 |
| 4 | prepare_images | L936~942 | 59.19 | 98.7% | CPU+GPU | VLM 编码，见 4a~4e |
| 4a | └ prepare_inputs | qwen2_5_vl:431~437 | ~5 | 8% | CPU | PIL 转换 + HF processor tokenize |
| 4b | └ VLM forward | qwen2_5_vl:439~444 | ~40 | 66% | GPU | Qwen3-VL-2B 29层前向，output_hidden_states=True |
| 4c | └ stack+mean hidden | qwen2_5_vl:446~453 | ~0.0 | <0.1% | GPU | 多层 hidden state stack + mean |
| 4d | └ _get_image_hidden_state | qwen2_5_vl:455~461 | ~14 | 23% | GPU | boolean mask + fancy indexing 抽取 image token |
| 4e | └ _get_text_hidden_state | qwen2_5_vl:463~469 | ~0.2 | 0.3% | GPU | for 循环找 vision_end/im_end 间 text token |
| 5 | prepare_geomap | L944~950 | 0.01 | 0.02% | - | GEOMAP 不在 additional_modality，直接 return None |
| 6 | prepare_state | L952~954 | 0.10 | 0.17% | CPU | masking_gripper + IdentityStateEncoder 透传 |
| 7 | prepare_affordance | L957~959 | 0.01 | 0.02% | - | batch 无 affordance data，直接 return None |
| 8 | prepare_vqa | L962~964 | 0.01 | 0.02% | - | batch 无 VQA data，直接 return None |
| | **from_batch 总计** | L886~1068 | **59.57** | **100%** | | 100 iter 均值 |

行号说明：L 开头为 dexforcevla_runner.py 行号，qwen2_5_vl: 开头为 qwen2_5_vl.py 行号。

> 注：步骤 4a~4e 的 VLM inline 打点只输出到终端日志，不在 speed_report.json 中。
> 4a 精度到 0.01s（日志输出 "0.00~0.01 seconds"），4b 同理（"0.04 seconds"），
> 4c~4e 精度到 0.0001s。JSON 中仅存 from_batch 整体（59.57ms）和 brain/cerebellum 层打点。
> 步骤 2（augment）在 100 轮稳态时打点被覆盖（profile_speed dict 只存最后一次的值），
> 从 warmup 数据推算约 ~1ms。

### 7.2 关键发现

1. **VLM forward 是绝对瓶颈**：~40ms，占 from_batch 的 65%，占 total_inference(83.74ms) 的 48%
2. **_get_image_hidden_state 意外偏高**：~14ms，占 23%。用 boolean mask + fancy indexing 从 hidden state 中抽取 image token 位置的向量，操作 [1, seq_len, 2048] 的 tensor，开销不小
3. **prepare_inputs（CPU 侧）**：~10ms，占 16%。PIL 转换 + HF processor tokenize，CPU 瓶颈
4. **其余步骤合计 < 1ms**：dtype_convert、prepare_lang、prepare_geomap、prepare_state、prepare_affordance、prepare_vqa 加起来可忽略

### 7.3 优化方向（基于实测数据）

| 优化目标 | 当前耗时 | 可选方案 | 预期收益 |
|----------|----------|----------|----------|
| VLM forward | ~40ms | TRT engine 替换 PyTorch forward（scripts/tllm/ 已有 ViT+LLM engine） | → ~5-10ms |
| _get_image_hidden_state | ~14ms | 预计算 token 位置 index，避免 boolean mask expand + fancy indexing | → ~1-2ms |
| prepare_inputs | ~10ms | 预缓存 tokenize 结果 / 跳过 PIL 转 RGB / 用 CUDA 图像处理 | → ~2-5ms |

### 7.6 prepare_inputs GPU 化优化（2026-08-31，进行中）

> 代码：`hpc_opt/trtllm/prepare_inputs/gpu_prepare_inputs.py` + 验证 `verify_gpu_prepare5.py`
> 核心结论：**prepare_inputs 的 11.29ms 里 9.3ms（82%）是 CPU 图像处理**（to_pil 2.89 + processor 6.45），
> 全部可 GPU 化，预计省 ~8ms。

**prepare_inputs 内部耗时剖析（profile 打点实测）**：

| 子模块 | 耗时 | 占比 | 设备 | 优化 |
|--------|------|------|------|------|
| processor（PIL resize + tokenize） | 6.45ms | 57% | CPU | **GPU resize/normalize** |
| to_pil（tensor→PIL） | 2.89ms | 26% | CPU | **跳过（直接 GPU tensor）** |
| to_gpu | 0.79ms | 7% | GPU | 保留 |
| vision_info | 0.17ms | 1.5% | CPU | 保留（小） |
| template | 0.16ms | 1.4% | CPU | 保留（小） |

**GPU 化方案（已实现 + 验证）**：
1. **图像不走 to_pil_image**——以 GPU tensor 直接处理
2. `smart_resize`（对齐 processor，factor=patch_size*merge_size=16*2=32）
3. `tvF.resize`（GPU，BICUBIC + antialias）——实测 CPU/GPU max_diff=0.0
4. `rescale_normalize`（GPU）——实测与 processor 一致 1.19e-7
5. `patch_merge`（view/permute/reshape，GPU）——对齐 processor
6. 文本部分（apply_chat_template + tokenize）**保留 CPU**（文本量小且依赖可变指令）

**关键参数**（Qwen3-VL-2B）：
- patch_size=16, merge_size=2, temporal_patch_size=2 → patch_dim = 3×2×16×16 = 1536
- smart_resize factor = 16×2 = 32
- 图像固定 2×384×1152（真实输入，4 相机 6 帧历史中的 2 相机）

**验证结果**（verify_gpu_prepare5.py，随机图 384×1152 BGR）：
```
原始 processor 输出: [1728, 1536]  gthw [1,24,72]
GPU 化输出:        [1728, 1536]  gthw [1,24,72]
max_diff = 1.19e-7（float32 舍入级）✅ 数值等价
```

**进度（2026-08-31）**：✅ **已完成**（图像 GPU 化 + hook 集成 + 对齐验证，详见 task1 4.6.1）：
- hook 集成：`trt_infer_shim.py` 对 Qwen25VLEncoder.self 装 `install_prepare_inputs_gpu_hook`（代理 image_processor）
- 单元对齐：input_ids/attention_mask/grid_thw max_diff=0.0，pixel_values 1.19e-7
- 端到端：真实链路 infer_trt.sh 验证，Input preparation 0.00s，VLM forward 0.01s，EXIT=0
- 性能：prepare_inputs **11.29ms → ~3.5ms**（中位数 0ms，加速 ~4 倍）
- ⏳ 剩余：延迟隐藏（overlap 图像处理与 LLM 推理）

**重要前提**：文本和图像**每帧都变**，所以**无"值"级预计算**（input_ids/position_ids/cos/sin/pixel_values 全依赖每帧内容）；可预计算的只有静态结构（engine IO buffer 复用 + CUDA graph 捕获）。

### 7.4 打点代码位置

已在容器内两个文件中加入打点：

| 文件 | 打点方式 | 新增 key |
|------|----------|----------|
| dexforcevla_runner.py | profile_push/pop (cuda.synchronize) | from_batch.dtype_convert, from_batch.augment, from_batch.prepare_lang, from_batch.prepare_images, from_batch.prepare_geomap, from_batch.prepare_state, from_batch.prepare_affordance, from_batch.prepare_vqa |
| qwen2_5_vl.py | time() inline log_info | VLM stack+mean hidden states time, VLM _get_image_hidden_state time, VLM _get_text_hidden_state time |

原文件备份：`.bak` 后缀，同目录。

### 7.5 整体推理耗时汇总（100 iter 均值，同次测试）

| 指标 | 耗时(ms) | 说明 |
|------|----------|------|
| model_init | 10286.08 | 模型加载+compile，仅一次 |
| data_prep | 30.26 | 预取数据到 GPU |
| brain_infer | 61.31 | (from_batch 59.57 + adaptors 0.57 + priviliges 0.01 + sub_sum差值 ~1.16) |
| cerebellum_infer | 20.18 | (inference 20.04 + post_process 0.02) |
| total_inference | 81.50 | brain + cerebellum |
| total_with_dataprep | 111.75 | 含数据准备 |

### 7.7 TrtEngine 输出 buffer 复用优化（2026-09-01）

> 详细见 task1 4.6.3。这里记录 TRT 链路的 engine 调用开销分项实测。

**TrtEngine.__call__ 分项耗时（实测，每次推理）**：

| 引擎 | 完整 __call__ | 输入处理 | set shape+addr | 输出分配 | 输出 set_addr | execute_v2 | sync |
|------|--------------|---------|---------------|---------|--------------|-----------|------|
| ViT | 4.43ms | 0.38ms | 0.43ms | **0.46ms** | **0.46ms** | 2.70ms | 0.003ms |
| LLM | 5.20ms | 0.02ms | 0.04ms | 0.04ms | 0.04ms | 5.06ms | 0.003ms |

**结论**：
- **ViT 引擎 21% 的开销（0.92ms）在输出分配 + 地址设置**——可复用消除
- LLM 引擎 97% 在 execute_v2（真实计算），优化空间小
- synchronize 几乎无开销（0.003ms），不是瓶颈

**优化**：输出 buffer 预分配复用（`reuse_output_buffers` 参数 + `TRT_REUSE_OUTPUT` 环境变量，默认开），收益 ViT 0.39ms + LLM 0.32ms ≈ **0.7ms**。shape/dtype 变化自动重分配；跨调用保存输出引用需关开关。

### 7.8 get_image_hs 预计算 index 优化（2026-09-01）

> 详细见 task1 4.6.4。`vlm.get_image_hs`（13.2ms）不是模型推理，是**后处理**——从 final_hidden_state 提取 image token 特征。

**性质**：GPU 操作（非 CPU），全在 cuda tensor 上。慢在 **boolean mask expand + fancy indexing**（不规则 gather，内存密集），对比 text 提取用连续切片（0.5ms 快）。

**关键发现（实测）**：image token 位置 index **不依赖图像内容**（只依赖图像数量，配置固定 2 图 → 288 个占位符）。即使文本变（seq_len 316/309/305），image token 位置都固定 [4..293]。

**优化**：预计算 index + `final_hidden_state[:, image_idx, :]`（index_select 连续 gather），预计 13.2ms → ~1-2ms。前提图像数量固定。

**实现**（已集成，`get_image_hidden_state/gpu_get_image_hs.py` + shim 安装）：
- index 首次从 input_ids 计算后缓存，每次调用校验 image token 数量（一致复用，变化重算）
- 单元验证：输出 max_diff=0.0，算法 0.052 → 0.037ms（1.4x）
- 端到端：get_image_hs 13.20 → 3.71ms（省 72%）

**⚠️ 根因分析**：TRT 链路耗时降低**主要是测量/等待机制**：
- `vlm.get_image_hs` 用 CPU time() 计时，GPU kernel 异步，`_get_image_hidden_state` 首操作隐式等待前面 vlm.forward kernel，等待计入计时
- torch 链路 vlm.forward 50ms（GPU 队列堆满）→ 等待大（13.2ms 多是不该计入的等待）
- TRT 链路 vlm.forward 13ms + engine execute_v2 自带 synchronize → 等待小（3.71ms）
- 独立测算法：torch/TRT 输入都是 **0.040ms**（几乎一样）——真实算法开销极小，预计算真实收益 1.4x 非 9.5ms

---

## 八、文件索引

### 核心文件

| 文件路径 | 功能 |
|----------|------|
| scripts/infer_realdata.sh | 启动脚本 |
| tests/models/test_eval_models_by_realdata_profile.py | 测试入口 |
| dexechain/evaluation/vla/dexforcevla_sim_by_realdata_profile.py | 评估/profile 主逻辑 |
| dexechain/agents/dexforce_vla/models/dexforcevla_runner.py | DexForceVLA 模型主体 |
| dexechain/agents/dexforce_vla/train/build_model.py | 模型构建 |
| dexechain/agents/dexforce_vla/config.py | 配置加载 |
| dexechain/agents/dexforce_vla/models/multimodal_encoders/vlms/qwen3_vl.py | Qwen3-VL 编码器 |
| dexechain/agents/dexforce_vla/models/multimodal_encoders/vlms/qwen2_5_vl.py | Qwen2.5-VL 基类 (forward 实现) |
| dexechain/agents/dexforce_vla/models/policy/dexdp.py | DexDiffusionPolicy (扩散策略) |
| dexechain/agents/dexforce_vla/models/policy/dexfm.py | DexFlowMatchingPolicy |
| dexechain/agents/dexforce_vla/models/policy/dexdp_blocks.py | DiT blocks + forward_action + refine |
| dexechain/agents/dexforce_vla/models/utils.py | post_process 等工具函数 |
| dexechain/utils/profile.py | ProfileConfig, nvtx, 计时工具 |
| dexechain/agents/dexforce_vla/train/vla_dataset.py | VLAConsumerDataset 数据集 |

### TRT 加速相关 (scripts/tllm/)

| 文件路径 | 功能 |
|----------|------|
| scripts/tllm/01_extract_llm.py + .sh | 从 HF 模型提取 LLM 权重 |
| scripts/tllm/02_cvt_llm_safetensor.py + .sh | 转换为 TRT-LLM 格式 |
| scripts/tllm/02_export_vit_onnx.py + .sh | 导出 ViT ONNX |
| scripts/tllm/03_cvt_llm_trt.sh | 构建 LLM TRT engine |
| scripts/tllm/03_cvt_vit_trt.py + .sh | 构建 ViT TRT engine |
| scripts/tllm/profile_speed_vit_llm.py | TRT ViT+LLM 性能测试 |
| scripts/tllm/export_qwen3vl_onnx.py | Qwen3-VL ONNX 导出 |
| scripts/tllm/inspect_trt_engine.py | TRT engine 检查 |
