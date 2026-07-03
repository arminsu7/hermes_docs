#!/usr/bin/env python3
"""
保存量化后的 Qwen3-ASR-1.7B 模型（绕过 save_pretrained 的 _tied_weights_keys 问题）

使用 safetensors 直接保存 state_dict，手动复制 config/tokenizer 等辅助文件。
"""

import os, sys, json, shutil
import torch
from safetensors.torch import save_file
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
from transformers import AutoTokenizer
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, initialize_module_for_quantization,
)
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization.lifecycle.forward import compute_dynamic_scales_and_zp

MODEL_PATH = "/root/repos/llm/model/qwen3-asr-1.7B"
OUTPUT_DIR = "/root/repos/llm/model/qwen3-asr-1.7B-int4-weight-only"
IGNORE = ["embed", "lm_head", "audio_tower", "feature_extractor", "norm", "rotary", "conv"]

print("=" * 60)
print("Loading model...")
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map="auto" if torch.cuda.is_available() else "cpu",
)
model.eval()

print("Applying quantization scheme...")
weight_args = QuantizationArgs(
    num_bits=4, type="int", symmetric=True,
    group_size=128, strategy="group",
)
scheme = QuantizationScheme(
    targets=["Linear"], weights=weight_args, input_activations=None,
)

for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and not any(p in name for p in IGNORE):
        mod.quantization_scheme = scheme
        initialize_module_for_quantization(mod, scheme)

print("Computing scales from weights...")
for name, mod in model.named_modules():
    if hasattr(mod, "weight_scale") and hasattr(mod, "quantization_scheme"):
        scale, zp = compute_dynamic_scales_and_zp(
            mod.weight, mod.quantization_scheme.weights, mod
        )
        mod.weight_scale.data = scale.to(mod.weight_scale.dtype)
        if hasattr(mod, "weight_zero_point") and zp is not None:
            mod.weight_zero_point.data = zp.to(mod.weight_zero_point.dtype)
print("Scale computation done.")
print("Compressing weights...")
compressor = ModelCompressor.from_pretrained_model(model)
compressor.compress_model(model)

print(f"Saving to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绕过 save_pretrained -- 直接用 safetensors 保存
state_dict = {}
for name, param in model.state_dict().items():
    state_dict[name] = param.contiguous().cpu()

safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
save_file(state_dict, safetensors_path)
print(f"  Saved {len(state_dict)} tensors -> {safetensors_path}")

# 复制辅助文件
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

for fn in ["config.json", "generation_config.json", "preprocessor_config.json",
           "merges.txt", "vocab.json", "chat_template.json", "tokenizer_config.json",
           "configuration.json", "README.md"]:
    src = os.path.join(MODEL_PATH, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUTPUT_DIR, fn))
        print(f"  Copied {fn}")

# 量化配置
with open(os.path.join(OUTPUT_DIR, "quantize_config.json"), "w") as f:
    json.dump({
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "group_size": 128,
                    "symmetric": True,
                    "strategy": "group"
                },
                "input_activations": None
            }
        }
    }, f, indent=2)

# 报告大小
osize = sum(os.path.getsize(os.path.join(MODEL_PATH, f))
            for f in os.listdir(MODEL_PATH) if os.path.isfile(os.path.join(MODEL_PATH, f)))
qsize = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f))
            for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f)))
print(f"\nOriginal: {osize/1e9:.2f} GB")
print(f"Quantized: {qsize/1e9:.2f} GB")
print(f"Ratio: {osize/qsize:.1f}x")
print("DONE!")
