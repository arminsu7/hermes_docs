# Qwen3-ASR-1.7B INT4 量化复现指南

从零开始，逐行可执行。所有命令在容器 `smr2508` 内运行。

## 前置条件

- 容器 `smr2508` 已运行（镜像 `nvcr.io/nvidia/pytorch:25.08-py3`）
- 原始模型在 `/root/repos/llm/model/qwen3-asr-1.7B/`
- venv 脚本: `/root/repos/hermes/scripts/activate_llmc.sh`

---

## 第一步：进入容器，激活环境，安装 qwen-asr

```bash
docker exec -it smr2508 bash
source /root/repos/hermes/scripts/activate_llmc.sh

# 验证环境
python --version                     # 3.12.3
python -c "import llmcompressor; print(llmcompressor.__version__)"  # 0.12.0
python -c "import torch; print(torch.cuda.get_device_name(0))"      # RTX 3060

# 安装 qwen-asr（会降级 transformers），再恢复
pip install qwen-asr
pip install "transformers>=5.9.0,<=5.10.1" "accelerate==1.13.0"
```

---

## 第二步：修复 6 个兼容性问题

全部在容器内逐段执行。

### 修复 1: check_model_inputs 装饰器 (坑2)

> 文件: `modeling_qwen3_asr.py` 第 1009 行

```bash
sed -i 's/@check_model_inputs()/@check_model_inputs/g' \
  /root/repos/hermes/llmcompressor_env/lib/python3.12/site-packages/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py
```

### 修复 2: thinker_config 初始化时序 (坑1bis)

> 文件: `configuration_qwen3_asr.py`

```bash
python3 << 'PYEOF'
path = "/root/repos/hermes/llmcompressor_env/lib/python3.12/site-packages/qwen_asr/core/transformers_backend/configuration_qwen3_asr.py"
with open(path) as f:
    c = f.read()

# 2a: super().__init__() 之前预置 thinker_config = None
c = c.replace(
    "    ):\n        super().__init__(**kwargs)\n        if thinker_config is None:",
    "    ):\n        object.__setattr__(self, \"thinker_config\", None)\n        super().__init__(**kwargs)\n        if thinker_config is None:"
)

# 2b: get_text_config 加 None 检查
c = c.replace(
    "return self.thinker_config.get_text_config()",
    "if self.thinker_config is not None:\n            return self.thinker_config.get_text_config()\n        return super().get_text_config(decoder=decoder)"
)

with open(path, "w") as f:
    f.write(c)
print("Done")
PYEOF
```

### 修复 3: rope_type fallback + compute_default_rope_parameters (坑3+坑5)

> 文件: `modeling_qwen3_asr.py`

```bash
python3 << 'PYEOF'
path = "/root/repos/hermes/llmcompressor_env/lib/python3.12/site-packages/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py"
with open(path) as f:
    c = f.read()

# 3a: 默认值 default -> llama3
c = c.replace(
    'self.rope_type = config.rope_scaling.get("rope_type", "default")',
    'self.rope_type = config.rope_scaling.get("rope_type", "llama3")'
)

# 3b: ROPE_INIT_FUNCTIONS fallback
old_block = """        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)"""
new_block = """        self.config = config
        if self.rope_type in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        else:
            self.rope_init_fn = None
            self.attention_scaling = 1.0
            base = self.config.rope_theta
            dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))"""
c = c.replace(old_block, new_block)

# 3c: 添加 compute_default_rope_parameters 静态方法
old_method = """        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):"""
new_method = """        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    def apply_interleaved_mrope(self, freqs, mrope_section):"""
c = c.replace(old_method, new_method)

with open(path, "w") as f:
    f.write(c)
print("Done")
PYEOF
```

### 修复 4: pad_token_id 安全访问 (坑4)

> 文件: `modeling_qwen3_asr.py` 第 1112 行

```bash
sed -i 's/self.config.pad_token_id if self.config.pad_token_id is not None else -1/getattr(self.config, "pad_token_id", None) if getattr(self.config, "pad_token_id", None) is not None else -1/' \
  /root/repos/hermes/llmcompressor_env/lib/python3.12/site-packages/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py
```

### 修复 5: generation_config.json temperature 冲突 (坑7)

```bash
python3 << 'PYEOF'
import json
with open("/root/repos/llm/model/qwen3-asr-1.7B/generation_config.json") as f:
    gc = json.load(f)
if gc.get("temperature") is not None and not gc.get("do_sample", False):
    del gc["temperature"]
with open("/root/repos/llm/model/qwen3-asr-1.7B/generation_config.json", "w") as f:
    json.dump(gc, f, indent=2)
print("Done")
PYEOF
```

### 验证: 加载模型

```bash
python << 'PYEOF'
import torch
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
m = Qwen3ASRForConditionalGeneration.from_pretrained(
    "/root/repos/llm/model/qwen3-asr-1.7B",
    torch_dtype=torch.float16, device_map="cpu"
)
print("OK:", type(m).__name__, sum(p.numel() for p in m.parameters()), "params")
PYEOF
```

预期: `OK: Qwen3ASRForConditionalGeneration 2349217408 params`

---

## 第三步：生成量化脚本并执行

```bash
mkdir -p /root/repos/hermes/docs/code_asr_awq
```

脚本 `quantize_awq.py` 和 `save_quant.py` 已存在于该目录，可直接使用:

```bash
cd /root/repos/hermes/docs/code_asr_awq

# 执行量化
python quantize_awq.py

# 如果 save_pretrained 崩溃，用备选方案:
python save_quant.py
```

预期输出:
```
Original: 4.70 GB
Quantized: 2.17 GB
Ratio: 2.2x
```

---

## 第四步：验证量化模型

```bash
python << 'PYEOF'
import torch
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
m = Qwen3ASRForConditionalGeneration.from_pretrained(
    "/root/repos/llm/model/qwen3-asr-1.7B-awq",
    torch_dtype=torch.float16, device_map="cpu"
)
print("OK:", type(m).__name__, sum(p.numel() for p in m.parameters()), "params")
PYEOF
```

---

## 修复汇总

| 修复 | 文件 | 行号 | 操作 |
|------|------|------|------|
| check_model_inputs | modeling_qwen3_asr.py | 1009 | 去掉括号 |
| thinker_config 时序 | configuration_qwen3_asr.py | 402,423-425 | 预置属性 + None 检查 |
| rope_type fallback | modeling_qwen3_asr.py | 785-804 | 改默认值 + fallback |
| compute_default_rope | modeling_qwen3_asr.py | 807-818 | 新增静态方法 |
| pad_token_id | modeling_qwen3_asr.py | 1112 | getattr 安全访问 |
| generation_config | generation_config.json | - | 删除 temperature |

## 输出文件

```
/root/repos/hermes/docs/code_asr_awq/
├── README.md          # 完整踩坑详解
├── REPRODUCE.md       # 本文件
├── quantize_awq.py    # 量化脚本
└── save_quant.py      # 保存脚本

/root/repos/llm/model/
├── qwen3-asr-1.7B/        # 原始 (4.70 GB)
└── qwen3-asr-1.7B-awq/    # 量化 (2.17 GB)
```
