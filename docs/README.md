# DDPO documentations

## Installation
Refer to the official documentation of verl0.5.0: https://verl.readthedocs.io/en/v0.5.x/start/install.html
```bash
conda create -n verl python==3.10
conda activate verl

# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install verl
cd verl
pip install --no-deps -e .
```

## Train

```bash
bash DDPO/examples/grpo_trainer/qwen3_accum_n_10_val_16.sh
```

