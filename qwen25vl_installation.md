# Environment Installation (for Qwen2.5-VL script)

```shell
conda create -n qwenvl python=3.10 -y
conda activate qwenvl

# CUDA 12.4
pip install torch torchvision torchaudio torchcodec \
  transformers==4.51.3 accelerate qwen-vl-utils pandas peft tqdm numpy scipy datasets deepspeed \
  --extra-index-url https://download.pytorch.org/whl/cu124

# CUDA 12.2 (use cu121 wheel, cu122 may fail)
pip install torch torchvision torchaudio torchcodec \
  transformers==4.51.3 accelerate qwen-vl-utils pandas peft tqdm numpy scipy datasets deepspeed \
  --extra-index-url https://download.pytorch.org/whl/cu121


# FlashAttention Installation (ABI Compatibility Fix)
pip install --upgrade setuptools wheel && \
pip uninstall -y flash-attn || true && \
pip cache purge && \
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation --no-cache-dir
```
