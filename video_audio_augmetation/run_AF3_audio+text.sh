#!/bin/bash

# ========== Config ==========
CONDA_ENV="af3_env"
SCRIPT="task1_Audio_Flamingo-3.py"

# dataset
DATASET_DIR="/mnt/SSD1/prateik/seizure_audition/data/audio"
TEXT_DIR="/mnt/SSD1/prateik/seizure_audition/data/text"
OUTPUT_DIR="/mnt/SSD3/chonghan/alm_results_af3_audio_text"
CACHE_DIR="/mnt/SSD3/chonghan/hf_cache"

# model weights
export MODEL_DIR="/mnt/SSD3/chonghan/hf_cache/models--nvidia--audio-flamingo-3/snapshots/8327ed61bfe7d635fc95f978750a8d1b6f4e7f5e"

# runtime params
GPU=0
MAX_GROUPS=-1
MAX_NEW_TOKENS=512
MAX_RETRIES=2
DISABLE_LOGS=true   # set to false if you want raw logs
TEXT_SUFFIX=".txt"
MAX_TXT_CHARS=5000
# ============================


# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

if [ $? -ne 0 ]; then
  echo "[ERR] Failed to activate conda environment: $CONDA_ENV"
  exit 1
fi
echo "[INFO] Conda environment $CONDA_ENV activated."


# Run inference
python $SCRIPT \
  --gpu $GPU \
  --dataset_dir $DATASET_DIR \
  --text_dir $TEXT_DIR \
  --output_dir $OUTPUT_DIR \
  --cache_dir $CACHE_DIR \
  --max_groups $MAX_GROUPS \
  --max_new_tokens $MAX_NEW_TOKENS \
  --max_retries $MAX_RETRIES \
  --disable_logs $DISABLE_LOGS \
  --text_suffix $TEXT_SUFFIX \
  --max_txt_chars $MAX_TXT_CHARS

echo "[DONE] Results saved to: $OUTPUT_DIR"
