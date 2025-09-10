#!/bin/bash



# Activate virtual environment
eval "$(conda shell.bash hook)"
# conda activate qwenvl
# conda activate internvl3_5
conda activate internvl

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment internvl3_5 activated."

# export HF_HOME=/mnt/SSD4/prateik
export HF_HOME=/mnt/SSD3/lina/SeizureSemiologyBench/cache

# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python internvl35_32B.py \
    --gpu 1,3 \
    --tp 2 \
    --videos_range 1-1000 \
    --output_dir /mnt/SSD1/prateik/icassp_vlm/output \
    --model_name OpenGVLab/InternVL3_5-38B \
    --dataset_dir /mnt/SSD3/tengyou/seizure_videos/segments/all_dataset \
    --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache 

echo "Done!"
