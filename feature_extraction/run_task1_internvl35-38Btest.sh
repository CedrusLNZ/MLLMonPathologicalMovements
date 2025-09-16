#!/bin/bash



# Activate virtual environment
eval "$(conda shell.bash hook)"
# conda activate qwenvl
conda activate internvl3_5

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment internvl3_5 activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python internvl3.5_8B.py \
    --gpu 2,3 \
    --tp 2 \
    --videos_range 1-100 \
    --output_dir /home/lina/SeizureSemiologyBench/output \
    --model_name OpenGVLab/InternVL3_5-38B \
    --dataset_dir /mnt/SSD3/lina/ucla2/3videos_in90 \
    --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache 

echo "Done!"
