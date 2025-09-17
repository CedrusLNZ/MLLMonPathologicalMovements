
#!/bin/bash

# Follow instructions on Sapiens repo to set up Sapiens Lite inference

############################## DECLARE VARIABLES ######################################
export SAPIENS_ROOT="/mnt/SSD1/prateik/sapiens"
export SAPIENS_LITE_ROOT="$SAPIENS_ROOT/lite"
export SAPIENS_LITE_CHECKPOINT_ROOT="/mnt/SSD1/prateik/sapiens/sapiens_lite_host/torchscript"
export TMPDIR="/mnt/SSD1/prateik/tmp"

PROCESSING_SCRIPT_POSE="/mnt/SSD1/prateik/sapiens/lite/scripts/demo/torchscript/pose_keypoints308.sh"
PROCESSING_SCRIPT_SEG="/mnt/SSD1/prateik/sapiens/lite/scripts/demo/torchscript/seg.sh"
PROCESSING_SCRIPT_NORMAL="/mnt/SSD1/prateik/sapiens/lite/scripts/demo/torchscript/normal.sh"
PROCESSING_SCRIPT_DEPTH="/mnt/SSD1/prateik/sapiens/lite/scripts/demo/torchscript/depth.sh"

EXTRACT_FRAMES=false
FRAMES_DIR=/mnt/SSD1/prateik/frames # ignored if EXTRACT_FRAMES=true
DELETE_INTERMEDIATE=false
FPS=10
NUM_WORKERS=100
export SAPIENS_BATCH_SIZE=42

INPUT_VIDEO_DIR="$1"
OUTPUT_VIDEO_DIR="$2"
#######################################################################################


############################### CREATE DIRS ######################################
mkdir -p "$OUTPUT_VIDEO_DIR"

# Create intermediate directories

echo "======================= VIDEO PROCESSING PIPELINE ======================="
echo "Input videos directory: $INPUT_VIDEO_DIR"
echo "Output videos directory: $OUTPUT_VIDEO_DIR"
echo "Processing script: $PROCESSING_SCRIPT"
echo "Processed frames directory: $PROCESSED_FRAMES_DIR"
echo "Output FPS: $FPS"
echo "Delete intermediate directories: $DELETE_INTERMEDIATE"
echo "====================================================================="
#################################################################################


############################### EXTRACT ALL FRAMES ######################################
if ["$EXTRACT_FRAMES" = true]; then
    FRAMES_DIR="$(mktemp -d -t frames_XXXXX)"
    echo
    echo "Step 1: Converting videos to frames..."
    python /mnt/SSD1/prateik/sapiens/video_processing/video_to_frames_parallel.py "$INPUT_VIDEO_DIR" "$FRAMES_DIR" --fps "$FPS" --workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract frames from videos" >&2
        exit 1
    fi
    python /mnt/SSD1/prateik/sapiens/video_processing/organize_jpgs.py "$FRAMES_DIR"
fi
#########################################################################################

PARENT_FRAMES_DIR="$FRAMES_DIR"
for FRAMES_DIR in "$PARENT_FRAMES_DIR"/*/; do
    if [ -d "$FRAMES_DIR" ]; then

        # Check if this directory was already processed
        NUM_INPUT_FRAMES=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
        EXISTING_DIR=$(ls -d "${TMPDIR:-/tmp}"/processed_frames_pose_"$(basename "$FRAMES_DIR")"_*/ 2>/dev/null | head -n 1)
        
        if [ -n "$EXISTING_DIR" ] && [ -d "$EXISTING_DIR/sapiens_1b" ]; then
            NUM_PROCESSED_FRAMES=$(ls "$EXISTING_DIR/sapiens_1b"/*.jpg 2>/dev/null | wc -l)
            if [ "$NUM_INPUT_FRAMES" -eq "$NUM_PROCESSED_FRAMES" ]; then
                echo "Directory $FRAMES_DIR was already processed, skipping..."
                continue
            fi
        fi

        echo "Processing frames in: $FRAMES_DIR"

        ############################### POSE ESTIMATION ######################################
        PROCESSED_FRAMES_DIR="$(mktemp -d -t processed_frames_pose_$(basename "$FRAMES_DIR")_XXXXX)"
        echo
        echo "Step 2: Pose Estimation"
        bash "$PROCESSING_SCRIPT_POSE" "$FRAMES_DIR" "$PROCESSED_FRAMES_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: Frame processing failed" >&2
            exit 1
        fi
        PROCESSED_FRAMES_CHILD_DIR=$(ls -d "$PROCESSED_FRAMES_DIR"/*/ 2>/dev/null | head -n 1)

        # Convert processed frames back to videos
        echo
        echo "Converting processed frames back to videos..."
        python /mnt/SSD1/prateik/sapiens/video_processing/frames_to_video.py "$PROCESSED_FRAMES_CHILD_DIR" "$OUTPUT_VIDEO_DIR/pose" --fps "$FPS"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to convert frames to videos" >&2
            exit 1
        fi

        # Clean up intermediate directories if requested
        if [ "$DELETE_INTERMEDIATE" = true ]; then
            echo
            echo "Cleaning up intermediate directories..."
            # echo "Removing $FRAMES_DIR"
            # rm -rf "$FRAMES_DIR"
            echo "Removing $PROCESSED_FRAMES_DIR"
            rm -rf "$PROCESSED_FRAMES_DIR"
            echo "Cleanup complete"
        fi
        ######################################################################################


        # ############################### SEGMENTATION ######################################
        # PROCESSED_FRAMES_DIR_SEG="$(mktemp -d -t processed_frames_XXXXXX)"
        # echo
        # echo "Step 3: Segmentation"
        # bash "$PROCESSING_SCRIPT_SEG" "$FRAMES_DIR" "$PROCESSED_FRAMES_DIR_SEG"
        # if [ $? -ne 0 ]; then
        #     echo "Error: Frame processing failed" >&2
        #     exit 1
        # fi

        # SEG_FRAMES_DIR=$(ls -d "$PROCESSED_FRAMES_DIR_SEG"/*/ 2>/dev/null | head -n 1)

        # echo
        # echo "Converting processed frames back to videos..."
        # python /mnt/SSD1/prateik/sapiens/video_processing/frames_to_video.py "$SEG_FRAMES_DIR" "$OUTPUT_VIDEO_DIR/seg" --fps "$FPS"
        # if [ $? -ne 0 ]; then
        #     echo "Error: Failed to convert frames to videos" >&2
        #     exit 1
        # fi

        # ######################################################################################


        # ############################### NORMAL ######################################
        # PROCESSED_FRAMES_DIR="$(mktemp -d -t processed_frames_XXXXXX)"
        # echo
        # echo "Step 4: Surface Normal"
        # bash "$PROCESSING_SCRIPT_NORMAL" "$FRAMES_DIR" "$PROCESSED_FRAMES_DIR" "$SEG_FRAMES_DIR"
        # if [ $? -ne 0 ]; then
        #     echo "Error: Frame processing failed" >&2
        #     exit 1
        # fi
        # PROCESSED_FRAMES_CHILD_DIR=$(ls -d "$PROCESSED_FRAMES_DIR"/*/ 2>/dev/null | head -n 1)

        # echo
        # echo "Converting processed frames back to videos..."
        # python /mnt/SSD1/prateik/sapiens/video_processing/frames_to_video.py "$PROCESSED_FRAMES_CHILD_DIR" "$OUTPUT_VIDEO_DIR/normal" --fps "$FPS"
        # if [ $? -ne 0 ]; then
        #     echo "Error: Failed to convert frames to videos" >&2
        #     exit 1
        # fi

        # if [ "$DELETE_INTERMEDIATE" = true ]; then
        #     echo
        #     echo "Cleaning up intermediate directories..."
        #     # echo "Removing $FRAMES_DIR"
        #     # rm -rf "$FRAMES_DIR"
        #     echo "Removing $PROCESSED_FRAMES_DIR"
        #     rm -rf "$PROCESSED_FRAMES_DIR"
        #     echo "Cleanup complete"
        # fi
        # ######################################################################################

        # ############################### DEPTH ######################################
        # PROCESSED_FRAMES_DIR="$(mktemp -d -t processed_frames_XXXXXX)"
        # echo
        # echo "Step 5: Depth"
        # bash "$PROCESSING_SCRIPT_NORMAL" "$FRAMES_DIR" "$PROCESSED_FRAMES_DIR" "$SEG_FRAMES_DIR"
        # if [ $? -ne 0 ]; then
        #     echo "Error: Frame processing failed" >&2
        #     exit 1
        # fi
        # PROCESSED_FRAMES_CHILD_DIR=$(ls -d "$PROCESSED_FRAMES_DIR"/*/ 2>/dev/null | head -n 1)

        # echo
        # echo "Converting processed frames back to videos..."
        # python /mnt/SSD1/prateik/sapiens/video_processing/frames_to_video.py "$PROCESSED_FRAMES_CHILD_DIR" "$OUTPUT_VIDEO_DIR/depth" --fps "$FPS"
        # if [ $? -ne 0 ]; then
        #     echo "Error: Failed to convert frames to videos" >&2
        #     exit 1
        # fi

        # if [ "$DELETE_INTERMEDIATE" = true ]; then
        #     echo
        #     echo "Cleaning up intermediate directories..."
        #     # echo "Removing $FRAMES_DIR"
        #     # rm -rf "$FRAMES_DIR"
        #     echo "Removing $PROCESSED_FRAMES_DIR"
        #     rm -rf "$PROCESSED_FRAMES_DIR"
        #     echo "Removing $PROCESSED_FRAMES_DIR_SEG"
        #     rm -rf "$PROCESSED_FRAMES_DIR_SEG"
        #     echo "Cleanup complete"
        # fi
        # ######################################################################################
    fi
done

echo
echo "====================================================================="
echo "Processing complete!"
echo "Processed videos saved to: $OUTPUT_VIDEO_DIR"
echo "====================================================================="
