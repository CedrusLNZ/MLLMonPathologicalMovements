import os
import argparse
import cv2
from pathlib import Path
import math

def extract_frames(input_folder, output_folder, fps=None):
    """
    Extract frames from MP4 files in the input folder and save them as JPEGs in the output folder.
    
    Args:
        input_folder (str): Path to folder containing MP4 files
        output_folder (str): Path to folder where extracted frames will be saved
        fps (float, optional): Target frames per second to extract. If None, extract all frames.
            If specified, frames will be sampled uniformly to achieve this rate.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all MP4 files in the input directory
    mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {input_folder}")
        return
    
    for video_file in mp4_files:
        # Get base name without extension (e.g., "a" from "a.mp4")
        base_name = os.path.splitext(video_file)[0]
        
        # Full path to the video file
        video_path = os.path.join(input_folder, video_file)
        
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"Error opening video file {video_file}")
            continue
        
        # Get video properties
        video_fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"Processing {video_file}...")
        print(f"  Original video: {total_frames} frames, {video_fps:.2f} FPS, {duration:.2f} seconds")
        
        if fps is None:
            # Extract all frames
            frame_indices = range(total_frames)
        else:
            # Calculate how many frames we should extract
            target_frame_count = math.ceil(duration * fps)
            
            if target_frame_count > total_frames:
                raise ValueError(f"Cannot extract at {fps} FPS: video {video_file} has only {video_fps:.2f} FPS ({total_frames} frames)")
            
            # Sample frames uniformly
            if target_frame_count == total_frames:
                frame_indices = range(total_frames)
            else:
                # Calculate frame step size to achieve target FPS
                step = total_frames / target_frame_count
                frame_indices = [int(i * step) for i in range(target_frame_count)]
        
        frame_count = 0
        extracted_count = 0
        
        print(f"  Extracting {len(frame_indices)} frames {'at ' + str(fps) + ' FPS' if fps else '(all frames)'}...")
        
        for frame_idx in frame_indices:
            # Set frame position
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read the frame
            success, frame = video.read()
            
            if success:
                # Create filename with format: base_name_0000001.jpg
                frame_filename = f"{base_name}_{extracted_count:07d}.jpg"
                output_path = os.path.join(output_folder, frame_filename)
                
                # Save the frame as JPEG
                cv2.imwrite(output_path, frame)
                
                extracted_count += 1
                
                # Print progress every 100 frames
                if extracted_count % 100 == 0:
                    print(f"    Extracted {extracted_count} frames so far")
            
            frame_count += 1
        
        # Release the video object
        video.release()
        
        print(f"Completed {video_file}: {extracted_count} frames extracted")
    
    print(f"All videos processed. Total videos: {len(mp4_files)}")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from MP4 videos')
    parser.add_argument('input_folder', help='Folder containing MP4 files')
    parser.add_argument('output_folder', help='Folder where extracted frames will be saved')
    parser.add_argument('--fps', type=float, help='Target frames per second to extract (default: extract all frames)', default=None)
    
    args = parser.parse_args()
    
    extract_frames(args.input_folder, args.output_folder, args.fps)

if __name__ == "__main__":
    main()