import os
import argparse
import cv2
from pathlib import Path
import math
import concurrent.futures
import time

def extract_frames_from_video(video_file, input_folder, output_folder, fps=None):
    """
    Extract frames from a single video file and save them as JPEGs.
    
    Args:
        video_file (str): Name of the video file
        input_folder (str): Path to folder containing the video file
        output_folder (str): Path to folder where extracted frames will be saved
        fps (float, optional): Target frames per second to extract. If None, extract all frames.
    
    Returns:
        tuple: (video_file, extracted_count) - file name and number of frames extracted
    """
    # Get base name without extension (e.g., "a" from "a.mp4")
    base_name = os.path.splitext(video_file)[0]
    
    # Full path to the video file
    video_path = os.path.join(input_folder, video_file)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error opening video file {video_file}")
        return video_file, 0
    
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
            print(f"Warning: Cannot extract at {fps} FPS: video {video_file} has only {video_fps:.2f} FPS ({total_frames} frames)")
            frame_indices = range(total_frames)
        else:
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
    
    # Create a subfolder for this video to keep extracted frames organized
    video_output_folder = os.path.join(output_folder, base_name)
    os.makedirs(video_output_folder, exist_ok=True)
    
    for frame_idx in frame_indices:
        # Set frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        success, frame = video.read()
        
        if success:
            # Create filename with format: base_name_0000001.jpg
            frame_filename = f"{base_name}_{extracted_count:07d}.jpg"
            output_path = os.path.join(video_output_folder, frame_filename)
            
            # Save the frame as JPEG
            cv2.imwrite(output_path, frame)
            
            extracted_count += 1
            
            # Print progress every 100 frames
            if extracted_count % 100 == 0:
                print(f"    {video_file}: Extracted {extracted_count} frames so far")
        
        frame_count += 1
    
    # Release the video object
    video.release()
    
    print(f"Completed {video_file}: {extracted_count} frames extracted")
    return video_file, extracted_count

def extract_frames(input_folder, output_folder, fps=None, max_workers=None):
    """
    Extract frames from MP4 files in the input folder in parallel and save them as JPEGs.
    
    Args:
        input_folder (str): Path to folder containing MP4 files
        output_folder (str): Path to folder where extracted frames will be saved
        fps (float, optional): Target frames per second to extract. If None, extract all frames.
        max_workers (int, optional): Maximum number of worker processes. If None, uses 
                                    the number of processors on the machine.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all MP4 files in the input directory
    mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {input_folder}")
        return
    
    print(f"Found {len(mp4_files)} video files to process")
    
    start_time = time.time()
    total_extracted = 0
    
    # Use ThreadPoolExecutor for I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each video file
        future_to_video = {
            executor.submit(extract_frames_from_video, video_file, input_folder, output_folder, fps): video_file
            for video_file in mp4_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            video_file = future_to_video[future]
            try:
                _, frames_extracted = future.result()
                total_extracted += frames_extracted
            except Exception as exc:
                print(f"{video_file} generated an exception: {exc}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nAll videos processed. Total videos: {len(mp4_files)}")
    print(f"Total frames extracted: {total_extracted}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from MP4 videos in parallel')
    parser.add_argument('input_folder', help='Folder containing MP4 files')
    parser.add_argument('output_folder', help='Folder where extracted frames will be saved')
    parser.add_argument('--fps', type=float, help='Target frames per second to extract (default: extract all frames)', default=None)
    parser.add_argument('--workers', type=int, help='Maximum number of worker processes (default: number of CPUs)', default=None)
    
    args = parser.parse_args()
    
    extract_frames(args.input_folder, args.output_folder, args.fps, args.workers)

if __name__ == "__main__":
    main()