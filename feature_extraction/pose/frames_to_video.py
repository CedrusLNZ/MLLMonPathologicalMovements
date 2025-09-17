import os
import argparse
import cv2
import re
from collections import defaultdict
from pathlib import Path

def frames_to_video(input_folder, output_folder, fps=30):
    """
    Convert frame images back to MP4 videos.
    
    Args:
        input_folder (str): Path to folder containing frame images
        output_folder (str): Path to folder where videos will be saved
        fps (int): Frames per second for the output videos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all jpg files in the input directory
    all_frames = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
    if not all_frames:
        print(f"No JPG files found in {input_folder}")
        return
    
    # Group frames by their base video name
    videos = defaultdict(list)
    pattern = re.compile(r'(.+?)_\d{7}\.jpg')
    
    print(f"Analyzing {len(all_frames)} frames...")
    
    for frame_file in all_frames:
        match = pattern.match(frame_file)
        if match:
            video_name = match.group(1)
            videos[video_name].append(frame_file)
    
    if not videos:
        print("No valid frame sequences found")
        return
    
    print(f"Found {len(videos)} video sequences to reconstruct")
    
    # Process each video sequence
    for video_name, frame_files in videos.items():
        # Sort frames by their number to ensure correct sequence
        frame_files.sort(key=lambda x: int(re.search(r'_(\d{7})\.jpg', x).group(1)))
        
        print(f"Processing {video_name}: {len(frame_files)} frames")
        
        # Get first frame to determine video dimensions
        first_frame_path = os.path.join(input_folder, frame_files[0])
        sample_frame = cv2.imread(first_frame_path)
        
        if sample_frame is None:
            print(f"Error reading frame: {first_frame_path}")
            continue
            
        height, width, layers = sample_frame.shape
        
        # Define the codec and create VideoWriter object
        output_path = os.path.join(output_folder, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use mp4v codec
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames_processed = 0
        
        # Add each frame to the video
        for frame_file in frame_files:
            frame_path = os.path.join(input_folder, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                video_writer.write(frame)
                frames_processed += 1
                
                # Print progress every 100 frames
                if frames_processed % 100 == 0:
                    print(f"  Added {frames_processed}/{len(frame_files)} frames to {video_name}.mp4")
        
        # Release the video writer
        video_writer.release()
        
        print(f"Completed {video_name}.mp4: {frames_processed} frames added")
    
    print(f"All videos reconstructed. Total videos: {len(videos)}")

def main():
    parser = argparse.ArgumentParser(description='Convert frame images to MP4 videos')
    parser.add_argument('input_folder', help='Folder containing frame images')
    parser.add_argument('output_folder', help='Folder where videos will be saved')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output videos (default: 30)')
    
    args = parser.parse_args()
    
    frames_to_video(args.input_folder, args.output_folder, args.fps)

if __name__ == "__main__":
    main()