import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np
from keypoint_info import keypoint_info
from tqdm import tqdm
from collections import deque
import multiprocessing

def diameter(points):
    max_x = np.max(points[:,0])
    min_x = np.min(points[:,0])
    max_y = np.max(points[:,1]) 
    min_y = np.min(points[:,1])
    diameter = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    return diameter

def crop_video(
    raw_frame_dir = '/mnt/SSD1/prateik/frames/A0002@5-13-2021@UA6693LK@sz_v1',
    pose_frame_dir = '/mnt/SSD1/prateik/tmp/processed_frames_pose_A0002@5-13-2021@UA6693LK@sz_v1_QwOMi/sapiens_1b',
    output_path = "/mnt/SSD1/prateik/face_detection/face_frames.gif",
    score_cutoff = 0.1,
    history_size = 50 
):
    frames = sorted(list(set([x.split('.')[0] for x in os.listdir(pose_frame_dir)])))
    face_indices = list(keypoint_info.keys())
    cropped_frames = []
    box_size = 200

    # Get first image to initialize coordinates
    first_image = Image.open(os.path.join(raw_frame_dir, frames[0]) + '.jpg')
    img_width, img_height = first_image.size
    
    # Initialize the center coordinates history with the center of the first image
    center_x_history = deque(maxlen=history_size)
    center_y_history = deque(maxlen=history_size)
    
    # Initialize with the center of the image
    initial_x, initial_y = img_width // 2, img_height // 2
    center_x_history.append(initial_x)
    center_y_history.append(initial_y)

    # diameter_list = []
    # for frame_idx, frame in enumerate(frames):
    #     frame_image_raw = Image.open(os.path.join(raw_frame_dir, frame) + '.jpg')
    #     with open(os.path.join(pose_frame_dir, frame) + '.json', 'r') as f:
    #         frame_json = json.load(f)
    #         num_people = len(frame_json['instance_info'])
    #         frame_json_points_list = []
    #         frame_json_scores_list = []
    #         for i in range(num_people):
    #             frame_json_points_list.append(frame_json['instance_info'][i]['keypoints'])
    #             frame_json_scores_list.append(frame_json['instance_info'][i]['keypoint_scores'])
    #     for frame_json_points, frame_json_scores in zip(frame_json_points_list, frame_json_scores_list): 
    #         points = np.array(frame_json_points)[face_indices]
    #         scores = np.array(frame_json_scores)[face_indices]
    #         above_score_cutoff = scores > score_cutoff
    #         points = points[above_score_cutoff]
    #         if not points.shape[0] == 0:
    #             diameter_list.append(diameter(points))
    # box_size = np.mean(np.array(diameter_list))
    # print(f'Average Face Diameter: {box_size}')

    for frame_idx, frame in enumerate(tqdm(frames)):
        frame_image_raw = Image.open(os.path.join(raw_frame_dir, frame) + '.jpg')
        
        with open(os.path.join(pose_frame_dir, frame) + '.json', 'r') as f:
            frame_json = json.load(f)
            num_people = len(frame_json['instance_info'])
            frame_json_points_list = []
            frame_json_scores_list = []
            for i in range(num_people):
                frame_json_points_list.append(frame_json['instance_info'][i]['keypoints'])
                frame_json_scores_list.append(frame_json['instance_info'][i]['keypoint_scores'])

        center_x_list = []
        center_y_list = []

        for frame_json_points, frame_json_scores in zip(frame_json_points_list, frame_json_scores_list): 
            points = np.array(frame_json_points)[face_indices]
            scores = np.array(frame_json_scores)[face_indices]
            above_score_cutoff = scores > score_cutoff
            points = points[above_score_cutoff]
            scores = scores[above_score_cutoff]
            assert points.shape[0] == scores.shape[0]
            if points.shape[0] == 0:
                continue
            else:
                center_x_list.append(np.mean(points[:,0]))
                center_y_list.append(np.mean(points[:,1]))
        
        assert len(center_x_list) == len(center_y_list), 'Mismatch between number of x and y coordinates'

        # Calculate the running average of previous centers
        running_avg_x = np.mean(center_x_history)
        running_avg_y = np.mean(center_y_history)

        num_faces = len(center_x_list)
        if num_faces == 0:
            # If no faces detected, use the running average
            center_x = running_avg_x
            center_y = running_avg_y
        elif num_faces == 1:
            # If only one face, use it
            center_x = center_x_list[0]
            center_y = center_y_list[0]
        else:
            # Find the face closest to the running average
            distances = [np.sqrt((x - running_avg_x)**2 + (y - running_avg_y)**2) 
                        for x, y in zip(center_x_list, center_y_list)]
            closest_idx = np.argmin(distances)
            center_x = center_x_list[closest_idx]
            center_y = center_y_list[closest_idx]

        # Add current center to history
        center_x_history.append(center_x)
        center_y_history.append(center_y)

        # Crop using current center point
        x_min = int(center_x) - box_size
        x_max = int(center_x) + box_size
        y_min = int(center_y) - box_size 
        y_max = int(center_y) + box_size
        
        # Crop the image to bounding box
        frame_image_raw = frame_image_raw.crop((x_min, y_min, x_max, y_max))
        cropped_frames.append(frame_image_raw)

    # Save frames as video
    print('Saving video...')
    print(f'No. frames in original:  {len(frames)}| No. frames in cropped: {len(cropped_frames)} ')
    if cropped_frames:
        frames = [np.array(frame) for frame in cropped_frames]
        height, width = frames[0].shape[:2]
        # Use avc1 codec instead of mp4v for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # Make sure video file can be created before writing frames
        out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=10.0, frameSize=(width, height))
        if not out.isOpened():
            print("Error: VideoWriter failed to open. Trying different codec...")
            for codec in ['XVID', 'MJPG']:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=10.0, frameSize=(width, height))
                if out.isOpened():
                    print(f"Successfully opened with codec: {codec}")
                    break
            if not out.isOpened():
                raise RuntimeError("Could not open video writer with any codec")
                
        for i, frame in enumerate(frames):
            # print(frame.shape)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success = out.write(frame_bgr)
            # if not success:
                # print(f"Failed to write frame {i}")
        
        out.release()
        # Verify the output video duration
        cap = cv2.VideoCapture(output_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            print(f"Video duration: {duration:.2f} seconds")
            print(f"Frame count: {frame_count}")
            print(f"FPS: {fps}")
        cap.release()
        print(f"Saved video to {output_path}")
    else:
        print("No frames to save!")

# def run():
#     raw_frame_root = '/mnt/SSD1/prateik/seizure_videos/frames/original'
#     pose_frame_root = '/mnt/SSD1/prateik/seizure_videos/frames/pose'
#     cropped_path = '/mnt/SSD1/prateik/seizure_videos/videos/cropped'
    
#     for raw_frame_dir in enumerate(os.listdir(raw_frame_root)):
#         raw_frame_dir_path = os.path.join(raw_frame_root, raw_frame_dir)
#         pose_frame_dir_path = os.path.join(pose_frame_root, raw_frame_dir, 'sapiens_1b')
#         output_path = os.path.join(cropped_path, raw_frame_dir + '.mp4')
#         crop_video(
#             raw_frame_dir = raw_frame_dir_path,
#             pose_frame_dir = pose_frame_dir_path,
#             output_path = output_path,
#             score_cutoff = 0.1,
#             history_size = 50 
#         )

def process_video(raw_frame_dir):
    raw_frame_root = '/mnt/SSD1/prateik/seizure_videos/frames/original'
    pose_frame_root = '/mnt/SSD1/prateik/seizure_videos/frames/pose'
    cropped_path = '/mnt/SSD1/prateik/seizure_videos/videos/cropped'
    
    raw_frame_dir_path = os.path.join(raw_frame_root, raw_frame_dir)
    pose_frame_dir_path = os.path.join(pose_frame_root, raw_frame_dir, 'sapiens_1b')
    output_path = os.path.join(cropped_path, raw_frame_dir + '.mp4')
    
    crop_video(
        raw_frame_dir=raw_frame_dir_path,
        pose_frame_dir=pose_frame_dir_path,
        output_path=output_path,
        score_cutoff=0.1,
        history_size=50 
    )

def run():
    raw_frame_root = '/mnt/SSD1/prateik/seizure_videos/frames/original'
    raw_frame_dirs = os.listdir(raw_frame_root)
    
    with multiprocessing.Pool(processes=100) as pool:
        pool.map(process_video, raw_frame_dirs)

if __name__ == '__main__':
    run()