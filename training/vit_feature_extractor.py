"""
ViT-based feature extractor for video frames.
Based on the ViT sample code from the Colab notebook.

Follows the same frame extraction pattern as MLLM code (Qwen/InternVL).
The code processes videos regardless of their native resolution/fps/audio format,
similar to how MLLM code handles video input.

Note: Recordings are typically 1920×1080@30fps with 44.1kHz mono audio,
but the code is format-agnostic like the existing MLLM implementation.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from torchvision.io import read_video
from tqdm import tqdm
import os
import hashlib

class ViTFeatureExtractor:
    """
    Vision Transformer feature extractor for video frames.
    Extracts features from video frames using a pre-trained ViT model.
    
    Handles videos with:
    - Resolution: 1920 × 1080 pixels
    - Frame rate: 30 fps
    - Audio: 44.1 kHz mono (not used in ViT extraction)
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda', max_frames=60, cache_dir=None):
        """
        Args:
            model_name: HuggingFace model name for ViT
            device: Device to run inference on
            max_frames: Maximum number of frames to extract from video
            cache_dir: Directory for caching extracted frames (similar to MLLM code)
        """
        self.device = device
        self.max_frames = max_frames
        self.cache_dir = cache_dir
        
        # Load ViT model and processor
        # ViT models typically expect 224x224 input, processor handles resizing
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get feature dimension
        self.feature_dim = self.model.config.hidden_size
        
    def get_video_frames(self, video_path, num_frames=None, cache_dir=None):
        """
        Extract frames from video, following the same pattern as MLLM code.
        Caches frames using MD5 hash for efficiency.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: self.max_frames)
            cache_dir: Directory for caching (default: self.cache_dir)
        
        Returns:
            frames: numpy array of frames [num_frames, H, W, C] as uint8
            timestamps: numpy array of timestamps
        """
        if num_frames is None:
            num_frames = self.max_frames
        if cache_dir is None:
            cache_dir = self.cache_dir
        
        # Use caching similar to MLLM code
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
            frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
            timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')
            
            if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
                frames = np.load(frames_cache_file)
                timestamps = np.load(timestamps_cache_file)
                return frames, timestamps
        
        # Read video using torchvision (same as MLLM)
        video_tensor, audio_tensor, video_info = read_video(video_path, pts_unit='sec')
        total_frames = video_tensor.shape[0]
        fps = video_info['video_fps']
        
        # Sample frames uniformly (same pattern as MLLM)
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        
        # Extract selected frames
        selected_frames = video_tensor[indices]  # Shape: [num_frames, H, W, C]
        
        # Convert to numpy array and ensure uint8 format (same as MLLM)
        frames = selected_frames.numpy().astype(np.uint8)
        
        # Calculate timestamps for selected frames
        timestamps = np.array([idx / fps for idx in indices])
        
        # Cache frames if cache_dir is provided
        if cache_dir is not None:
            np.save(frames_cache_file, frames)
            np.save(timestamps_cache_file, timestamps)
        
        return frames, timestamps
    
    def extract_frames(self, video_path, num_frames=None):
        """
        Extract frames and convert to PIL Images for ViT processing.
        Uses get_video_frames() which follows MLLM caching pattern.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: self.max_frames)
        
        Returns:
            frames: List of PIL Images
            timestamps: List of timestamps
        """
        # Get frames as numpy arrays (with caching)
        frames_np, timestamps = self.get_video_frames(video_path, num_frames)
        
        # Convert to PIL Images for ViT processing
        frames = []
        for frame_np in frames_np:
            # Ensure RGB format (handle grayscale if needed)
            if len(frame_np.shape) == 2:
                frame_np = np.stack([frame_np] * 3, axis=-1)
            elif frame_np.shape[2] == 1:
                frame_np = np.repeat(frame_np, 3, axis=2)
            elif frame_np.shape[2] == 4:  # RGBA
                frame_np = frame_np[:, :, :3]  # Take RGB channels
            
            frame_pil = Image.fromarray(frame_np, mode='RGB')
            frames.append(frame_pil)
        
        return frames, timestamps
    
    def extract_features(self, video_path, pooling='mean'):
        """
        Extract ViT features from video.
        
        Processes frames from 1920×1080 resolution videos at 30 fps.
        The ViT processor automatically resizes frames to the model's input size (224×224).
        
        Args:
            video_path: Path to video file
            pooling: How to pool frame features ('mean', 'max', 'last', or 'all')
        
        Returns:
            features: Extracted features (shape depends on pooling)
        """
        # Extract frames (1920×1080 resolution)
        frames, timestamps = self.extract_frames(video_path)
        
        # Process frames through ViT
        frame_features = []
        
        with torch.no_grad():
            for frame in frames:
                # Process image - processor handles resizing from 1920×1080 to 224×224
                # and normalization
                inputs = self.processor(images=frame, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                outputs = self.model(**inputs)
                # Use [CLS] token (first token) which contains global image representation
                frame_feat = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
                frame_features.append(frame_feat.cpu())
        
        # Stack frame features
        frame_features = torch.cat(frame_features, dim=0)  # [num_frames, hidden_size]
        
        # Pool features across temporal dimension
        if pooling == 'mean':
            features = frame_features.mean(dim=0)  # [hidden_size]
        elif pooling == 'max':
            features = frame_features.max(dim=0)[0]  # [hidden_size]
        elif pooling == 'last':
            features = frame_features[-1]  # [hidden_size]
        elif pooling == 'all':
            features = frame_features.flatten()  # [num_frames * hidden_size]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return features.numpy()
    
    def extract_features_batch(self, video_paths, pooling='mean', batch_size=8):
        """
        Extract features from multiple videos in batches.
        
        Args:
            video_paths: List of video paths
            pooling: How to pool frame features
            batch_size: Batch size for processing
        
        Returns:
            features_list: List of feature arrays
        """
        features_list = []
        
        for video_path in tqdm(video_paths, desc="Extracting ViT features"):
            try:
                features = self.extract_features(video_path, pooling=pooling)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                # Use zero features as fallback
                if pooling == 'all':
                    features_list.append(np.zeros(self.max_frames * self.feature_dim))
                else:
                    features_list.append(np.zeros(self.feature_dim))
        
        return np.array(features_list)
