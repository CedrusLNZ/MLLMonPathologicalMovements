"""
ViT-based feature extractor for video frames.
Based on the ViT sample code from the Colab notebook.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from torchvision.io import read_video
from tqdm import tqdm
import os

class ViTFeatureExtractor:
    """
    Vision Transformer feature extractor for video frames.
    Extracts features from video frames using a pre-trained ViT model.
    """
    
    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda', max_frames=60):
        """
        Args:
            model_name: HuggingFace model name for ViT
            device: Device to run inference on
            max_frames: Maximum number of frames to extract from video
        """
        self.device = device
        self.max_frames = max_frames
        
        # Load ViT model and processor
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get feature dimension
        self.feature_dim = self.model.config.hidden_size
        
    def extract_frames(self, video_path, num_frames=None):
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: self.max_frames)
        
        Returns:
            frames: List of PIL Images
            timestamps: List of timestamps
        """
        if num_frames is None:
            num_frames = self.max_frames
        
        # Read video
        video_tensor, audio_tensor, video_info = read_video(video_path, pts_unit='sec')
        total_frames = video_tensor.shape[0]
        fps = video_info['video_fps']
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        selected_frames = video_tensor[indices]  # Shape: [num_frames, H, W, C]
        
        # Convert to PIL Images
        frames = []
        timestamps = []
        for idx, frame_tensor in enumerate(selected_frames):
            # Convert tensor to PIL Image
            frame_np = frame_tensor.numpy().astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)
            frames.append(frame_pil)
            timestamps.append(indices[idx] / fps)
        
        return frames, timestamps
    
    def extract_features(self, video_path, pooling='mean'):
        """
        Extract ViT features from video.
        
        Args:
            video_path: Path to video file
            pooling: How to pool frame features ('mean', 'max', 'last', or 'all')
        
        Returns:
            features: Extracted features (shape depends on pooling)
        """
        # Extract frames
        frames, timestamps = self.extract_frames(video_path)
        
        # Process frames through ViT
        frame_features = []
        
        with torch.no_grad():
            for frame in frames:
                # Process image
                inputs = self.processor(images=frame, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                outputs = self.model(**inputs)
                # Use [CLS] token or mean pool
                frame_feat = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
                frame_features.append(frame_feat.cpu())
        
        # Stack frame features
        frame_features = torch.cat(frame_features, dim=0)  # [num_frames, hidden_size]
        
        # Pool features
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
