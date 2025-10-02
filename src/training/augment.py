import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union


class TemporalAugmentation:
    """Temporal augmentation for sequence data."""
    
    def __init__(self, 
                 temporal_dropout: float = 0.1,
                 noise_std: float = 0.01,
                 time_warp: bool = False,
                 time_warp_factor: float = 0.1):
        self.temporal_dropout = temporal_dropout
        self.noise_std = noise_std
        self.time_warp = time_warp
        self.time_warp_factor = time_warp_factor
    
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentation to sequence.
        
        Args:
            sequence: Input sequence tensor (B, T, D) or (T, D)
            
        Returns:
            Augmented sequence tensor
        """
        if sequence.dim() == 2:
            # (T, D) -> (1, T, D)
            sequence = sequence.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply augmentations
        if self.temporal_dropout > 0:
            sequence = self._temporal_dropout(sequence)
        
        if self.noise_std > 0:
            sequence = self._add_noise(sequence)
        
        if self.time_warp:
            sequence = self._time_warp(sequence)
        
        if squeeze_output:
            sequence = sequence.squeeze(0)
        
        return sequence
    
    def _temporal_dropout(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply temporal dropout."""
        batch_size, seq_len, _ = sequence.shape
        mask = torch.rand(batch_size, seq_len, 1, device=sequence.device) > self.temporal_dropout
        return sequence * mask
    
    def _add_noise(self, sequence: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(sequence) * self.noise_std
        return sequence + noise
    
    def _time_warp(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply time warping."""
        batch_size, seq_len, _ = sequence.shape
        
        # Generate warping factors
        warp_factors = 1.0 + torch.randn(batch_size, device=sequence.device) * self.time_warp_factor
        warp_factors = torch.clamp(warp_factors, 0.5, 2.0)
        
        # Apply warping
        warped_sequences = []
        for i in range(batch_size):
            warped_len = int(seq_len * warp_factors[i])
            if warped_len != seq_len:
                # Interpolate to original length
                indices = torch.linspace(0, warped_len - 1, seq_len, device=sequence.device)
                warped_seq = torch.nn.functional.interpolate(
                    sequence[i].transpose(0, 1).unsqueeze(0),
                    size=warped_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(0).transpose(0, 1)
                
                # Sample back to original length
                warped_seq = warped_seq[indices.long()]
            else:
                warped_seq = sequence[i]
            
            warped_sequences.append(warped_seq)
        
        return torch.stack(warped_sequences)


class SpatialAugmentation:
    """Spatial augmentation for landmark data."""
    
    def __init__(self, 
                 rotation_range: float = 0.1,
                 translation_range: float = 0.05,
                 scale_range: float = 0.1,
                 noise_std: float = 0.01):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
    
    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply spatial augmentation to landmarks.
        
        Args:
            landmarks: Input landmarks tensor (B, T, D) or (T, D)
                      where D = num_landmarks * 3 (x, y, z)
            
        Returns:
            Augmented landmarks tensor
        """
        if landmarks.dim() == 2:
            # (T, D) -> (1, T, D)
            landmarks = landmarks.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply augmentations
        if self.rotation_range > 0:
            landmarks = self._apply_rotation(landmarks)
        
        if self.translation_range > 0:
            landmarks = self._apply_translation(landmarks)
        
        if self.scale_range > 0:
            landmarks = self._apply_scaling(landmarks)
        
        if self.noise_std > 0:
            landmarks = self._add_noise(landmarks)
        
        if squeeze_output:
            landmarks = landmarks.squeeze(0)
        
        return landmarks
    
    def _apply_rotation(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to landmarks."""
        batch_size, seq_len, _ = landmarks.shape
        
        # Generate random rotation angles
        angles = torch.rand(batch_size, device=landmarks.device) * 2 * np.pi * self.rotation_range
        
        # Apply rotation to x, y coordinates (assuming landmarks are in x, y, z format)
        rotated_landmarks = landmarks.clone()
        for i in range(batch_size):
            angle = angles[i]
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            
            # Reshape to (num_landmarks, 3)
            lm = landmarks[i].view(-1, 3)
            
            # Apply rotation to x, y coordinates
            x, y = lm[:, 0], lm[:, 1]
            lm[:, 0] = x * cos_a - y * sin_a
            lm[:, 1] = x * sin_a + y * cos_a
            
            # Reshape back
            rotated_landmarks[i] = lm.view(seq_len, -1)
        
        return rotated_landmarks
    
    def _apply_translation(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random translation to landmarks."""
        batch_size, seq_len, _ = landmarks.shape
        
        # Generate random translations
        translations = torch.randn(batch_size, 3, device=landmarks.device) * self.translation_range
        
        # Apply translation
        translated_landmarks = landmarks.clone()
        for i in range(batch_size):
            # Reshape to (num_landmarks, 3)
            lm = landmarks[i].view(-1, 3)
            
            # Apply translation
            lm += translations[i]
            
            # Reshape back
            translated_landmarks[i] = lm.view(seq_len, -1)
        
        return translated_landmarks
    
    def _apply_scaling(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random scaling to landmarks."""
        batch_size, seq_len, _ = landmarks.shape
        
        # Generate random scale factors
        scales = 1.0 + torch.randn(batch_size, device=landmarks.device) * self.scale_range
        scales = torch.clamp(scales, 0.5, 2.0)
        
        # Apply scaling
        scaled_landmarks = landmarks.clone()
        for i in range(batch_size):
            # Reshape to (num_landmarks, 3)
            lm = landmarks[i].view(-1, 3)
            
            # Apply scaling
            lm *= scales[i]
            
            # Reshape back
            scaled_landmarks[i] = lm.view(seq_len, -1)
        
        return scaled_landmarks
    
    def _add_noise(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to landmarks."""
        noise = torch.randn_like(landmarks) * self.noise_std
        return landmarks + noise


class MixUp:
    """MixUp augmentation for classification tasks."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation.
        
        Args:
            x: Input features (B, ...)
            y: Input labels (B,)
            
        Returns:
            Mixed features and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix augmentation for classification tasks."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation.
        
        Args:
            x: Input features (B, T, D)
            y: Input labels (B,)
            
        Returns:
            Mixed features and labels
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # For sequence data, cut along time dimension
        seq_len = x.size(1)
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_start + cut_len] = x[index, cut_start:cut_start + cut_len]
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

