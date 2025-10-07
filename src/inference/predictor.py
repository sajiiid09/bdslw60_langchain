import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union
import os

from ..models.factory import create_model
from ..data.mediapipe_runner import HolisticExtractor
from ..utils.rqe import RelativeQuantizationEncoder
from ..utils.io import read_yaml


class SignLanguagePredictor:
    """Sign language prediction pipeline."""
    
    def __init__(self, 
                 model_path: str,
                 model_config: Dict[str, Any],
                 device: str = "auto"):
        """Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            model_config: Model configuration
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        self.model_config = model_config
        
        # Load model
        self.model = self._load_model(model_path, model_config)
        self.model.eval()
        
        # Initialize preprocessing components
        self.pose_extractor = HolisticExtractor()
        self.rqe_encoder = RelativeQuantizationEncoder()
        
        # Load preprocessing parameters if available
        self.preprocessing_params = self._load_preprocessing_params(model_path)
    
    def _get_device(self, device: str) -> torch.device:
        """Get device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str, model_config: Dict[str, Any]) -> nn.Module:
        """Load trained model."""
        # Create model (support both full config with 'model' section or flat)
        cfg = model_config.get("model", model_config)
        model = create_model(cfg)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # If adapter model, ensure adapter is initialized before loading state dict
        try:
            from ..models.videomae_adapter import VideoMAEFromCodes
            if isinstance(model, VideoMAEFromCodes):
                # Try to recover feature dim from checkpoint metadata
                feature_dim = None
                if "train_metrics" in checkpoint and "val_metrics" in checkpoint:
                    # no direct feature dim; attempt from config if saved
                    pass
                # Fallback: infer from first linear weight if present in state dict
                sd = checkpoint.get("model_state_dict", {})
                proj_w = sd.get("adapter.project_features.weight", None)
                if proj_w is not None:
                    # shape: (C*h*w, D)
                    feature_dim = int(proj_w.shape[1])
                if feature_dim is not None and getattr(model, "adapter", None) is None:
                    model.init_adapter(feature_dim)
        except Exception:
            pass
        
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def _load_preprocessing_params(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Load preprocessing parameters from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            return checkpoint.get("preprocessing_params", None)
        except:
            return None
    
    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """Preprocess video for inference.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Extract pose landmarks
        landmarks_data = self.pose_extractor.extract_video(video_path)
        landmarks = landmarks_data["landmarks"]  # (T, D)
        
        # Apply RQE encoding if preprocessing params available
        if self.preprocessing_params is not None:
            mean = self.preprocessing_params.get("rqe_mean")
            std = self.preprocessing_params.get("rqe_std")
            if mean is not None and std is not None:
                codes = self.rqe_encoder.encode(landmarks, mean, std)
                # Convert to tensor
                input_tensor = torch.from_numpy(codes).float()
            else:
                # Use raw landmarks
                input_tensor = torch.from_numpy(landmarks).float()
        else:
            # Use raw landmarks
            input_tensor = torch.from_numpy(landmarks).float()
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def preprocess_landmarks(self, landmarks: np.ndarray) -> torch.Tensor:
        """Preprocess landmarks for inference.
        
        Args:
            landmarks: Landmarks array (T, D)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Apply RQE encoding if preprocessing params available
        if self.preprocessing_params is not None:
            mean = self.preprocessing_params.get("rqe_mean")
            std = self.preprocessing_params.get("rqe_std")
            if mean is not None and std is not None:
                codes = self.rqe_encoder.encode(landmarks, mean, std)
                input_tensor = torch.from_numpy(codes).float()
            else:
                input_tensor = torch.from_numpy(landmarks).float()
        else:
            input_tensor = torch.from_numpy(landmarks).float()
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def predict(self, input_data: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Make prediction on input data.
        
        Args:
            input_data: Video path, landmarks array, or tensor
            
        Returns:
            Prediction results
        """
        # Preprocess input
        if isinstance(input_data, str):
            # Video path
            input_tensor = self.preprocess_video(input_data)
        elif isinstance(input_data, np.ndarray):
            # Landmarks array
            input_tensor = self.preprocess_landmarks(input_data)
        elif isinstance(input_data, torch.Tensor):
            # Already a tensor
            if input_data.dim() == 2:
                input_tensor = input_data.unsqueeze(0)
            else:
                input_tensor = input_data
        else:
            raise ValueError("Input must be video path, landmarks array, or tensor")
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
        
        return {
            "predicted_class": predicted_class.cpu().item(),
            "confidence": confidence.cpu().item(),
            "probabilities": probabilities.cpu().numpy()[0],
            "logits": logits.cpu().numpy()[0]
        }
    
    def predict_batch(self, input_data_list: List[Union[str, np.ndarray, torch.Tensor]]) -> List[Dict[str, Any]]:
        """Make predictions on batch of input data.
        
        Args:
            input_data_list: List of video paths, landmarks arrays, or tensors
            
        Returns:
            List of prediction results
        """
        results = []
        for input_data in input_data_list:
            result = self.predict(input_data)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_config.get("name", "Unknown"),
            "model_type": self.model_config.get("type", "Unknown"),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,
            "device": str(self.device),
            "preprocessing_params": self.preprocessing_params is not None
        }


def load_predictor_from_checkpoint(checkpoint_path: str, 
                                  model_config_path: str,
                                  device: str = "auto") -> SignLanguagePredictor:
    """Load predictor from checkpoint and config files.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config_path: Path to model config YAML
        device: Device to run inference on
        
    Returns:
        Initialized predictor
    """
    model_config = read_yaml(model_config_path)
    return SignLanguagePredictor(checkpoint_path, model_config, device)
