from typing import Dict, Any, Type
import torch.nn as nn

from .videomae import VideoMAE
from .vivit import ViViT
from .internvideo import InternVideo
from .st_former import STFormer
from .bilstm_attn import BiLSTMAttention


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "videomae": VideoMAE,
    "vivit": ViViT,
    "internvideo": InternVideo,
    "stformer": STFormer,
    "bilstm_attn": BiLSTMAttention,
}


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Create a model from configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_name = model_config.get("name", "").lower()
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    # Extract model parameters from config
    model_params = model_config.get("model", {})
    
    # Remove name and type from model params
    model_params = {k: v for k, v in model_params.items() if k not in ["name", "type"]}
    
    # Create model instance
    model = model_class(**model_params)
    
    return model


def list_available_models() -> list:
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = MODEL_REGISTRY[model_name]
    
    return {
        "name": model_name,
        "class": model_class.__name__,
        "module": model_class.__module__,
        "docstring": model_class.__doc__,
    }
