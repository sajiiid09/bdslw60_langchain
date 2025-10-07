from typing import Dict, Any, Type
import torch.nn as nn

from .videomae import VideoMAE
from .videomae_adapter import VideoMAEFromCodes
from .vivit import ViViT
from .vivit_adapter import ViViTFromCodes
from .internvideo import InternVideo
from .st_former import STFormer
from .stformer_adapter import STFormerFromCodes
from .bilstm_attn import BiLSTMAttention


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "videomae": VideoMAE,
    "videomae_from_codes": VideoMAEFromCodes,
    "vivit": ViViT,
    "vivit_from_codes": ViViTFromCodes,
    "internvideo": InternVideo,
    "stformer": STFormer,
    "stformer_from_codes": STFormerFromCodes,
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

    # Extract model parameters from config. Support both flat and nested ("model") styles.
    raw_params = model_config.get("model", None)
    if isinstance(raw_params, dict) and len(raw_params) > 0:
        params_source = raw_params
    else:
        params_source = model_config

    # Remove name and type from params
    model_params: Dict[str, Any] = {k: v for k, v in params_source.items() if k not in ["name", "type"]}

    # Special handling for videomae_from_codes: pack backbone kwargs if given flat
    if model_name == "videomae_from_codes":
        # Separate known adapter args
        adapter_keys = {"input_feature_dim", "adapter_channels", "adapter_small_h", "adapter_small_w"}
        # All other args are for the VideoMAE backbone
        videomae_kwargs = {k: v for k, v in model_params.items() if k not in adapter_keys}
        adapter_params = {k: v for k, v in model_params.items() if k in adapter_keys}
        model_params = {"videomae_kwargs": videomae_kwargs, **adapter_params}
    elif model_name == "stformer_from_codes":
        adapter_keys = {"input_feature_dim", "adapter_channels", "adapter_small_h", "adapter_small_w"}
        stformer_kwargs = {k: v for k, v in model_params.items() if k not in adapter_keys}
        adapter_params = {k: v for k, v in model_params.items() if k in adapter_keys}
        model_params = {"stformer_kwargs": stformer_kwargs, **adapter_params}
    elif model_name == "vivit_from_codes":
        adapter_keys = {"input_feature_dim", "adapter_channels", "adapter_small_h", "adapter_small_w"}
        vivit_kwargs = {k: v for k, v in model_params.items() if k not in adapter_keys}
        adapter_params = {k: v for k, v in model_params.items() if k in adapter_keys}
        model_params = {"vivit_kwargs": vivit_kwargs, **adapter_params}

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
