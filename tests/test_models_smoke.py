import torch
import pytest
from src.models.factory import create_model, list_available_models


@pytest.mark.parametrize("model_name", list_available_models())
def test_model_creation(model_name):
    """Test that all models can be created without errors."""
    config = {
        "name": model_name,
        "type": "transformer" if "transformer" in model_name else "rnn",
        "num_classes": 60,
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "input_size": 1629,  # For BiLSTM
        "patch_size": 16,
        "num_frames": 8,
        "image_size": 224,
        "input_channels": 3,
    }
    
    model = create_model(config)
    assert model is not None
    assert hasattr(model, 'forward')


def test_videomae_forward():
    """Test VideoMAE forward pass."""
    config = {
        "name": "videomae",
        "type": "transformer",
        "num_classes": 60,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "patch_size": 16,
        "num_frames": 16,
        "image_size": 224,
        "input_channels": 3,
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 16, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 60)
    assert not torch.isnan(output).any()


def test_vivit_forward():
    """Test ViViT forward pass."""
    config = {
        "name": "vivit",
        "type": "transformer",
        "num_classes": 60,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "patch_size": 16,
        "num_frames": 16,
        "image_size": 224,
        "input_channels": 3,
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 16, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 60)
    assert not torch.isnan(output).any()


def test_internvideo_forward():
    """Test InternVideo forward pass."""
    config = {
        "name": "internvideo",
        "type": "transformer",
        "num_classes": 60,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "patch_size": 16,
        "num_frames": 8,
        "image_size": 224,
        "input_channels": 3,
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 8, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 60)
    assert not torch.isnan(output).any()


def test_stformer_forward():
    """Test ST-Former forward pass."""
    config = {
        "name": "stformer",
        "type": "transformer",
        "num_classes": 60,
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "patch_size": 16,
        "num_frames": 16,
        "image_size": 224,
        "input_channels": 3,
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 16, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 60)
    assert not torch.isnan(output).any()


def test_bilstm_attn_forward():
    """Test BiLSTM+Attention forward pass."""
    config = {
        "name": "bilstm_attn",
        "type": "rnn",
        "num_classes": 60,
        "input_size": 1629,
        "hidden_size": 512,
        "num_layers": 2,
        "dropout": 0.1,
        "bidirectional": True,
        "attention_type": "dot",
        "attention_hidden_size": 256,
    }
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    input_tensor = torch.randn(batch_size, seq_len, 1629)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 60)
    assert not torch.isnan(output).any()


def test_model_factory():
    """Test model factory functionality."""
    # Test listing available models
    models = list_available_models()
    assert len(models) > 0
    assert "videomae" in models
    assert "vivit" in models
    assert "internvideo" in models
    assert "stformer" in models
    assert "bilstm_attn" in models
    
    # Test unknown model
    with pytest.raises(ValueError):
        create_model({"name": "unknown_model"})

