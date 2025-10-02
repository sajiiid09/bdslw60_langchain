import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InternVideo(nn.Module):
    """InternVideo model for sign language recognition."""
    
    def __init__(self, 
                 patch_size: int = 16,
                 num_frames: int = 8,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 input_channels: int = 3,
                 image_size: int = 224,
                 num_classes: int = 60,
                 classifier_dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            input_channels, hidden_size,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Position embedding
        num_patches = num_frames * (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Multi-scale temporal modeling
        self.temporal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, hidden_size, T, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.position_embedding
        
        # Temporal convolution
        x_temporal = x.transpose(1, 2)  # (B, hidden_size, num_patches)
        x_temporal = self.temporal_conv(x_temporal)
        x = x_temporal.transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # CLS token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
