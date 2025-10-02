import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class STFormer(nn.Module):
    """Spatial-Temporal Former model for sign language recognition."""
    
    def __init__(self, 
                 patch_size: int = 16,
                 num_frames: int = 16,
                 hidden_size: int = 512,
                 num_hidden_layers: int = 8,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 2048,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 spatial_attention: bool = True,
                 temporal_attention: bool = True,
                 cross_attention: bool = True,
                 input_channels: int = 3,
                 image_size: int = 224,
                 num_classes: int = 60,
                 classifier_dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.cross_attention = cross_attention
        
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
        
        # Spatial-Temporal attention layers
        self.st_layers = nn.ModuleList([
            STAttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps,
                spatial_attention=spatial_attention,
                temporal_attention=temporal_attention,
                cross_attention=cross_attention
            ) for _ in range(num_hidden_layers)
        ])
        
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
        
        # Spatial-Temporal attention layers
        for layer in self.st_layers:
            x = layer(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # CLS token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits


class STAttentionLayer(nn.Module):
    """Spatial-Temporal attention layer."""
    
    def __init__(self, 
                 hidden_size: int,
                 num_attention_heads: int,
                 intermediate_size: int,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 spatial_attention: bool = True,
                 temporal_attention: bool = True,
                 cross_attention: bool = True):
        super().__init__()
        
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.cross_attention = cross_attention
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Spatial attention
        if spatial_attention:
            self.spatial_attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=attention_probs_dropout_prob,
                batch_first=True
            )
        
        # Temporal attention
        if temporal_attention:
            self.temporal_attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=attention_probs_dropout_prob,
                batch_first=True
            )
        
        # Cross attention
        if cross_attention:
            self.cross_attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=attention_probs_dropout_prob,
                batch_first=True
            )
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Spatial attention
        if self.spatial_attention:
            # Reshape for spatial attention (simplified)
            attn_output, _ = self.spatial_attention_layer(x, x, x)
            x = self.norm2(x + attn_output)
        
        # Temporal attention
        if self.temporal_attention:
            # Reshape for temporal attention (simplified)
            attn_output, _ = self.temporal_attention_layer(x, x, x)
            x = self.norm3(x + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + ff_output
        
        return x
