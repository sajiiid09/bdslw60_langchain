import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .videomae import VideoMAE


class SequenceToVideoAdapter(nn.Module):
    """Adapt (B, T, D) RQE/code sequences into (B, C, T', H, W) pseudo-video.

    - Projects feature dimension D to C * h * w via a linear layer
    - Reshapes to (B, T, C, h, w)
    - Resamples to (num_frames, image_size, image_size) expected by video models
    """

    def __init__(
        self,
        input_feature_dim: int,
        out_channels: int = 3,
        small_h: int = 8,
        small_w: int = 8,
        target_num_frames: int = 8,
        target_image_size: int = 224,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.small_h = small_h
        self.small_w = small_w
        self.target_num_frames = target_num_frames
        self.target_image_size = target_image_size

        projection_dim = out_channels * small_h * small_w
        self.project_features = nn.Linear(input_feature_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        x = self.project_features(x)  # (B, T, C*h*w)
        x = x.view(b, t, self.out_channels, self.small_h, self.small_w)  # (B, T, C, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, h, w)

        # Resample to target sizes
        x = F.interpolate(
            x,
            size=(self.target_num_frames, self.target_image_size, self.target_image_size),
            mode="trilinear",
            align_corners=False,
        )  # (B, C, T', H, W)
        return x


class VideoMAEFromCodes(nn.Module):
    """Wrapper that allows VideoMAE to consume (B, T, D) code sequences via an adapter.

    Config keys expected inside model_config["model"]:
      - from_codes: bool
      - adapter_channels: int (default 3)
      - adapter_small_h: int (default 8)
      - adapter_small_w: int (default 8)
    """

    def __init__(
        self,
        videomae_kwargs: dict,
        input_feature_dim: int | None = None,
        adapter_channels: int = 3,
        adapter_small_h: int = 8,
        adapter_small_w: int = 8,
    ) -> None:
        super().__init__()

        self._lazy_input_feature_dim = input_feature_dim
        self._adapter_channels = int(adapter_channels)
        self._adapter_small_h = int(adapter_small_h)
        self._adapter_small_w = int(adapter_small_w)

        # Adapter may be created lazily if input_feature_dim is unknown
        self.adapter: SequenceToVideoAdapter | None = None

        # Ensure VideoMAE expects the adapter's channel count
        videomae_kwargs = dict(videomae_kwargs)
        videomae_kwargs["input_channels"] = self._adapter_channels
        self._videomae_kwargs = videomae_kwargs
        self.backbone = VideoMAE(**videomae_kwargs)

    def init_adapter(self, input_feature_dim: int) -> None:
        """Initialize the adapter explicitly using a known feature dimension."""
        self.adapter = SequenceToVideoAdapter(
            input_feature_dim=input_feature_dim,
            out_channels=self._adapter_channels,
            small_h=self._adapter_small_h,
            small_w=self._adapter_small_w,
            target_num_frames=self._videomae_kwargs.get("num_frames", 8),
            target_image_size=self._videomae_kwargs.get("image_size", 224),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if self.adapter is None:
            # Initialize adapter lazily based on input feature dimension
            input_feature_dim = x.shape[-1]
            self.adapter = SequenceToVideoAdapter(
                input_feature_dim=input_feature_dim,
                out_channels=self._adapter_channels,
                small_h=self._adapter_small_h,
                small_w=self._adapter_small_w,
                target_num_frames=self._videomae_kwargs.get("num_frames", 8),
                target_image_size=self._videomae_kwargs.get("image_size", 224),
            )
        x_vid = self.adapter(x)
        return self.backbone(x_vid)


