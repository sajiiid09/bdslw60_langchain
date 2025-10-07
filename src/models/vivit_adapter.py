import torch
import torch.nn as nn
import torch.nn.functional as F

from .vivit import ViViT


class SequenceToVideoAdapter(nn.Module):
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
        self.out_channels = int(out_channels)
        self.small_h = int(small_h)
        self.small_w = int(small_w)
        self.target_num_frames = int(target_num_frames)
        self.target_image_size = int(target_image_size)

        projection_dim = self.out_channels * self.small_h * self.small_w
        self.project_features = nn.Linear(int(input_feature_dim), projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        x = self.project_features(x)  # (B, T, C*h*w)
        x = x.view(b, t, self.out_channels, self.small_h, self.small_w)  # (B, T, C, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, h, w)
        x = F.interpolate(
            x,
            size=(self.target_num_frames, self.target_image_size, self.target_image_size),
            mode="trilinear",
            align_corners=False,
        )
        return x


class ViViTFromCodes(nn.Module):
    def __init__(
        self,
        vivit_kwargs: dict,
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

        self.adapter: SequenceToVideoAdapter | None = None

        vivit_kwargs = dict(vivit_kwargs)
        vivit_kwargs["input_channels"] = self._adapter_channels
        self._vivit_kwargs = vivit_kwargs
        self.backbone = ViViT(**vivit_kwargs)

    def init_adapter(self, input_feature_dim: int) -> None:
        self.adapter = SequenceToVideoAdapter(
            input_feature_dim=int(input_feature_dim),
            out_channels=self._adapter_channels,
            small_h=self._adapter_small_h,
            small_w=self._adapter_small_w,
            target_num_frames=self._vivit_kwargs.get("num_frames", 8),
            target_image_size=self._vivit_kwargs.get("image_size", 224),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter is None:
            self.init_adapter(int(x.shape[-1]))
        x_vid = self.adapter(x)
        return self.backbone(x_vid)

