from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_surgery import load_vit, tuckerize_vit


class ViT_Tucker_Seg(nn.Module):
    """ViT segmentation with a detail branch for small, thin oil slicks."""

    def __init__(
        self,
        num_classes: int = 5,
        ranks: Tuple[int, int, int] = (4, 64, 64),
        model_name: str = "facebook/dinov2-large",
        use_tucker: bool = False,
    ) -> None:
        super().__init__()
        self.vit = load_vit(model_name, num_labels=num_classes, device="cpu")
        if use_tucker:
            self.vit, _ = tuckerize_vit(self.vit, ranks, verbose=True)

        hidden_size = self.vit.config.hidden_size
        self.patch_size = self.vit.config.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]

        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.transformer_projection = nn.Conv2d(hidden_size, 128, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
        pixel_values = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

        if self.vit.config.model_type != "dinov2":
            outputs = self.vit.vit(
                pixel_values=pixel_values,
                interpolate_pos_encoding=True,
            ).last_hidden_state
        else:
            outputs = self.vit.vit(pixel_values=pixel_values).last_hidden_state
        patch_tokens = outputs[:, 1:, :]
        grid_height = pixel_values.shape[-2] // self.patch_size
        grid_width = pixel_values.shape[-1] // self.patch_size
        if patch_tokens.shape[1] != grid_height * grid_width:
            raise RuntimeError(
                "The ViT token count does not match its input patch grid: "
                f"{patch_tokens.shape[1]} tokens for {grid_height}x{grid_width}"
            )

        features = patch_tokens.transpose(1, 2).reshape(
            x.shape[0], patch_tokens.shape[2], grid_height, grid_width
        )
        detail_features = self.detail_encoder(x)
        features = F.interpolate(
            self.transformer_projection(features),
            size=detail_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fusion(features + detail_features)
        logits = F.interpolate(
            self.decoder(fused),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return logits
