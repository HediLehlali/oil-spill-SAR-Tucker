"""The five segmentation architectures behind one interface.

Every wrapper exposes
  forward(images)            -> logits (B, 2, H, W) used for evaluation
  training_loss(images, m)   -> scalar loss for one batch
  encoder_prefixes           -> parameter-name prefixes treated as "encoder" (lower LR)
  tucker_families            -> attention blocks that get Tucker-compressed
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.train_seg import dice_loss

warnings.filterwarnings("ignore", message=".*symlinks.*")
IMG = 224


class SegWrapper(nn.Module):
    key = ""
    pretty = ""
    hf_id = ""
    encoder_prefixes: tuple = ()
    tucker_families: tuple = ()
    uses_native_loss = False

    def __init__(self):
        super().__init__()
        self.class_weights: torch.Tensor | None = None

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self.class_weights = weights

    def training_loss(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        logits = self(images).float()
        ce = F.cross_entropy(logits, masks, weight=self.class_weights)
        return ce + dice_loss(logits, masks)

    def split_params(self):
        enc, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if name.startswith(self.encoder_prefixes) else head).append((name, p))
        return enc, head


class SegFormerB2(SegWrapper):
    key, pretty = "segformer_b2", "SegFormer-B2"
    hf_id = "nvidia/segformer-b2-finetuned-ade-512-512"
    encoder_prefixes = ("net.segformer.",)
    tucker_families = ("segformer",)

    def __init__(self, num_classes=2):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.net = SegformerForSemanticSegmentation.from_pretrained(
            self.hf_id, num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class UperNetSwinT(SegWrapper):
    key, pretty = "upernet_swin_t", "Swin-T + UperNet"
    hf_id = "openmmlab/upernet-swin-tiny"
    encoder_prefixes = ("net.backbone.",)
    tucker_families = ("hf_swin",)

    def __init__(self, num_classes=2):
        super().__init__()
        from transformers import UperNetForSemanticSegmentation

        self.net = UperNetForSemanticSegmentation.from_pretrained(
            self.hf_id, num_labels=num_classes, ignore_mismatched_sizes=True,
            use_auxiliary_head=False)

    def forward(self, x):
        return self.net(pixel_values=x).logits


class Mask2FormerSwinT(SegWrapper):
    key, pretty = "mask2former_swin_t", "Mask2Former (Swin-T)"
    hf_id = "facebook/mask2former-swin-tiny-ade-semantic"
    encoder_prefixes = ("net.model.pixel_level_module.encoder.",)
    tucker_families = ("hf_swin", "m2f_self", "m2f_cross")
    uses_native_loss = True

    def __init__(self, num_classes=2):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation

        self.num_classes = num_classes
        self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.hf_id, num_labels=num_classes, ignore_mismatched_sizes=True)

    @staticmethod
    def _semantic(out, size):
        cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
        msk = out.masks_queries_logits.float().sigmoid()
        seg = torch.einsum("bqc,bqhw->bchw", cls, msk)
        return F.interpolate(seg, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self._semantic(self.net(pixel_values=x), x.shape[-2:])

    def training_loss(self, images, masks):
        labels, classes = [], []
        for m in masks:
            present = torch.unique(m)
            labels.append(torch.stack([(m == c).float() for c in present]))
            classes.append(present.long())
        return self.net(pixel_values=images, mask_labels=labels, class_labels=classes).loss


class SwinUNETR2D(SegWrapper):
    key, pretty = "swin_unetr", "Swin-UNETR (2D)"
    hf_id = "MONAI/swin_unetr_btcv_segmentation"
    encoder_prefixes = ("net.swinViT.",)
    tucker_families = ("monai_swin",)

    def __init__(self, num_classes=2):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.net = SwinUNETR(in_channels=1, out_channels=num_classes, feature_size=48,
                             depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                             spatial_dims=2, use_checkpoint=False)
        self.transfer_report = load_btcv_weights_2d(self.net)

    def forward(self, x):
        return self.net(x[:, :1])


def load_btcv_weights_2d(net: nn.Module) -> Dict[str, int]:
    """Initialise a 2D Swin-UNETR from the 3D MONAI BTCV bundle.

    Linear/LayerNorm tensors copy 1:1. 3D->2D conversions: conv kernels are summed
    over depth, the relative-position table keeps its depth-0 plane, and the patch
    merging projection keeps/sums the 4 (of 8) neighbour groups that exist in 2D.
    Only the Swin encoder and the skip encoders are transferred; the UNETR decoder
    and the 2-class output layer start from scratch.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("MONAI/swin_unetr_btcv_segmentation", "models/model.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        state = state.state_dict()
    state = state.get("state_dict", state)
    own = net.state_dict()
    copied = converted = skipped = 0
    skipped_keys: list = []
    new_state = {}
    for key, src in state.items():
        if not key.startswith(("swinViT.", "encoder")) or key not in own:
            continue
        dst = own[key]
        value = None
        if src.shape == dst.shape:
            value = src
        elif key.endswith("relative_position_bias_table"):
            w = int(round(src.shape[0] ** (1 / 3)))
            value = src.reshape(w, w, w, -1)[(w - 1) // 2].reshape(-1, src.shape[-1])
        elif key.endswith("downsample.reduction.weight"):
            # 3D groups (d,h,w): idx = 4*d + 2*h + 1*w in MONAI's concat order x0..x7
            order = {(0, 0, 0): 0, (1, 0, 0): 1, (0, 1, 0): 2, (0, 0, 1): 3,
                     (1, 1, 0): 4, (1, 0, 1): 5, (0, 1, 1): 6, (1, 1, 1): 7}
            c = src.shape[1] // 8
            blocks = src.reshape(src.shape[0], 8, c)
            value = torch.cat([blocks[:, order[(0, h, w)]] + blocks[:, order[(1, h, w)]]
                               for h, w in ((0, 0), (1, 0), (0, 1), (1, 1))], dim=1)
        elif key.endswith("downsample.norm.weight") or key.endswith("downsample.norm.bias"):
            order = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
            c = src.shape[0] // 8
            blocks = src.reshape(8, c)
            idx = {o: i for i, o in enumerate(order)}
            value = torch.cat([(blocks[idx[(0, h, w)]] + blocks[idx[(1, h, w)]]) / 2
                               for h, w in ((0, 0), (1, 0), (0, 1), (1, 1))])
        elif src.dim() == dst.dim() + 1 and src.shape[:2] == dst.shape[:2]:
            value = src.sum(dim=2)
        if value is not None and value.shape == dst.shape:
            new_state[key] = value
            converted += int(value is not src)
            copied += int(value is src)
        else:
            skipped += 1
            skipped_keys.append(f"{key}{tuple(src.shape)}->{tuple(dst.shape)}")
    net.load_state_dict(new_state, strict=False)
    return dict(copied=copied, converted_3d_to_2d=converted, skipped=skipped,
                skipped_keys=skipped_keys,
                transferred_tensors=len(new_state), own_tensors=len(own))


class TransUNet(SegWrapper):
    """R50 + ViT-B/16 encoder with the cascaded-upsampler decoder of TransUNet (Chen et al. 2021)."""

    key, pretty = "transunet", "TransUNet (R50-ViT-B/16)"
    hf_id = "vit_base_r50_s16_224.orig_in21k"
    encoder_prefixes = ("backbone.", "proj.", "blocks.", "norm.", "pos_embed")
    tucker_families = ("timm_vit",)

    def __init__(self, num_classes=2):
        super().__init__()
        import timm

        vit = timm.create_model(self.hf_id, pretrained=True, num_classes=0)
        self.backbone = vit.patch_embed.backbone
        self.proj = vit.patch_embed.proj
        self.pos_embed = nn.Parameter(vit.pos_embed[:, 1:].detach().clone())
        self.blocks = vit.blocks
        self.norm = vit.norm
        del self.backbone.head
        self.backbone.norm = nn.Identity()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

        self.conv_more = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.dec1 = block(512 + 512, 256)
        self.dec2 = block(256 + 256, 128)
        self.dec3 = block(128 + 64, 64)
        self.dec4 = block(64, 16)
        self.seg_head = nn.Conv2d(16, num_classes, 3, padding=1)

    def forward(self, x):
        stem = self.backbone.stem
        s2 = stem.norm(stem.conv(x))                  # 1/2, 64 ch
        f = stem.pool(s2)
        s4 = self.backbone.stages[0](f)               # 1/4, 256 ch
        s8 = self.backbone.stages[1](s4)              # 1/8, 512 ch
        s16 = self.backbone.stages[2](s8)             # 1/16, 1024 ch
        t = self.proj(s16)
        b, c, h, w = t.shape
        t = t.flatten(2).transpose(1, 2) + self.pos_embed
        t = self.norm(self.blocks(t))
        y = self.conv_more(t.transpose(1, 2).reshape(b, c, h, w))
        up = lambda z: F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([up(y), s8], 1))
        y = self.dec2(torch.cat([up(y), s4], 1))
        y = self.dec3(torch.cat([up(y), s2], 1))
        y = self.dec4(up(y))
        return self.seg_head(y)


MODELS: Dict[str, type] = {
    cls.key: cls for cls in (SegFormerB2, UperNetSwinT, Mask2FormerSwinT, SwinUNETR2D, TransUNet)
}


def build(key: str, num_classes: int = 2) -> SegWrapper:
    return MODELS[key](num_classes=num_classes)
