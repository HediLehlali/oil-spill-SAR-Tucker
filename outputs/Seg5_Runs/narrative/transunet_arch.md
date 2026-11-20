### Architecture and what is compressed

**Encoder.** A ResNet-50 (BiT-style: weight-standardised convs + GroupNorm) provides a 3-level feature pyramid (½: 64 ch, ¼: 256 ch, ⅛: 512 ch) and a 1/16 map (1024 ch). The 1/16 map is projected to 768 channels, flattened to 196 tokens, given a learned positional embedding and passed through **12 standard ViT-B/16 blocks** (12 heads, MLP 768-3072-768). No class token is used. Weights: timm `vit_base_r50_s16_224.orig_in21k` (the same R50+ViT-B/16 ImageNet-21k checkpoint the TransUNet paper starts from); the official TransUNet repository has no HF checkpoint, so the decoder is re-implemented here.

**What Tucker touches.** The 12 ViT multi-head attention blocks (28.35M parameters, 27 % of the model) — exactly the setting of the earlier ViT-S/B/L study.

**After the attention (not compressed).** Output projection (folded), residual + LayerNorm, the **ViT MLP** (56.7M, 54 % of the model, twice the attention), the R50 backbone (12.9M) and the **cascaded upsampler**: 3×3 conv 768→512, then four stages of (bilinear ×2, concat skip from R50 at ⅛/¼/½, two 3×3 conv-BN-ReLU) with 256-128-64-16 channels, and a 3×3 segmentation head (7.4M parameters).

**FFN Tucker opportunity.** The 12 identical (768×3072) MLP matrices are the cleanest Tucker target in the benchmark: a single (12, 768, 3072) tensor (and its transpose partner) holds 56.7M parameters, 2× the attention budget.
