"""
vit_surgery.py
==============
Replace every self-attention block of a pretrained HuggingFace ViT with a
Tucker Attention block initialised via HOSVD (Appendix C.1.4).

Only the attention weights (Q,K,V,O) change -- patch embedding, MLPs,
LayerNorms and the classifier head are left untouched, exactly as in the paper
("In every case only the attention block was compressed").

Baselines for comparison (GQA, MLA) are intentionally NOT implemented here;
this module is the Tucker path. Add them the same way if you want the full
Figure 6-8 panel.
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

from models.tucker_attention import TuckerViTAttention


def load_vit(model_name: str, num_labels: int, device="cuda"):
    """Load a pretrained ViT classifier and reset the head to `num_labels`.

    Canonical endpoints (Table 5 / Appendix C.1.3 -- note the paper text has a
    copy/paste slip listing vit-large for every row; use the matching size):
        ViT-B/16  -> 'google/vit-base-patch16-224-in21k'   (d=768,  H=12, N=196)
        ViT-L/32  -> 'google/vit-large-patch32-224-in21k'  (d=1024, H=16, N=49)
        ViT-H/14  -> 'google/vit-huge-patch14-224-in21k'   (d=1280, H=16) # ~ 'G/14'
    """
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    return model.to(device)


def tuckerize_vit(model, ranks: Tuple[int, int, int],
                  ranks_post: Optional[Tuple[int, int, int]] = None,
                  share_kv_downproj: bool = False,
                  verbose: bool = True):
    """In-place: swap each `ViTAttention` for a Tucker block, HOSVD-initialised."""
    cfg = model.config
    d_model, n_heads = cfg.hidden_size, cfg.num_attention_heads
    layers = model.vit.encoder.layer

    mha_params = 0
    tuck_params = 0
    for layer in layers:
        old = layer.attention                       # ViTAttention
        # count MHA Q,K,V,O weights (+biases) of the block being replaced
        mha_params += sum(p.numel() for p in old.parameters())
        new = TuckerViTAttention.from_mha(
            old, ranks, ranks_post, share_kv_downproj).to(
            next(old.parameters()).device, dtype=next(old.parameters()).dtype)
        layer.attention = new
        tuck_params += new.num_attention_params()

    if verbose:
        print(f"[tuckerize] ranks={ranks} post={ranks_post or ranks} "
              f"share_kv={share_kv_downproj}")
        print(f"[tuckerize] attention params: {mha_params:,} (MHA) "
              f"-> {tuck_params:,} (Tucker)  "
              f"= {tuck_params / mha_params:.3f}x  "
              f"({mha_params / max(tuck_params,1):.2f}x smaller)")
    return model, dict(mha_attn_params=mha_params, tucker_attn_params=tuck_params)


def freeze_non_attention(model, train_head: bool = True):
    """Optional: freeze everything except the Tucker blocks (and the head).

    The paper fine-tunes the whole model in transfer learning; this helper is
    only for fast rank-sweep smoke tests where you want to isolate the attention
    factors."""
    for name, p in model.named_parameters():
        p.requires_grad = False
    for layer in model.vit.encoder.layer:
        for p in layer.attention.parameters():
            p.requires_grad = True
    if train_head:
        for p in model.classifier.parameters():
            p.requires_grad = True
    return model


def attention_param_report(model) -> int:
    """Total trainable params inside the (Tucker) attention blocks."""
    return sum(p.numel()
               for layer in model.vit.encoder.layer
               for p in layer.attention.parameters())
