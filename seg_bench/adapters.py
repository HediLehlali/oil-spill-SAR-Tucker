"""Drop-in Tucker replacements for the attention blocks of each benchmarked architecture."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from seg_bench.tucker_core import TuckerAttentionCore


def _bias_of(linear: nn.Linear):
    return None if linear.bias is None else linear.bias


def _window_bias(table, index, n_heads, n, mask, batch):
    """Relative-position bias (+ shifted-window mask) as a (B,H,N,N) additive logit bias."""
    rel = table[index.reshape(-1)].reshape(n, n, n_heads).permute(2, 0, 1)
    rel = rel.unsqueeze(0)
    if mask is None:
        return rel
    n_win = mask.shape[0]
    bias = rel + mask.unsqueeze(1)
    return bias.unsqueeze(0).expand(batch // n_win, n_win, n_heads, n, n).reshape(batch, n_heads, n, n)


class SegformerTuckerAttention(nn.Module):
    def __init__(self, orig, cache_key, **kw):
        super().__init__()
        sa = orig.self
        self.core = TuckerAttentionCore.from_weights(
            sa.query.weight, sa.key.weight, sa.value.weight, orig.output.dense.weight,
            _bias_of(sa.query), _bias_of(sa.value), _bias_of(orig.output.dense),
            sa.num_attention_heads, cache_key, **kw,
        )
        self.sr_ratio = sa.sr_ratio
        if sa.sr_ratio > 1:
            self.sr, self.layer_norm = sa.sr, sa.layer_norm

    def forward(self, hidden_states, height, width, output_attentions=False):
        kv = hidden_states
        if self.sr_ratio > 1:
            b, n, c = hidden_states.shape
            kv = hidden_states.permute(0, 2, 1).reshape(b, c, height, width)
            kv = self.sr(kv).reshape(b, c, -1).permute(0, 2, 1)
            kv = self.layer_norm(kv)
        return (self.core(hidden_states, kv, kv),)


class HFSwinTuckerAttention(nn.Module):
    def __init__(self, orig, cache_key, **kw):
        super().__init__()
        sa = orig.self
        self.core = TuckerAttentionCore.from_weights(
            sa.query.weight, sa.key.weight, sa.value.weight, orig.output.dense.weight,
            _bias_of(sa.query), _bias_of(sa.value), _bias_of(orig.output.dense),
            sa.num_attention_heads, cache_key, **kw,
        )
        self.table = sa.relative_position_bias_table
        self.register_buffer("index", sa.relative_position_index.clone(), persistent=False)
        self.n_heads = sa.num_attention_heads

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        b, n, _ = hidden_states.shape
        bias = _window_bias(self.table, self.index, self.n_heads, n, attention_mask, b)
        return (self.core(hidden_states, hidden_states, hidden_states, bias),)


class MonaiSwinTuckerAttention(nn.Module):
    def __init__(self, orig, cache_key, **kw):
        super().__init__()
        d = orig.dim
        w, bq = orig.qkv.weight, orig.qkv.bias
        self.core = TuckerAttentionCore.from_weights(
            w[:d], w[d:2 * d], w[2 * d:], orig.proj.weight,
            None if bq is None else bq[:d], None if bq is None else bq[2 * d:],
            orig.proj.bias, orig.num_heads, cache_key, **kw,
        )
        self.table = orig.relative_position_bias_table
        self.register_buffer("index", orig.relative_position_index.clone(), persistent=False)
        self.n_heads = orig.num_heads

    def forward(self, x, mask):
        b, n, _ = x.shape
        bias = _window_bias(self.table, self.index[:n, :n], self.n_heads, n, mask, b)
        return self.core(x, x, x, bias)


class TimmViTTuckerAttention(nn.Module):
    def __init__(self, orig, cache_key, **kw):
        super().__init__()
        d = orig.qkv.in_features
        w, b = orig.qkv.weight, orig.qkv.bias
        self.core = TuckerAttentionCore.from_weights(
            w[:d], w[d:2 * d], w[2 * d:], orig.proj.weight,
            None if b is None else b[:d], None if b is None else b[2 * d:],
            orig.proj.bias, orig.num_heads, cache_key, **kw,
        )

    def forward(self, x, *args, **kwargs):
        return self.core(x, x, x)


class M2FSelfAttentionTucker(nn.Module):
    """Mask2Former decoder self-attention (sequence-first, positional embedding on q/k only)."""

    def __init__(self, orig, cache_key, **kw):
        super().__init__()
        self.core = TuckerAttentionCore.from_weights(
            orig.q_proj.weight, orig.k_proj.weight, orig.v_proj.weight, orig.out_proj.weight,
            orig.q_proj.bias, orig.v_proj.bias, orig.out_proj.bias,
            orig.num_heads, cache_key, **kw,
        )

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                key_value_states=None, key_value_position_embeddings=None,
                output_attentions=False):
        x = hidden_states.permute(1, 0, 2)
        qk = x if position_embeddings is None else x + position_embeddings.permute(1, 0, 2)
        out = self.core(qk, qk, x)
        return out.permute(1, 0, 2), None


class M2FCrossAttentionTucker(nn.Module):
    """Mask2Former masked cross-attention (nn.MultiheadAttention, sequence-first)."""

    def __init__(self, orig: nn.MultiheadAttention, cache_key, **kw):
        super().__init__()
        d = orig.embed_dim
        w, b = orig.in_proj_weight, orig.in_proj_bias
        self.core = TuckerAttentionCore.from_weights(
            w[:d], w[d:2 * d], w[2 * d:], orig.out_proj.weight,
            None if b is None else b[:d], None if b is None else b[2 * d:],
            orig.out_proj.bias, orig.num_heads, cache_key, **kw,
        )
        self.n_heads = orig.num_heads

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, **kwargs):
        q, k, v = (t.permute(1, 0, 2) for t in (query, key, value))
        bias = None
        if attn_mask is not None:
            b, n, m = q.shape[0], q.shape[1], k.shape[1]
            mask = attn_mask.reshape(b, self.n_heads, n, m) if attn_mask.dim() == 3 else attn_mask
            if mask.dtype == torch.bool:
                bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(mask, float("-inf"))
            else:
                bias = mask
        return self.core(q, k, v, bias).permute(1, 0, 2), None


def _is(cls_name: str) -> Callable[[nn.Module], bool]:
    return lambda module: type(module).__name__ == cls_name


# family -> (predicate, adapter). Each predicate must match only the attention
# block itself so that nested projections are replaced together with it.
FAMILIES = {
    "segformer": (_is("SegformerAttention"), SegformerTuckerAttention),
    "hf_swin": (_is("SwinAttention"), HFSwinTuckerAttention),
    "monai_swin": (_is("WindowAttention"), MonaiSwinTuckerAttention),
    "timm_vit": (lambda m: type(m).__name__ == "Attention" and hasattr(m, "qkv"), TimmViTTuckerAttention),
    "m2f_self": (_is("Mask2FormerAttention"), M2FSelfAttentionTucker),
    "m2f_cross": (
        lambda m: isinstance(m, nn.MultiheadAttention) and getattr(m, "_qkv_same_embed_dim", False),
        M2FCrossAttentionTucker,
    ),
}


@torch.no_grad()
def tuckerize(model: nn.Module, families, tag: str, **kw) -> dict:
    """Swap every matching attention block for its Tucker version, in place.

    Returns parameter statistics and the per-layer ranks.
    """
    targets = []
    for name, module in model.named_modules():
        for fam in families:
            pred, adapter = FAMILIES[fam]
            if pred(module):
                targets.append((name, module, adapter, fam))
                break

    dense = tucker = 0
    layers = []
    for name, module, adapter, fam in targets:
        # Q/K/V/O projections only: position-bias tables and SegFormer's sequence
        # reduction conv + norm are kept as-is by the Tucker block
        kept = ("relative_position_bias_table", ".sr.", ".layer_norm.")
        before = sum(p.numel() for n, p in module.named_parameters()
                     if not any(k in f".{n}" for k in kept))
        new = adapter(module, (tag, name), **kw)
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, new)
        after = new.core.num_attention_params()
        dense += before
        tucker += after
        layers.append(dict(name=name, family=fam, dense=before, tucker=after,
                           ranks=new.core.ranks, ranks_post=new.core.ranks_post))
    return dict(dense_attn_params=dense, tucker_attn_params=tucker,
                compression=dense / max(tucker, 1), n_layers=len(layers), layers=layers)
