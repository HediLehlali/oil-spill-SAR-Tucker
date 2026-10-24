# tucker_attention.py
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _unfold(T: torch.Tensor, mode: int) -> torch.Tensor:
    return torch.movedim(T, mode, 0).reshape(T.shape[mode], -1)


def _mode_dot(T: torch.Tensor, M: torch.Tensor, mode: int) -> torch.Tensor:
    T2 = torch.movedim(T, mode, 0)
    out = (M @ T2.reshape(T2.shape[0], -1)).reshape((M.shape[0],) + T2.shape[1:])
    return torch.movedim(out, 0, mode)


# Full left singular vectors per (model, layer, tensor). The SVDs do not depend on
# the ranks or the seed, so a rank sweep only pays for them once.
_HOSVD_CACHE: dict = {}


def hosvd(T: torch.Tensor, ranks: Tuple[int, int, int], cache_key=None):
    """Truncated higher-order SVD. Returns (core, [U1, U2, U3])."""
    full = _HOSVD_CACHE.get(cache_key) if cache_key is not None else None
    if full is None:
        full = [torch.linalg.svd(_unfold(T, m), full_matrices=False)[0].cpu()
                for m in range(3)]
        if cache_key is not None:
            _HOSVD_CACHE[cache_key] = full
    factors = [full[m][:, :r].to(T.device).contiguous() for m, r in enumerate(ranks)]
    core = T
    for mode, U in enumerate(factors):
        core = _mode_dot(core, U.t(), mode)
    return core.contiguous(), factors


class TuckerViTAttention(nn.Module):
    """Tucker-factorised self-attention block for a ViT encoder layer."""

    def __init__(self, d_model: int, n_heads: int,
                 ranks: Tuple[int, int, int],
                 ranks_post: Optional[Tuple[int, int, int]] = None,
                 share_kv_downproj: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0
        r1, r2, r3 = ranks
        rt1, rt2, rt3 = ranks_post if ranks_post is not None else ranks
        assert 0 < r1 <= n_heads and 0 < rt1 <= n_heads
        self.ranks, self.ranks_post = (r1, r2, r3), (rt1, rt2, rt3)
        self.share_kv_downproj = share_kv_downproj

        # pre-softmax factors
        self.C  = nn.Parameter(torch.zeros(r1, r2, r3))
        self.U1 = nn.Parameter(torch.zeros(n_heads, r1))
        self.U2 = nn.Parameter(torch.zeros(d_model, r2))
        self.U3 = nn.Parameter(torch.zeros(d_model, r3))

        # post-softmax factors
        self.Ct  = nn.Parameter(torch.zeros(rt1, rt2, rt3))
        self.Ut1 = nn.Parameter(torch.zeros(n_heads, rt1))
        self.Ut2 = nn.Parameter(torch.zeros(d_model, rt2))
        if share_kv_downproj:
            assert rt3 == r3
            self.Ut3 = None
        else:
            self.Ut3 = nn.Parameter(torch.zeros(d_model, rt3))

        self.b_out = nn.Parameter(torch.zeros(d_model))

    @property
    def _Ut3(self) -> torch.Tensor:
        return self.U3 if self.share_kv_downproj else self.Ut3

    @classmethod
    def from_mha(cls, vit_attention, ranks, ranks_post=None,
                 share_kv_downproj: bool = False, cache_key=None):
        """Build a Tucker block from a pretrained ViTAttention via HOSVD.

        The decomposition runs on the GPU when available (~30x faster than CPU).
        `cache_key` identifies the layer so its SVDs can be reused across ranks.
        """
        self_attn = vit_attention.attention
        out_dense = vit_attention.output.dense
        d_model = self_attn.query.in_features
        n_heads = getattr(self_attn, "num_attention_heads")
        d_head = d_model // n_heads

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Wq = self_attn.query.weight.data.to(dev)
        Wk = self_attn.key.weight.data.to(dev)
        Wv = self_attn.value.weight.data.to(dev)
        Wo = out_dense.weight.data.to(dev)

        W  = torch.empty(n_heads, d_model, d_model, device=dev)
        Wt = torch.empty(n_heads, d_model, d_model, device=dev)
        for i in range(n_heads):
            sl = slice(i * d_head, (i + 1) * d_head)
            Wq_i = Wq[sl, :].t()
            Wk_i = Wk[sl, :].t()
            Wv_i = Wv[sl, :].t()
            Wo_i = Wo[:, sl].t()
            W[i]  = Wq_i @ Wk_i.t()
            Wt[i] = (Wv_i @ Wo_i).t()

        mod = cls(d_model, n_heads, ranks, ranks_post, share_kv_downproj)
        with torch.no_grad():
            key = (lambda s: None if cache_key is None else (*cache_key, s))
            C, (U1, U2, U3) = hosvd(W, mod.ranks, key("pre"))
            mod.C.copy_(C); mod.U1.copy_(U1); mod.U2.copy_(U2); mod.U3.copy_(U3)
            Ct, (Ut1, Ut2, Ut3) = hosvd(Wt, mod.ranks_post, key("post"))
            mod.Ct.copy_(Ct); mod.Ut1.copy_(Ut1); mod.Ut2.copy_(Ut2)
            if not share_kv_downproj:
                mod.Ut3.copy_(Ut3)

            b_out = out_dense.bias.data.to(dev).clone()
            if self_attn.value.bias is not None:
                bv = self_attn.value.bias.data.to(dev)
                for i in range(n_heads):
                    sl = slice(i * d_head, (i + 1) * d_head)
                    b_out = b_out + bv[sl] @ Wo[:, sl].t()
            mod.b_out.copy_(b_out)
        return mod

    def forward(self, hidden_states: torch.Tensor,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False):
        X = hidden_states
        q = X @ self.U2
        k = X @ self.U3
        v = X @ self._Ut3

        G  = torch.einsum('ha,abc->hbc', self.U1,  self.C)
        Gt = torch.einsum('ha,abc->hbc', self.Ut1, self.Ct)

        logits = torch.einsum('bnr,hrs,bms->bhnm', q, G, k)
        logits = logits / math.sqrt(self.d_head)
        probs = logits.softmax(dim=-1)
        if head_mask is not None:
            probs = probs * head_mask

        ctx = torch.einsum('bhnm,bms->bhns', probs, v)
        ctx = torch.einsum('bhns,hts->bhnt', ctx, Gt)
        out = torch.einsum('bhnt,dt->bhnd', ctx, self.Ut2)
        out = out.sum(dim=1) + self.b_out

        return (out, probs) if output_attentions else (out,)

    def num_attention_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def mha_attention_params(d_model: int) -> int:
        return 4 * d_model * d_model
