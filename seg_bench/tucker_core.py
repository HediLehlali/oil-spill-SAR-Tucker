"""Architecture-agnostic Tucker attention.

Per layer, the pre-softmax tensor W[h] = Wq_h Wk_h^T and the post-softmax tensor
Wt[h] = (Wv_h Wo_h)^T are stacked over heads into (H, d, d) tensors and
compressed with a truncated HOSVD (same maths as models/tucker_attention.py).

Differences from the ViT-only version:
  * separate query / key / value inputs (cross-attention, positional embeddings
    added to q/k only, SegFormer sequence reduction),
  * additive logit bias (Swin relative position bias, shifted-window masks),
  * the query bias is folded in exactly (key_bias) instead of being dropped,
  * ranks can be chosen per layer from the energy kept by the HOSVD spectrum.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _unfold(T: torch.Tensor, mode: int) -> torch.Tensor:
    return torch.movedim(T, mode, 0).reshape(T.shape[mode], -1)


def _mode_dot(T: torch.Tensor, M: torch.Tensor, mode: int) -> torch.Tensor:
    T2 = torch.movedim(T, mode, 0)
    out = (M @ T2.reshape(T2.shape[0], -1)).reshape((M.shape[0],) + T2.shape[1:])
    return torch.movedim(out, 0, mode)


# (U, S) of every mode unfolding, keyed per layer; independent of the ranks, so a
# rank/energy sweep only pays for the SVDs once per model.
_SVD_CACHE: dict = {}


def _mode_svds(T: torch.Tensor, cache_key):
    cached = _SVD_CACHE.get(cache_key) if cache_key is not None else None
    if cached is None:
        cached = []
        for mode in range(3):
            U, S, _ = torch.linalg.svd(_unfold(T, mode), full_matrices=False)
            cached.append((U.cpu(), S.cpu()))
        if cache_key is not None:
            _SVD_CACHE[cache_key] = cached
    return cached


def clear_svd_cache() -> None:
    _SVD_CACHE.clear()


def energy_ranks(svds, energy: float) -> Tuple[int, int, int]:
    """Smallest per-mode rank whose squared singular values keep `energy` of the total."""
    ranks = []
    for _, S in svds:
        cum = torch.cumsum(S.double() ** 2, dim=0)
        cum = cum / cum[-1].clamp_min(1e-30)
        rank = int(torch.searchsorted(cum, torch.tensor(energy, dtype=cum.dtype)).item()) + 1
        ranks.append(min(rank, len(S)))
    return tuple(ranks)


def hosvd_from_svds(T: torch.Tensor, svds, ranks: Sequence[int]):
    factors = [svds[m][0][:, : ranks[m]].to(T.device).contiguous() for m in range(3)]
    core = T
    for mode, U in enumerate(factors):
        core = _mode_dot(core, U.t(), mode)
    return core.contiguous(), factors


class TuckerAttentionCore(nn.Module):
    """Tucker-factorised multi-head attention (all heads share the factor matrices)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 ranks: Sequence[int], ranks_post: Sequence[int]):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_head
        self.ranks, self.ranks_post = tuple(ranks), tuple(ranks_post)
        r1, r2, r3 = self.ranks
        t1, t2, t3 = self.ranks_post
        self.C = nn.Parameter(torch.zeros(r1, r2, r3))
        self.U1 = nn.Parameter(torch.zeros(n_heads, r1))
        self.U2 = nn.Parameter(torch.zeros(d_model, r2))
        self.U3 = nn.Parameter(torch.zeros(d_model, r3))
        self.Ct = nn.Parameter(torch.zeros(t1, t2, t3))
        self.Ut1 = nn.Parameter(torch.zeros(n_heads, t1))
        self.Ut2 = nn.Parameter(torch.zeros(d_model, t2))
        self.Ut3 = nn.Parameter(torch.zeros(d_model, t3))
        self.key_bias = nn.Parameter(torch.zeros(n_heads, d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    @classmethod
    @torch.no_grad()
    def from_weights(cls, Wq, Wk, Wv, Wo, bq, bv, bo, n_heads: int, cache_key,
                     level=(1 / 3, 1 / 4), ranks=None, ranks_post=None, energy=None,
                     device="cuda"):
        """Wq/Wk/Wv: (H*dh, d) nn.Linear weights, Wo: (d, H*dh); biases may be None.

        Ranks come from, in priority order: explicit `ranks`, an HOSVD `energy`
        threshold, or `level=(a, b)` meaning r1 = ceil(a*H) and r2 = r3 = round(b*d_head).
        """
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        Wq, Wk, Wv, Wo = [w.detach().float().to(dev) for w in (Wq, Wk, Wv, Wo)]
        d_model = Wq.shape[1]
        d_head = Wq.shape[0] // n_heads
        W = torch.empty(n_heads, d_model, d_model, device=dev)
        Wt = torch.empty(n_heads, d_model, d_model, device=dev)
        for h in range(n_heads):
            sl = slice(h * d_head, (h + 1) * d_head)
            W[h] = Wq[sl].t() @ Wk[sl]
            Wt[h] = (Wv[sl].t() @ Wo[:, sl].t()).t()

        svd_pre = _mode_svds(W, (*cache_key, "pre"))
        svd_post = _mode_svds(Wt, (*cache_key, "post"))
        if ranks is not None:
            r_pre, r_post = tuple(ranks), tuple(ranks_post or ranks)
        elif energy is not None:
            r_pre = energy_ranks(svd_pre, energy)
            r_post = energy_ranks(svd_post, energy)
        else:
            a, b = level
            r_pre = r_post = (
                min(n_heads, max(1, math.ceil(a * n_heads))),
                min(d_model, max(2, round(b * d_head))),
                min(d_model, max(2, round(b * d_head))),
            )

        mod = cls(d_model, n_heads, d_head, r_pre, r_post).to(dev)
        C, (U1, U2, U3) = hosvd_from_svds(W, svd_pre, r_pre)
        Ct, (Ut1, Ut2, Ut3) = hosvd_from_svds(Wt, svd_post, r_post)
        for name, value in dict(C=C, U1=U1, U2=U2, U3=U3, Ct=Ct, Ut1=Ut1, Ut2=Ut2, Ut3=Ut3).items():
            getattr(mod, name).copy_(value)

        b_out = torch.zeros(d_model, device=dev) if bo is None else bo.detach().float().to(dev).clone()
        if bv is not None:
            bv = bv.detach().float().to(dev)
            for h in range(n_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                b_out += bv[sl] @ Wo[:, sl].t()
        mod.b_out.copy_(b_out)
        if bq is not None:
            bq = bq.detach().float().to(dev)
            for h in range(n_heads):
                sl = slice(h * d_head, (h + 1) * d_head)
                mod.key_bias[h].copy_(Wk[sl].t() @ bq[sl])
        return mod

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q_in (B,N,d), k_in/v_in (B,M,d); bias broadcastable to (B,H,N,M). Returns (B,N,d)."""
        q = q_in @ self.U2
        k = k_in @ self.U3
        v = v_in @ self.Ut3
        G = torch.einsum("ha,abc->hbc", self.U1, self.C)
        Gt = torch.einsum("ha,abc->hbc", self.Ut1, self.Ct)

        logits = torch.einsum("bnr,hrs->bhns", q, G) @ k.transpose(1, 2).unsqueeze(1)
        logits = logits + torch.einsum("bmd,hd->bhm", k_in, self.key_bias).unsqueeze(2)
        logits = logits / math.sqrt(self.d_head)
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        probs = logits.float().softmax(dim=-1).to(v.dtype)

        ctx = probs @ v.unsqueeze(1)
        out = torch.einsum("bhns,hts->bnt", ctx, Gt)
        return out @ self.Ut2.t() + self.b_out

    def num_attention_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
