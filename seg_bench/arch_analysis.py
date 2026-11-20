"""Parameter inventory by role (attention / FFN / other encoder / decoder) and FFN Tucker potential."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

from seg_bench.models import MODELS, build

ATTN = re.compile(
    r"(attention\.self\.(query|key|value)|attention\.output\.dense|attn\.(qkv|proj)\b|"
    r"self_attn\.(q_proj|k_proj|v_proj|out_proj)|cross_attn\.(in_proj|out_proj))")
FFN = re.compile(
    r"(mlp\.(dense1|dense2|dwconv|fc1|fc2|linear1|linear2)|\.intermediate\.dense|"
    r"layers?\.\d+\.output\.dense|blocks?\.\d+\.output\.dense|blocks?\.\d+\.mlp|\.fc1\.|\.fc2\.|"
    r"layers\.\d+\.blocks\.\d+\.mlp)")


def category(name: str, model) -> str:
    # HF Swin: encoder.layers.i.blocks.j.{intermediate.dense, output.dense}; attention.output.dense is attention
    if ATTN.search(name):
        return "attention"
    if FFN.search(name) and "attention" not in name:
        return "ffn"
    if name.startswith(model.encoder_prefixes):
        return "encoder_other"
    return "decoder_head"


def ffn_weights(model):
    """Return list of (name, weight[out, in]) for FFN linear layers."""
    out = []
    for name, p in model.named_parameters():
        if p.dim() == 2 and name.endswith("weight") and category(name, model) == "ffn":
            out.append((name, p.detach().float()))
    return out


def svd_rank(S: torch.Tensor, energy: float) -> int:
    c = torch.cumsum(S.double() ** 2, 0)
    c = c / c[-1]
    return int(torch.searchsorted(c, torch.tensor(energy, dtype=c.dtype)).item()) + 1


def stacked_tucker(weights, energy):
    """HOSVD of the layer-stacked FFN tensor (L, out, in); returns ranks and parameter ratio."""
    T = torch.stack(weights)
    L, m, n = T.shape
    ranks = []
    for mode in range(3):
        unf = torch.movedim(T, mode, 0).reshape(T.shape[mode], -1)
        S = torch.linalg.svdvals(unf)
        ranks.append(svd_rank(S, energy))
    r1, r2, r3 = ranks
    params = r1 * r2 * r3 + L * r1 + m * r2 + n * r3
    return ranks, params / T.numel()


def analyse(key: str) -> dict:
    model = build(key)
    cats = defaultdict(int)
    for name, p in model.named_parameters():
        cats[category(name, model)] += p.numel()
    total = sum(cats.values())

    # group FFN weights by (shape, position: fc1 vs fc2) for cross-layer Tucker
    groups = defaultdict(list)
    per_matrix = {0.90: [0, 0], 0.95: [0, 0]}
    for name, w in ffn_weights(model):
        if w.shape[0] < 8 or w.shape[1] < 8:
            continue
        S = torch.linalg.svdvals(w)
        for e in per_matrix:
            r = svd_rank(S, e)
            per_matrix[e][0] += min(r * (w.shape[0] + w.shape[1]), w.numel())
            per_matrix[e][1] += w.numel()
        groups[tuple(w.shape)].append(w)

    stacked = {0.90: [0, 0], 0.95: [0, 0]}
    detail = []
    for shape, ws in groups.items():
        if len(ws) < 2:
            continue
        for e in stacked:
            ranks, ratio = stacked_tucker(ws, e)
            stacked[e][0] += min(ratio, 1.0) * len(ws) * ws[0].numel()
            stacked[e][1] += len(ws) * ws[0].numel()
            if e == 0.95:
                detail.append(dict(shape=list(shape), layers=len(ws), ranks=ranks, ratio=min(ratio, 1.0)))
    return dict(
        key=key, pretty=model.pretty, total=total,
        categories={k: v for k, v in cats.items()},
        ffn_matrix_svd={str(e): v[0] / max(v[1], 1) for e, v in per_matrix.items()},
        ffn_stacked_tucker={str(e): v[0] / max(v[1], 1) for e, v in stacked.items()},
        ffn_stacked_detail=detail,
    )


def main(keys):
    out = {}
    for key in keys:
        out[key] = analyse(key)
        r = out[key]
        c = r["categories"]
        print(r["pretty"], {k: f"{v / 1e6:.2f}M ({100 * v / r['total']:.0f}%)" for k, v in c.items()},
              "FFN matrix-SVD ratio", {k: round(v, 2) for k, v in r["ffn_matrix_svd"].items()},
              "stacked Tucker ratio", {k: round(v, 2) for k, v in r["ffn_stacked_tucker"].items()}, flush=True)
    path = Path("outputs/Seg5_Runs/architecture_analysis.json")
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
