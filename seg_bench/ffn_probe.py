"""FFN compression probe: what happens if the feed-forward weights (after attention) are Tucker/low-rank compressed?

Applied post-hoc to the fine-tuned baseline checkpoint, evaluated without retraining (same protocol as
"Tucker frozen"). Two schemes at a given FFN parameter ratio:
  matrix  - per-layer truncated SVD (Tucker-2 of one weight matrix)
  stacked - HOSVD (Tucker-3) of the layer-stacked tensor (L, out, in) of equally shaped FFN weights,
            i.e. factor matrices shared across layers
Weights are reconstructed densely: this measures accuracy only; the parameter ratio is analytic.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from data.sar_dataset import SAR_Dataset
from seg_bench.arch_analysis import category
from seg_bench.models import MODELS, build
from seg_bench.run import ROOT, CKPT_ROOT, OUT_ROOT, evaluate, make_loaders
from torch.utils.data import DataLoader


def ffn_linears(model):
    return [(n[:-7], m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and category(n + ".weight", model) == "ffn"
            and min(m.weight.shape) >= 8]


def lowrank(W: torch.Tensor, ratio: float):
    m, n = W.shape
    r = max(1, int(ratio * m * n / (m + n)))
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r], r * (m + n) / (m * n)


def stacked_tucker(Ws: torch.Tensor, ratio: float):
    L, m, n = Ws.shape

    def params(f):
        r1, r2, r3 = max(1, round(f * L)), max(1, round(f * m)), max(1, round(f * n))
        return r1, r2, r3, r1 * r2 * r3 + L * r1 + m * r2 + n * r3

    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if params(mid)[3] <= ratio * Ws.numel():
            lo = mid
        else:
            hi = mid
    r1, r2, r3, p = params(lo)
    core = Ws
    factors = []
    for mode, r in enumerate((r1, r2, r3)):
        unf = torch.movedim(Ws, mode, 0).reshape(Ws.shape[mode], -1)
        U = torch.linalg.svd(unf, full_matrices=False)[0][:, :r]
        factors.append(U)
    core = torch.einsum("lmn,la,mb,nc->abc", Ws, *factors)
    rec = torch.einsum("abc,la,mb,nc->lmn", core, *factors)
    return rec, p / Ws.numel(), (r1, r2, r3)


@torch.no_grad()
def apply(model, scheme: str, ratio: float):
    groups = defaultdict(list)
    for name, mod in ffn_linears(model):
        groups[tuple(mod.weight.shape)].append(mod)
    total_before = total_after = 0.0
    for shape, mods in groups.items():
        Ws = torch.stack([m.weight.data.float() for m in mods])
        if scheme == "matrix" or len(mods) < 2:
            new, ratios = [], []
            for W in Ws:
                r, pr = lowrank(W, ratio)
                new.append(r)
                ratios.append(pr)
            new = torch.stack(new)
            after = sum(ratios) * shape[0] * shape[1]
        else:
            new, pr, _ = stacked_tucker(Ws, ratio)
            after = pr * Ws.numel()
        for mod, w in zip(mods, new):
            mod.weight.data.copy_(w.to(mod.weight.dtype))
        total_before += Ws.numel()
        total_after += after
    return total_after / total_before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--ratios", default="0.5,0.25")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--data-root", default=str(ROOT / "Deep_SAR" / "versions" / "1"))
    args = ap.parse_args()
    device = torch.device("cuda")
    _, val_loader = make_loaders(args, device)
    ckpt = CKPT_ROOT / args.model / "baseline_best.pt"
    results = {}
    for scheme in ("none", "matrix", "stacked"):
        for ratio in ([1.0] if scheme == "none" else [float(r) for r in args.ratios.split(",")]):
            model = build(args.model).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            achieved = 1.0 if scheme == "none" else apply(model, scheme, ratio)
            m = evaluate(model, val_loader, device)
            results[f"{scheme}@{ratio}"] = dict(scheme=scheme, target_ratio=ratio, ffn_param_ratio=achieved,
                                                oil_spill_iou=m["oil_spill_iou"], f1=m["f1"],
                                                accuracy=m["accuracy"])
            print(args.model, scheme, ratio, f"param ratio {achieved:.3f} IoU {m['oil_spill_iou']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
    (OUT_ROOT / args.model / "ffn_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
