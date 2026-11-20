"""Extra unfrozen run at another Tucker level, reusing the finished baseline of seg_bench.run.

Usage: python -m seg_bench.extra_unfrozen --model segformer_b2 --level L1_h1/3_d1/4 --epochs 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seg_bench.adapters import tuckerize
from seg_bench.models import MODELS, build
from seg_bench.run import (BASE_LRS, CKPT_ROOT, LEVELS, OUT_ROOT, ROOT, class_weights, evaluate, log,
                           make_loaders, profile_model, train_config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--data-root", default=str(ROOT / "Deep_SAR" / "versions" / "1"))
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = OUT_ROOT / args.model
    ckpt_dir = CKPT_ROOT / args.model
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    name = f"unfrozen_{args.level.split('_')[0]}"
    fh = open(out_dir / "run_log.txt", "a", encoding="utf-8")
    loaders = make_loaders(args, device)
    weights = class_weights(loaders[0], device, CKPT_ROOT / f"class_weights_{args.size}_{args.subset}.json")
    enc_lr, head_lr = BASE_LRS[args.model]

    m = build(args.model).to(device)
    m.load_state_dict(torch.load(summary["baseline"]["best_path"], map_location=device))
    tuckerize(m, m.tucker_families, args.model, level=LEVELS[args.level])
    info = train_config(m, name, args, loaders, weights, (enc_lr / 10, head_lr / 10, enc_lr),
                        out_dir, ckpt_dir, fh)
    m.load_state_dict(torch.load(info["best_path"], map_location=device))
    metrics = evaluate(m, loaders[1], device)
    summary.setdefault("extra_unfrozen", {})[args.level] = dict(
        **info, metrics=metrics, profile=profile_model(m, loaders[1], device, args.size), csv=f"{name}_epochs.csv")
    summary_path.write_text(json.dumps(summary, indent=1))
    log(f"{name} best IoU={metrics['oil_spill_iou']:.4f}", fh)


if __name__ == "__main__":
    main()
