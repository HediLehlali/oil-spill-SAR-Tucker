"""Baseline / Tucker-frozen / Tucker-unfrozen benchmark for one segmentation model.

  baseline        full model fine-tuned on the SAR oil-spill data
  tucker_frozen   attention compressed with Tucker, evaluated as-is (no training)
  tucker_unfrozen the compressed model fine-tuned end-to-end

Usage:  python -m seg_bench.run --model segformer_b2 [--epochs 20] [--batch-size 8]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.sar_dataset import SAR_Dataset
from seg_bench.adapters import tuckerize
from seg_bench.models import MODELS, SegWrapper, build
from seg_bench.plots import plot_comparison
from seg_bench.tucker_core import clear_svd_cache
from train.train_seg import _class_weights, evaluate_seg

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "Seg5_Runs"
CKPT_ROOT = ROOT / "checkpoints" / "seg5"

# (encoder LR, decoder/head LR) used for the baseline fine-tune
BASE_LRS = {
    "segformer_b2": (6e-5, 6e-4),
    "upernet_swin_t": (6e-5, 6e-4),
    "mask2former_swin_t": (1e-5, 1e-4),
    "swin_unetr": (1e-4, 1e-3),
    "transunet": (5e-5, 5e-4),
}
CLIP = {"mask2former_swin_t": 0.1}

# (r1 / n_heads, r2 = r3 / d_head); the first one is the ViT-study setting (4/12 heads, 16/64)
LEVELS = {
    "L1_h1/3_d1/4": (1 / 3, 1 / 4),
    "L2_h1/2_d1/2": (1 / 2, 1 / 2),
    "L3_h1_d1": (1.0, 1.0),
    "L4_h1_d2": (1.0, 2.0),
}
FROZEN_DROP_TOLERANCE = 0.03  # headline level: most compressed with <=3 pt oil-IoU loss


class AutocastEval(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(x).float()


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def make_loaders(args, device):
    common = dict(root=args.data_root, img_size=(args.size, args.size), num_classes=2, seed=42)
    train_set = SAR_Dataset(split="train", augment=True, **common)
    val_set = SAR_Dataset(split="validation", augment=False, **common)
    if args.subset:
        train_set = Subset(train_set, range(0, len(train_set), max(1, len(train_set) // args.subset)))
        val_set = Subset(val_set, range(0, len(val_set), max(1, len(val_set) // args.subset)))
    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                persistent_workers=args.workers > 0)
    gen = torch.Generator().manual_seed(42)
    return (DataLoader(train_set, shuffle=True, generator=gen, drop_last=True, **opts),
            DataLoader(val_set, shuffle=False, **opts))


def class_weights(train_loader, device, cache: Path):
    if cache.is_file():
        return torch.tensor(json.loads(cache.read_text()), device=device)
    w = _class_weights(train_loader, 2, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(w.tolist()))
    return w


def build_optimizer(model: SegWrapper, enc_lr, head_lr, tucker_lr=None, wd=0.01):
    groups = {"tucker": [], "enc": [], "head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if tucker_lr is not None and ".core." in name:
            groups["tucker"].append(p)
        elif name.startswith(model.encoder_prefixes):
            groups["enc"].append(p)
        else:
            groups["head"].append(p)
    lrs = {"tucker": tucker_lr, "enc": enc_lr, "head": head_lr}
    param_groups = [dict(params=ps, lr=lrs[k], base_lr=lrs[k]) for k, ps in groups.items() if ps]
    return torch.optim.AdamW(param_groups, weight_decay=wd)


def set_lr(opt, step, total, warmup):
    scale = (step + 1) / warmup if step < warmup else \
        0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * scale


def profile_model(model: SegWrapper, val_loader, device, size) -> dict:
    """Parameters, FLOPs, latency (bf16 + fp32) and inference memory."""
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    x1 = torch.randn(1, 3, size, size, device=device)
    from torch.utils.flop_counter import FlopCounterMode

    flops = 0.0
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x1)
        flops = fc.get_total_flops()
    except Exception as exc:  # flop counting is best-effort
        print("flop counter failed:", exc)

    out = {"params_m": n_params / 1e6, "gflops": flops / 1e9}
    x8 = torch.randn(8, 3, size, size, device=device)
    for tag, ctx in (("bf16", lambda: torch.autocast("cuda", dtype=torch.bfloat16)),
                     ("fp32", lambda: torch.autocast("cuda", enabled=False))):
        with torch.no_grad(), ctx():
            for _ in range(3):
                model(x8)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                model(x8)
            torch.cuda.synchronize()
        out[f"ms_per_img_{tag}"] = 1000 * (time.perf_counter() - t0) / (reps * 8)
        out[f"infer_peak_mem_mb_{tag}"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out


@torch.no_grad()
def evaluate(model, val_loader, device) -> dict:
    return evaluate_seg(AutocastEval(model), val_loader, 2, device)


def train_config(model: SegWrapper, name: str, args, loaders, weights, lrs, out_dir: Path,
                 ckpt_dir: Path, fh) -> dict:
    """Train `epochs` epochs; per-epoch CSV + best/last checkpoints. Returns timing/memory info."""
    device = torch.device("cuda")
    train_loader, val_loader = loaders
    model.to(device)
    model.set_class_weights(weights)
    enc_lr, head_lr, tucker_lr = lrs
    opt = build_optimizer(model, enc_lr, head_lr, tucker_lr)
    total_steps = args.epochs * len(train_loader)
    warmup = min(300, total_steps // 10)
    best_path, last_path = ckpt_dir / f"{name}_best.pt", ckpt_dir / f"{name}_last.pt"
    csv_path = out_dir / f"{name}_epochs.csv"
    rows, start_epoch, best_iou, train_time = [], 0, -1.0, 0.0
    if last_path.is_file():
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rows, start_epoch, best_iou, train_time = st["rows"], st["epoch"], st["best_iou"], st["train_time"]
        log(f"{name}: resumed after epoch {start_epoch}", fh)

    torch.cuda.reset_peak_memory_stats(device)
    clip = CLIP.get(args.model, 1.0)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            set_lr(opt, epoch * len(train_loader) + step, total_steps, warmup)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.training_loss(images, masks)
            if not torch.isfinite(loss):
                log(f"{name}: non-finite loss at epoch {epoch + 1} step {step}; skipping batch", fh)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item()
            n += 1
        epoch_train = time.perf_counter() - t0
        train_time += epoch_train
        m = evaluate(model, val_loader, device)
        row = dict(epoch=epoch + 1, loss=total / max(n, 1), f1=m["f1"], mean_f1=m["mean_f1"],
                   oil_spill_iou=m["oil_spill_iou"], mean_iou=m["mean_iou"],
                   accuracy=m["accuracy"], precision=m["precision"], recall=m["recall"],
                   train_time_s=epoch_train, eval_time_s=m["eval_time_s"],
                   confusion=json.dumps(m["confusion_matrix"]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log(f"{name} ep {epoch + 1}/{args.epochs} loss={row['loss']:.4f} f1={row['f1']:.4f} "
            f"iou={row['oil_spill_iou']:.4f} acc={row['accuracy']:.4f} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} ({epoch_train:.0f}s train, {m['eval_time_s']:.0f}s eval)", fh)
        if row["oil_spill_iou"] > best_iou:
            best_iou = row["oil_spill_iou"]
            torch.save(model.state_dict(), best_path)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rows=rows, epoch=epoch + 1,
                        best_iou=best_iou, train_time=train_time), last_path)
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    return dict(train_time_s=train_time, train_peak_mem_mb=peak, best_path=str(best_path),
                best_iou=best_iou, epochs=len(rows))


def fresh_model(key):
    m = build(key).cuda()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-root", default=str(ROOT / "Deep_SAR" / "versions" / "1"))
    ap.add_argument("--subset", type=int, default=0, help="limit each split to ~N images (smoke tests)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--tag", default="")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--headline-level", default="L3_h1_d1",
                    help="Tucker level fine-tuned in the unfrozen run ('' = most compressed level "
                         "within 3 pt of baseline oil-IoU)")
    ap.add_argument("--unfrozen-epochs", type=int, default=0, help="default: same as --epochs")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    key = args.model
    out_dir = Path(args.out) / (key + args.tag)
    ckpt_dir = CKPT_ROOT / (key + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "run_log.txt", "a", encoding="utf-8")
    log(f"=== {key}: epochs={args.epochs} batch={args.batch_size} size={args.size} ===", fh)

    loaders = make_loaders(args, device)
    weights = class_weights(loaders[0], device, CKPT_ROOT / f"class_weights_{args.size}_{args.subset}.json")
    enc_lr, head_lr = BASE_LRS[key]
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}

    # ---- 1. baseline -------------------------------------------------------
    model = fresh_model(key)
    summary["model"] = dict(key=key, pretty=model.pretty, hf_id=model.hf_id,
                            total_params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    if "baseline" not in summary:
        info = train_config(model, "baseline", args, loaders, weights,
                            (enc_lr, head_lr, None), out_dir, ckpt_dir, fh)
        model.load_state_dict(torch.load(info["best_path"], map_location=device))
        metrics = evaluate(model, loaders[1], device)
        summary["baseline"] = dict(**info, metrics=metrics, profile=profile_model(model, loaders[1], device, args.size))
        summary_path.write_text(json.dumps(summary, indent=1))
    else:
        model.load_state_dict(torch.load(summary["baseline"]["best_path"], map_location=device))
    base_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    log(f"baseline best IoU={summary['baseline']['metrics']['oil_spill_iou']:.4f}", fh)
    del model
    torch.cuda.empty_cache()

    # ---- 2. Tucker frozen: compress + inference, rank-level sweep ---------
    if "sweep" not in summary:
        sweep = {}
        for level_name in args.levels.split(","):
            m = fresh_model(key)
            m.load_state_dict(base_state)
            t0 = time.perf_counter()
            stats = tuckerize(m, m.tucker_families, key, level=LEVELS[level_name])
            compress_time = time.perf_counter() - t0
            metrics = evaluate(m, loaders[1], device)
            prof = profile_model(m, loaders[1], device, args.size)
            sweep[level_name] = dict(
                metrics=metrics, profile=prof, compress_time_s=compress_time,
                dense_attn_params=stats["dense_attn_params"], tucker_attn_params=stats["tucker_attn_params"],
                compression=stats["compression"], n_layers=stats["n_layers"],
                example_ranks=[(l["name"], l["ranks"]) for l in stats["layers"][:3]])
            log(f"frozen {level_name}: attn {stats['compression']:.1f}x smaller, "
                f"IoU={metrics['oil_spill_iou']:.4f} F1={metrics['f1']:.4f}", fh)
            del m
            torch.cuda.empty_cache()
        summary["sweep"] = sweep
        summary_path.write_text(json.dumps(summary, indent=1))
    sweep = summary["sweep"]

    base_iou = summary["baseline"]["metrics"]["oil_spill_iou"]
    if args.headline_level:
        level_name = args.headline_level
    else:
        ok = [n for n, r in sweep.items() if base_iou - r["metrics"]["oil_spill_iou"] <= FROZEN_DROP_TOLERANCE]
        level_name = max(ok, key=lambda n: sweep[n]["compression"]) if ok else max(
            sweep, key=lambda n: sweep[n]["metrics"]["oil_spill_iou"])
    summary["headline_level"] = level_name
    summary["tucker_frozen"] = sweep[level_name]
    summary_path.write_text(json.dumps(summary, indent=1))
    log(f"headline Tucker level: {level_name} ({sweep[level_name]['compression']:.1f}x)", fh)

    # ---- 3. Tucker unfrozen: fine-tune the compressed model ---------------
    if "tucker_unfrozen" not in summary:
        m = fresh_model(key)
        m.load_state_dict(base_state)
        tuckerize(m, m.tucker_families, key, level=LEVELS[level_name])
        sub = copy.copy(args)
        sub.epochs = args.unfrozen_epochs or args.epochs
        info = train_config(m, "tucker_unfrozen", sub, loaders, weights,
                            (enc_lr / 10, head_lr / 10, enc_lr), out_dir, ckpt_dir, fh)
        m.load_state_dict(torch.load(info["best_path"], map_location=device))
        metrics = evaluate(m, loaders[1], device)
        summary["tucker_unfrozen"] = dict(**info, metrics=metrics,
                                          profile=profile_model(m, loaders[1], device, args.size),
                                          level=level_name)
        summary_path.write_text(json.dumps(summary, indent=1))
    log(f"unfrozen best IoU={summary['tucker_unfrozen']['metrics']['oil_spill_iou']:.4f}", fh)

    plot_comparison(out_dir, summary, args.epochs, args.batch_size)
    log("done", fh)


if __name__ == "__main__":
    main()
