"""
Run a Tucker-rank sweep for the SAR oil-spill segmentation model.

For each seed, the sweep compares the uncompressed ViT attention baseline with
HOSVD-initialized Tucker attention variants. It reports validation segmentation
metrics and attention-parameter savings, then identifies both the first
consecutive IoU drop ("knee") and the smallest model within the configured
accuracy and oil-spill-IoU tolerances of the baseline.

Quick pipeline check:
    python rank_sweep.py --smoke

Full sweep:
    python rank_sweep.py --seeds 3 --epochs 80
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.sar_dataset import SAR_Dataset
from models.vit_surgery import attention_param_report
from models.vit_tucker_seg import ViT_Tucker_Seg
from train.train_seg import train_seg


DEFAULT_TIED_R = [64, 48, 32, 24, 16, 12]
DEFAULT_R1_SWEEP = [4, 2, 1]


def build_rank_grid(
    n_heads: int, d_model: int, r1_fixed: int
) -> list[tuple[str, tuple[int, int, int]]]:
    """Build tied r2/r3 ranks and a small r1 sweep, clipped to model dimensions."""
    max_factor_rank = min(64, d_model)
    r1_fixed = min(r1_fixed, n_heads)
    grid: list[tuple[str, tuple[int, int, int]]] = []
    for rank in DEFAULT_TIED_R:
        rank = min(rank, d_model)
        grid.append((f"tied_r{rank}", (r1_fixed, rank, rank)))
    for r1 in DEFAULT_R1_SWEEP:
        if r1 <= n_heads:
            grid.append((f"r1_{r1}", (r1, max_factor_rank, max_factor_rank)))

    seen: set[tuple[int, int, int]] = set()
    unique: list[tuple[str, tuple[int, int, int]]] = []
    for tag, ranks in grid:
        if ranks not in seen:
            seen.add(ranks)
            unique.append((tag, ranks))
    return unique


def detect_knee(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    """Return the last safe tied-rank row before an adjacent oil-IoU drop."""
    tied = [row for row in rows if row["tag"].startswith("tied_")]
    tied.sort(key=lambda row: row["attn_params"], reverse=True)
    for larger, smaller in zip(tied, tied[1:]):
        drop = larger["oil_spill_iou_mean"] - smaller["oil_spill_iou_mean"]
        if drop > threshold:
            return {
                "last_safe": larger,
                "falls_at": smaller,
                "drop": drop,
            }
    return None


def find_smallest_acceptable(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    max_accuracy_drop: float,
    max_iou_drop: float,
) -> dict[str, Any] | None:
    """Choose the smallest Tucker model within both baseline metric tolerances."""
    eligible = [
        row
        for row in rows
        if row["tag"] != "MHA"
        and row["attn_params"] < baseline["attn_params"]
        and row["accuracy_mean"] >= baseline["accuracy_mean"] - max_accuracy_drop
        and row["oil_spill_iou_mean"]
        >= baseline["oil_spill_iou_mean"] - max_iou_drop
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row["attn_params"],
            -row["oil_spill_iou_mean"],
            -row["accuracy_mean"],
        ),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.pstdev(values) if len(values) > 1 else 0.0,
    )


def run(args: argparse.Namespace) -> None:
    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    if args.r1_fixed < 1:
        raise ValueError("--r1-fixed must be at least 1")
    if args.subset < 0:
        raise ValueError("--subset must be zero or a positive number")
    if args.knee_thresh < 0 or args.max_accuracy_drop < 0 or args.max_iou_drop < 0:
        raise ValueError("Metric drop tolerances must be non-negative")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seeds = list(range(args.seeds))
    epochs = args.epochs
    subset_size = args.subset
    if args.smoke:
        epochs = 1
        seeds = [0]
        subset_size = subset_size or 16

    print(f"[device] training on: {device}")
    train_set = SAR_Dataset(
        root=args.data_root,
        split="train",
        img_size=(args.image_width, args.image_height),
        num_classes=args.num_classes,
        augment=True,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    val_set = SAR_Dataset(
        root=args.data_root,
        split="validation",
        img_size=(args.image_width, args.image_height),
        num_classes=args.num_classes,
        augment=False,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    if subset_size:
        train_set = Subset(train_set, range(min(subset_size, len(train_set))))
        val_set = Subset(val_set, range(min(subset_size, len(val_set))))
    if not len(train_set) or not len(val_set):
        raise ValueError("The rank sweep requires non-empty train and validation sets")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    print(
        f"[data] {len(train_set)} train / {len(val_set)} validation samples; "
        f"{args.num_classes} classes"
    )

    probe = ViT_Tucker_Seg(
        num_classes=args.num_classes,
        model_name=args.model_name,
    )
    n_heads = probe.vit.config.num_attention_heads
    d_model = probe.vit.config.hidden_size
    mha_attention_params = sum(
        parameter.numel()
        for layer in probe.vit.vit.encoder.layer
        for parameter in layer.attention.parameters()
    )
    del probe
    if args.r1_fixed > n_heads:
        raise ValueError(
            f"--r1-fixed ({args.r1_fixed}) cannot exceed the model's "
            f"{n_heads} attention heads"
        )

    def train_config(tag: str, ranks: tuple[int, int, int] | None) -> dict[str, Any]:
        per_seed: list[dict[str, float]] = []
        attention_params: int | None = None
        for seed in seeds:
            _seed_everything(seed)
            model = ViT_Tucker_Seg(
                num_classes=args.num_classes,
                ranks=ranks or (min(4, n_heads), min(64, d_model), min(64, d_model)),
                model_name=args.model_name,
                use_tucker=ranks is not None,
            )
            if ranks is None:
                attention_params = sum(
                    parameter.numel()
                    for layer in model.vit.vit.encoder.layer
                    for parameter in layer.attention.parameters()
                )
            else:
                attention_params = attention_param_report(model.vit)

            with tempfile.TemporaryDirectory(prefix="sar-rank-sweep-") as temp_dir:
                checkpoint = Path(temp_dir) / "best.pt"
                try:
                    metrics = train_seg(
                        model,
                        train_loader,
                        val_loader,
                        num_classes=args.num_classes,
                        device=device,
                        epochs=epochs,
                        lr=args.learning_rate,
                        encoder_lr=args.encoder_learning_rate,
                        unfreeze_last=args.unfreeze_last,
                        stage1_epochs=args.stage1_epochs,
                        patience=args.patience,
                        checkpoint_path=checkpoint,
                    )
                    per_seed.append(metrics)
                finally:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        assert attention_params is not None
        row: dict[str, Any] = {
            "tag": tag,
            "r1": ranks[0] if ranks else "",
            "r2": ranks[1] if ranks else "",
            "r3": ranks[2] if ranks else "",
            "n_seeds": len(per_seed),
            "attn_params": attention_params,
            "attention_reduction_pct": (
                100.0 * (1.0 - attention_params / mha_attention_params)
                if ranks is not None
                else 0.0
            ),
            "compression_ratio": (
                mha_attention_params / attention_params
                if ranks is not None
                else 1.0
            ),
        }
        for metric_name in (
            "accuracy",
            "oil_spill_iou",
            "mean_iou_no_background",
            "precision",
            "recall",
        ):
            mean, std = _mean_std([result[metric_name] for result in per_seed])
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std

        print(
            f"[{tag}] ranks={ranks or 'MHA'} params={attention_params:,} "
            f"reduction={row['attention_reduction_pct']:.1f}% "
            f"accuracy={row['accuracy_mean']:.4f} "
            f"oil_iou={row['oil_spill_iou_mean']:.4f} "
            f"mean_iou_no_bg={row['mean_iou_no_background_mean']:.4f}"
        )
        return row

    rows: list[dict[str, Any]] = []

    print("\n=== PHASE A: baseline and reference ranks ===")
    baseline = train_config("MHA", None)
    rows.append(baseline)
    reference_ranks = [
        (min(4, n_heads), min(64, d_model), min(64, d_model)),
        (min(4, n_heads), min(16, d_model), min(16, d_model)),
    ]
    completed = {tuple(row[key] for key in ("r1", "r2", "r3")) for row in rows}
    for ranks in reference_ranks:
        if ranks not in completed:
            rows.append(train_config(f"ref_{ranks[0]}_{ranks[1]}_{ranks[2]}", ranks))
            completed.add(ranks)

    print("\n=== PHASE B: rank sweep ===")
    if args.smoke:
        rank_grid = [
            (f"tied_r{min(8, d_model)}", (min(args.r1_fixed, n_heads), min(8, d_model), min(8, d_model)))
        ]
    else:
        rank_grid = build_rank_grid(n_heads, d_model, args.r1_fixed)
    for tag, ranks in rank_grid:
        if ranks not in completed:
            rows.append(train_config(tag, ranks))
            completed.add(ranks)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with out.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[wrote] {out}")

    print("\n=== COMPRESSION REPORT ===")
    print(
        f"MHA baseline: accuracy={baseline['accuracy_mean']:.4f}, "
        f"oil_spill_iou={baseline['oil_spill_iou_mean']:.4f}, "
        f"attention_params={baseline['attn_params']:,}"
    )
    acceptable = find_smallest_acceptable(
        rows,
        baseline,
        args.max_accuracy_drop,
        args.max_iou_drop,
    )
    if acceptable:
        print(
            f"Smallest model within baseline tolerances "
            f"(accuracy drop <= {args.max_accuracy_drop:.4f}, "
            f"oil-spill IoU drop <= {args.max_iou_drop:.4f}): "
            f"{acceptable['tag']} ranks=({acceptable['r1']},"
            f"{acceptable['r2']},{acceptable['r3']}), "
            f"{acceptable['attn_params']:,} attention params "
            f"({acceptable['attention_reduction_pct']:.1f}% reduction, "
            f"{acceptable['compression_ratio']:.2f}x smaller)."
        )
    else:
        print("No Tucker configuration met both baseline metric tolerances.")

    knee = detect_knee(rows, args.knee_thresh)
    if knee:
        last_safe, falls_at = knee["last_safe"], knee["falls_at"]
        print(
            f"First tied-rank IoU drop > {args.knee_thresh:.4f}: "
            f"{last_safe['tag']} ({last_safe['oil_spill_iou_mean']:.4f}) -> "
            f"{falls_at['tag']} ({falls_at['oil_spill_iou_mean']:.4f}), "
            f"drop={knee['drop']:.4f}."
        )
    else:
        print(
            f"No adjacent tied-rank oil-spill IoU drop exceeded "
            f"{args.knee_thresh:.4f}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Tucker attention ranks for SAR oil-spill segmentation."
    )
    parser.add_argument("--data-root", default="Deep_SAR/versions/1")
    parser.add_argument("--model-name", default="facebook/dinov2-large")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--unfreeze-last", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--r1-fixed", type=int, default=4)
    parser.add_argument(
        "--knee-thresh",
        type=float,
        default=0.01,
        help="adjacent oil-spill IoU drop that identifies a knee (fraction)",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.01,
        help="maximum absolute pixel-accuracy drop from MHA (fraction)",
    )
    parser.add_argument(
        "--max-iou-drop",
        type=float,
        default=0.01,
        help="maximum absolute oil-spill IoU drop from MHA (fraction)",
    )
    parser.add_argument("--subset", type=int, default=0, help="limit samples per split")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results_sar_rank_sweep.csv")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one epoch, one seed, and a tiny baseline/reference/rank pipeline",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
