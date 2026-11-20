"""Per-epoch comparison curves in the style of outputs/ViTL_Runs/batch8_comparison_curves.png."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["f1", "oil_spill_iou", "accuracy", "precision", "recall", "loss"]
COLORS = {"baseline": "#1f77b4", "frozen": "#ff7f0e", "unfrozen": "#2ca02c"}


def plot_comparison(out_dir, summary: dict, epochs: int, batch_size: int, title_prefix: str = "") -> Path:
    out_dir = Path(out_dir)
    base = pd.read_csv(out_dir / "baseline_epochs.csv")
    unf = pd.read_csv(out_dir / "tucker_unfrozen_epochs.csv")
    frozen = summary["tucker_frozen"]["metrics"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))
    for ax, metric in zip(axes.ravel(), METRICS):
        ax.plot(base["epoch"], base[metric], "o-", ms=5, alpha=0.85, color=COLORS["baseline"], label="Baseline")
        if metric != "loss":
            ax.axhline(frozen[metric], ls="--", lw=1.8, alpha=0.9, color=COLORS["frozen"],
                       label="Tucker (frozen, no training)")
            x = [0] + list(unf["epoch"])
            y = [frozen[metric]] + list(unf[metric])
        else:
            x, y = list(unf["epoch"]), list(unf[metric])
        ax.plot(x, y, "o-", ms=5, alpha=0.85, color=COLORS["unfrozen"], label="Tucker (unfrozen)")
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        last = int(max(base["epoch"].max(), unf["epoch"].max()))
        ax.set_xticks(range(1, last + 1, 2))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    pretty = summary["model"]["pretty"]
    fig.suptitle(f"{title_prefix}{pretty}: Per-Epoch Metrics Comparison "
                 f"(Batch Size {batch_size}, {epochs} Epochs)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "comparison_curves.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
