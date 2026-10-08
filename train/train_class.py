# train_class.py
"""
Entraînement classification (CIFAR-10 / CIFAR-100)
avec ViT + Tucker Attention.

Utilise :
    - vit_surgery.tuckerize_vit
    - tucker_attention.TuckerViTAttention
    - train.py (boucle d'entraînement de Rafik)
"""

from __future__ import annotations
import argparse
import torch

from models.vit_surgery import load_vit, tuckerize_vit
from train import PRESETS, TrainCfg, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True,
                    choices=["vit-b16-cifar10", "vit-b16-cifar100"])
    ap.add_argument("--r1", type=int, default=8)
    ap.add_argument("--r2", type=int, default=64)
    ap.add_argument("--r3", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    preset = PRESETS[args.preset]

    print(f"[device] {device}")
    print(f"[preset] {args.preset}")

    ranks = (args.r1, args.r2, args.r3)
    print(f"[tucker] ranks = {ranks}")

    results = []
    for seed in range(args.seeds):
        print(f"\n=== Seed {seed} ===")

        cfg = TrainCfg(
            model_name=preset["model_name"],
            dataset=preset["dataset"],
            num_labels=preset["num_labels"],
            batch_size=preset["batch_size"],
            lr=preset["lr"],
            weight_decay=preset["weight_decay"],
            epochs=preset["epochs"],
            seed=seed,
            device=device,
        )

        # Charger ViT HF
        model = load_vit(cfg.model_name, cfg.num_labels, device)

        # Tuckeriser
        model, stats = tuckerize_vit(model, ranks, verbose=True)

        # Entraîner
        res = train_one(model, cfg)
        results.append(res)

    print("\n=== Résultats ===")
    for i, r in enumerate(results):
        print(f"Seed {i}: top1={r['top1']:.2f}  top5={r['top5']:.2f}")


if __name__ == "__main__":
    main()
