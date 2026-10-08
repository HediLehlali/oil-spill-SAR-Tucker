from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from data.sar_dataset import SAR_Dataset
from models.vit_tucker_seg import ViT_Tucker_Seg
from train.train_seg import train_seg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the SAR oil-spill segmentation model."
    )
    parser.add_argument("--data-root", default="Deep_SAR/versions/1")
    parser.add_argument("--output", default="checkpoints/sar_vit_best.pt")
    parser.add_argument(
        "--model-name", default="WinKawaks/vit-small-patch16-224"
    )
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tucker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    image_size = (args.image_width, args.image_height)
    common = {
        "root": args.data_root,
        "img_size": image_size,
        "num_classes": args.num_classes,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
    }
    train_set = SAR_Dataset(split="train", augment=True, **common)
    val_set = SAR_Dataset(split="validation", augment=False, **common)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    print(
        f"Device: {device}; samples: {len(train_set)} train, "
        f"{len(val_set)} validation"
    )
    model = ViT_Tucker_Seg(
        num_classes=args.num_classes,
        model_name=args.model_name,
        use_tucker=args.tucker,
    )
    best_metrics = train_seg(
        model,
        train_loader,
        val_loader,
        num_classes=args.num_classes,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        encoder_lr=args.encoder_learning_rate,
        unfreeze_last=args.unfreeze_last,
        stage1_epochs=args.stage1_epochs,
        patience=args.patience,
        checkpoint_path=args.output,
    )

    print("Best validation metrics:")
    for name, value in best_metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
