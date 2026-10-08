from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    class_losses = []
    for class_id in range(logits.shape[1]):
        prediction = probabilities[:, class_id]
        truth = (target == class_id).to(dtype=prediction.dtype)
        intersection = (prediction * truth).sum()
        denominator = prediction.sum() + truth.sum()
        class_losses.append(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))
    return torch.stack(class_losses[1:]).mean()


def _class_weights(
    train_loader: DataLoader, num_classes: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, masks in tqdm(train_loader, desc="Counting train pixels", leave=False):
        counts += torch.bincount(masks.reshape(-1), minlength=num_classes)
    if counts.sum() == 0:
        raise ValueError("The training set contains no labeled pixels")

    frequencies = counts / counts.sum()
    weights = torch.sqrt(frequencies[0] / frequencies.clamp_min(1e-8))
    weights = weights.clamp(min=1.0, max=5.0)
    weights[0] = 1.0
    return weights.float().to(device)


@torch.no_grad()
def evaluate_seg(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    device: torch.device | str,
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )
    for images, masks in data_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        confusion += torch.bincount(
            masks.reshape(-1) * num_classes + predictions.reshape(-1),
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    true_positive = confusion.diag().float()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
    iou = true_positive / union.clamp_min(1)
    total_pixels = confusion.sum().float()
    spill_true_positive = confusion[1, 1].float()
    spill_false_positive = confusion[:, 1].sum().float() - spill_true_positive
    spill_false_negative = confusion[1, :].sum().float() - spill_true_positive
    present = union > 0
    non_background = present.clone()
    non_background[0] = False
    result = {
        "accuracy": (
            true_positive.sum() / total_pixels.clamp_min(1)
        ).item(),
        "precision": (
            spill_true_positive
            / (spill_true_positive + spill_false_positive).clamp_min(1)
        ).item(),
        "recall": (
            spill_true_positive
            / (spill_true_positive + spill_false_negative).clamp_min(1)
        ).item(),
    }
    result.update(
        {
            f"class_{class_id}_iou": iou[class_id].item()
            for class_id in range(num_classes)
        }
    )
    result["oil_spill_iou"] = iou[1].item()
    result["mean_iou"] = iou[present].mean().item() if present.any() else 0.0
    result["mean_iou_no_background"] = (
        iou[non_background].mean().item() if non_background.any() else 0.0
    )
    return result


def _set_trainable_parameters(model: nn.Module, unfreeze_last: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.detail_encoder,
        model.transformer_projection,
        model.fusion,
        model.decoder,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True

    if unfreeze_last:
        layers = model.vit.vit.encoder.layer
        if not 0 < unfreeze_last <= len(layers):
            raise ValueError(
                f"unfreeze_last must be between 0 and {len(layers)}"
            )
        for layer in layers[-unfreeze_last:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def _make_optimizer(
    model: nn.Module, head_lr: float, encoder_lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("vit.vit.")
    ]
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("vit.vit.")
    ]
    parameter_groups = [{"params": head_parameters, "lr": head_lr}]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": encoder_lr})
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def train_seg(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device | str = "cuda",
    epochs: int = 80,
    lr: float = 1e-3,
    encoder_lr: float = 1e-5,
    weight_decay: float = 0.01,
    unfreeze_last: int = 0,
    stage1_epochs: int = 10,
    patience: int = 12,
    checkpoint_path: str | Path = "checkpoints/sar_vit_best.pt",
) -> dict[str, float]:
    device = torch.device(device)
    model.to(device)
    _set_trainable_parameters(model, 0)

    class_weights = _class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_oil_iou = -1.0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    optimizer: torch.optim.Optimizer | None = None

    for epoch in range(epochs):
        fine_tuning = unfreeze_last > 0 and epoch >= stage1_epochs
        current_unfreeze = unfreeze_last if fine_tuning else 0
        _set_trainable_parameters(model, current_unfreeze)
        if optimizer is None or (fine_tuning and epoch == stage1_epochs):
            optimizer = _make_optimizer(
                model,
                head_lr=lr,
                encoder_lr=encoder_lr,
                weight_decay=weight_decay,
            )

        model.train()
        total_loss = 0.0
        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, masks) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        metrics = evaluate_seg(model, val_loader, num_classes, device)
        average_loss = total_loss / max(len(train_loader), 1)
        stage = "fine-tuning" if fine_tuning else "decoder"
        print(
            f"Epoch {epoch + 1}/{epochs} [{stage}] "
            f"loss={average_loss:.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"oil_spill_iou={metrics['oil_spill_iou']:.4f} "
            f"mean_iou_no_background={metrics['mean_iou_no_background']:.4f}"
        )

        if metrics["oil_spill_iou"] > best_oil_iou:
            best_oil_iou = metrics["oil_spill_iou"]
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training finished without producing a best checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return best_metrics
