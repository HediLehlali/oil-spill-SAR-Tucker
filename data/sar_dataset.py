from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLIT_NAMES = {
    "train": ("train", "training"),
    "validation": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def _mask_array(image: Image.Image, num_classes: int) -> np.ndarray:
    if image.mode in ("P", "L", "1", "I", "I;16", "I;16B", "I;16L"):
        mask = np.asarray(image)
    elif image.mode in ("RGB", "RGBA"):
        channels = np.asarray(image.convert("RGB"))
        if np.array_equal(channels[..., 0], channels[..., 1]) and np.array_equal(
            channels[..., 0], channels[..., 2]
        ):
            mask = channels[..., 0]
        else:
            raise ValueError(
                f"Mask image mode '{image.mode}' is not a single-channel class-ID "
                "mask. Convert it to integer class IDs using the source dataset's "
                "label map."
            )
    else:
        raise ValueError(
            f"Mask image mode '{image.mode}' is not a single-channel class-ID mask. "
            "Convert it to integer class IDs using the source dataset's label map."
        )

    if num_classes == 2 and np.any(mask > 1):
        return (mask >= 128).astype(np.uint8)
    return mask


def _find_split_directories(
    root: Path, split: str, required: bool = True
) -> tuple[Path, Path] | None:
    directory_pairs = (
        ("images", "masks"),
        ("images_1D", "labels_1D"),
        ("images_1D", "masks"),
        ("images", "labels_1D"),
        ("images/images", "masks/masks"),
    )
    split_names = SPLIT_NAMES[split]
    for image_dir_name, mask_dir_name in directory_pairs:
        for split_name in split_names:
            image_dir = root / image_dir_name / split_name
            mask_dir = root / mask_dir_name / split_name
            if image_dir.is_dir() and mask_dir.is_dir():
                return image_dir, mask_dir

    if required:
        raise FileNotFoundError(
            f"Could not find paired image and mask directories for '{split}' "
            f"under {root}. Checked the supported image/mask layouts and split "
            f"names: {', '.join(split_names)}."
        )
    return None


def _paired_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    mask_by_stem = {
        path.stem.casefold(): path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    }
    image_paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not image_paths:
        raise ValueError(f"No supported SAR images found in {image_dir}")

    pairs = []
    missing_masks = []
    for image_path in image_paths:
        mask_path = mask_by_stem.get(image_path.stem.casefold())
        if mask_path is None:
            missing_masks.append(image_path.name)
        else:
            pairs.append((image_path, mask_path))
    if missing_masks:
        sample = ", ".join(missing_masks[:5])
        raise FileNotFoundError(
            f"{len(missing_masks)} image(s) in {image_dir} have no matching "
            f"mask in {mask_dir}; examples: {sample}"
        )
    return pairs


def _train_validation_indices(
    pairs: list[tuple[Path, Path]],
    validation_fraction: float,
    seed: int,
    num_classes: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    groups: dict[bool, list[int]] = {False: [], True: []}
    for index, (_, mask_path) in enumerate(pairs):
        with Image.open(mask_path) as mask_image:
            has_oil = bool(np.any(_mask_array(mask_image, num_classes) == 1))
        groups[has_oil].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in groups.values():
        rng.shuffle(indices)
        validation_count = round(len(indices) * validation_fraction)
        if len(indices) > 1:
            validation_count = max(1, min(len(indices) - 1, validation_count))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    if not train_indices or not validation_indices:
        raise ValueError(
            "Need at least two training samples to create a train/validation split"
        )
    return train_indices, validation_indices


class SAR_Dataset(Dataset):
    """Paired SAR images and class-ID masks for the documented oil-spill dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: tuple[int, int] = (256, 256),
        num_classes: int = 2,
        augment: bool = False,
        validation_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = "validation" if split in ("val", "validation") else split
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.num_classes = num_classes
        self.augment = augment and self.split == "train"

        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        if self.split == "test":
            image_dir, mask_dir = _find_split_directories(self.root, "test")
            self.files = _paired_files(image_dir, mask_dir)
        else:
            image_dir, mask_dir = _find_split_directories(self.root, "train")
            train_pairs = _paired_files(image_dir, mask_dir)
            validation_directories = _find_split_directories(
                self.root, "validation", required=False
            )
            if validation_directories is not None:
                if self.split == "train":
                    self.files = train_pairs
                else:
                    self.files = _paired_files(*validation_directories)
                return

            train_indices, validation_indices = _train_validation_indices(
                train_pairs, validation_fraction, seed, num_classes
            )
            indices = train_indices if self.split == "train" else validation_indices
            self.files = [train_pairs[index] for index in indices]

    def __len__(self) -> int:
        return len(self.files)

    def _augment_pair(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        angle = random.uniform(-5.0, 5.0)
        image = image.rotate(
            angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)
        )
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.files[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(
                self.img_size, Image.Resampling.BILINEAR
            )
        with Image.open(mask_path) as source:
            mask_array = _mask_array(source, self.num_classes)

        invalid = (mask_array < 0) | (mask_array >= self.num_classes)
        if np.any(invalid):
            labels = np.unique(mask_array[invalid]).tolist()
            raise ValueError(
                f"Mask {mask_path} contains class IDs {labels}, outside "
                f"the configured range 0..{self.num_classes - 1}. "
                "Check the dataset label mapping instead of silently remapping it."
            )

        if self.augment:
            mask = Image.fromarray(mask_array).resize(
                self.img_size, Image.Resampling.NEAREST
            )
            image, mask = self._augment_pair(image, mask)
            mask_array = np.asarray(mask)
        else:
            mask = Image.fromarray(mask_array).resize(
                self.img_size, Image.Resampling.NEAREST
            )
            mask_array = np.asarray(mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        mask_array = np.asarray(mask_array, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array.copy())
        return image_tensor, mask_tensor
