# SAR Oil-Spill Segmentation

This research project trains a transformer-based model to segment marine oil
spills in grayscale SAR images. The segmentation pipeline uses a pretrained
Hugging Face ViT-S/16 encoder, a high-resolution detail branch, and a
lightweight pixel decoder. Tucker attention is available as an optional
experiment; the baseline is used by default.

## SAR data

The repository includes the public
[Deep-SAR Oil Spill Segmentation (Refined) dataset](https://www.kaggle.com/datasets/bakhtiyar2222/deep-sar-oil-spill-segmentation-refined)
under `Deep_SAR/versions/1`. Its paired images and masks are arranged as:

```text
Deep_SAR/versions/1/
├── images/images/{train,val}/
└── masks/masks/{train,val}/
```

The masks are binary: black is background (class 0) and white is foreground
(class 1). The loader converts grayscale and RGB binary masks, including
anti-aliased grayscale values, to class IDs. This dataset has train and
validation splits but no independent test split, so the reported metrics are
validation metrics. It is a two-class dataset and is not the five-class
1,002/110 corpus described in the project notes; do not apply that corpus's
label mapping to this dataset.

If you need to fetch a fresh copy, install the optional `kagglehub` package
(`python -m pip install kagglehub`) and run `dlSAR_Dataset.py`. It downloads the
dataset to the KaggleHub cache and prints the actual download path. The
training script uses the copy in the repository by default.

## Install

Use Python 3.10 or newer and install the project dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA setup
from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Train and evaluate

The default input size is 256 × 256 pixels, matching the dataset and divisible
by the ViT patch size of 16. The input is converted to RGB for the pretrained
encoder and normalized with ImageNet statistics. Training augmentation applies
paired horizontal/vertical flips, mild rotations, and light
brightness/contrast changes.

Run the pipeline from the project root with the included dataset:

```powershell
python -m starter.starter_seg
```

The script uses the dataset's existing train and validation splits. For a
different dataset location, set `--data-root`. It trains the decoder/detail
branch first, uses class-weighted cross-entropy plus Dice loss, selects the
checkpoint by validation foreground IoU, and reports validation accuracy,
precision, and recall. Precision and recall are calculated for the oil-spill
foreground class; accuracy is calculated over all pixels. IoU metrics are also
reported and the oil-spill IoU remains the checkpoint-selection criterion.

The default batch size is 2 to limit memory use at full image resolution. If
the GPU has enough memory, increase it with `--batch-size`. The encoder remains
frozen by default; if validation results indicate that its pretrained
features need adaptation, fine-tune the last four transformer blocks with:

```powershell
python -m starter.starter_seg --unfreeze-last 4
```

Use `--tucker` to enable Tucker attention in the ViT encoder. This option is
experimental and may increase model initialization time.

Useful options:

```text
--epochs 80
--batch-size 2
--validation-fraction 0.15  # used when the dataset has no validation split
--image-width 256
--image-height 256
--output checkpoints/sar_vit_best.pt
--num-workers 0
```

## Project files

```text
data/sar_dataset.py          SAR image/mask pairing, splitting, and preprocessing
models/vit_tucker_seg.py     ViT encoder, detail branch, and segmentation decoder
models/vit_surgery.py        Optional Tucker attention conversion
train/train_seg.py           Losses, metrics, staged fine-tuning, and checkpoints
starter/starter_seg.py       SAR training and validation evaluation entry point
```

The papers included in the project folder are background references, not dataset
archives. Model weights are downloaded from Hugging Face on first use.
