# SAR Oil-Spill Segmentation

This research project trains a transformer-based model to segment marine oil
spills in grayscale SAR images. The segmentation pipeline uses a selectable
pretrained ViT encoder, a high-resolution detail branch, and a lightweight
pixel decoder. ViT-L/14 is the default; Tucker attention is available as an
optional experiment and the baseline uses standard attention.

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

The default input size is 256 × 256 pixels. The model pads the bottom and
right edges as needed so image dimensions work with the selected ViT's patch
size, then returns a mask at the original input resolution. Images are
converted to RGB for the pretrained encoder and normalized with ImageNet
statistics. Training augmentation applies paired horizontal/vertical flips,
mild rotations, and light brightness/contrast changes.

Run the pipeline from the project root with the included dataset:

```powershell
python -m starter.starter_seg
```

Choose an encoder with `--model`:

```powershell
python -m starter.starter_seg --model vit-l14  # default: ViT-L/14 (DINOv2)
python -m starter.starter_seg --model vit-b16  # ViT-B/16
python -m starter.starter_seg --model vit-s16  # ViT-S/16
```

The presets use `facebook/dinov2-large`, `google/vit-base-patch16-224-in21k`,
and `WinKawaks/vit-small-patch16-224`, respectively. To use a different
Hugging Face model or a local model directory, pass `--model-name PATH_OR_ID`;
it overrides the preset.

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

## Tucker rank study

`rank_sweep.py` applies the repository's HOSVD-initialized Tucker attention
conversion to the SAR segmentation model. It compares an MHA baseline, two
reference Tucker ranks, a descending tied-rank sweep (`r2 = r3`), and a small
`r1` sweep. Each result is trained and evaluated with the same dataset split
and segmentation pipeline as `starter/starter_seg.py`.

Run a small wiring check before a full experiment:

```powershell
python rank_sweep.py --smoke
```

Run the full sweep and average over three training seeds:

```powershell
python rank_sweep.py --seeds 3 --epochs 80
```

Results are written to `results_sar_rank_sweep.csv`. The CSV includes pixel
accuracy, foreground oil-spill IoU, non-background mean IoU, attention
parameter count, and compression ratio, with means and standard deviations
across seeds. The report identifies the smallest Tucker configuration within
both baseline tolerances (`--max-accuracy-drop` and `--max-iou-drop`, each
defaulting to 0.01) and reports the first tied-rank oil-spill-IoU knee
(`--knee-thresh`, also defaulting to 0.01). These thresholds are absolute
metric fractions (0.01 is one percentage point). Use `--subset N` to limit
each data split during exploratory runs; `--output` selects a different CSV
path. Tucker factor ranks are bounded by the loaded ViT's dimensions. The rank sweep
defaults to the same ViT-L/14 encoder as the starter; use `--model-name` to
override it.

Generate a Markdown analysis report from the sweep CSV and run log:

```powershell
python analyze_output.py
```

By default this reads `results_sar_rank_sweep.csv` and
`outputs/out_ranksweep2.txt`, then writes `analysis_report.md`. Use `--csv`,
`--log`, and `--output` to select different input or report paths.

## Project files

```text
data/sar_dataset.py          SAR image/mask pairing, splitting, and preprocessing
models/vit_tucker_seg.py     ViT encoder, detail branch, and segmentation decoder
models/vit_surgery.py        Optional Tucker attention conversion
train/train_seg.py           Losses, metrics, staged fine-tuning, and checkpoints
starter/starter_seg.py       SAR training and validation evaluation entry point
rank_sweep.py                Tucker-rank compression and validation-metric study
```

The papers included in the project folder are background references, not dataset
archives. Model weights are downloaded from Hugging Face on first use.
