# SAR Tucker Rank Sweep Analysis

- Results CSV: `C:\Users\hedil\Desktop\ELAB4SPACE\Oil Spills\results_sar_rank_sweep.csv`
- Training log: `C:\Users\hedil\Desktop\ELAB4SPACE\Oil Spills\outputs\out_ranksweep2.txt`
- Configurations: 9
- Seeds per configuration: 1
- Training device: cuda
- Data split: 6,455 train / 1,615 validation samples; 2 classes

## Key findings

- MHA baseline: accuracy 92.56%, oil-spill IoU 77.07%, 100,761,600 attention parameters.
- Highest oil-spill IoU: **ref_4_64_64** (77.16%; 92.9% attention-parameter reduction).
- Highest pixel accuracy: **ref_4_64_64** (92.79%).
- Baseline tolerances used: accuracy drop <= 0.0100, oil-spill IoU drop <= 0.0100 (absolute metric fractions).
- Smallest configuration within both tolerances: **tied_r12** ranks (4, 12, 12), 1,234,944 parameters, 98.77% reduction (81.59x smaller).
- No adjacent tied-rank IoU drop exceeded 0.0100.

## Configuration metrics

| Configuration | Ranks | Attention params | Reduction | Compression | Accuracy (mean +/- std) | Oil-spill IoU (mean +/- std) | Mean IoU, no background | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_4_64_64 | (4, 64, 64) | 7,105,536 | 92.95% | 14.18x | 92.79% +/- 0.00% | 77.16% +/- 0.00% | 77.16% +/- 0.00% | 84.32% +/- 0.00% | 90.09% +/- 0.00% |
| tied_r48 | (4, 48, 48) | 5,188,608 | 94.85% | 19.42x | 92.77% +/- 0.00% | 77.13% +/- 0.00% | 77.13% +/- 0.00% | 84.19% +/- 0.00% | 90.20% +/- 0.00% |
| MHA | MHA | 100,761,600 | 0.00% | 1.00x | 92.56% +/- 0.00% | 77.07% +/- 0.00% | 77.07% +/- 0.00% | 82.26% +/- 0.00% | 92.44% +/- 0.00% |
| r1_1 | (1, 64, 64) | 6,513,408 | 93.54% | 15.47x | 92.78% +/- 0.00% | 77.06% +/- 0.00% | 77.06% +/- 0.00% | 84.51% +/- 0.00% | 89.73% +/- 0.00% |
| r1_2 | (2, 64, 64) | 6,710,784 | 93.34% | 15.01x | 92.74% +/- 0.00% | 76.97% +/- 0.00% | 76.97% +/- 0.00% | 84.43% +/- 0.00% | 89.71% +/- 0.00% |
| tied_r12 | (4, 12, 12) | 1,234,944 | 98.77% | 81.59x | 92.68% +/- 0.00% | 76.95% +/- 0.00% | 76.95% +/- 0.00% | 83.88% +/- 0.00% | 90.31% +/- 0.00% |
| tied_r32 | (4, 32, 32) | 3,369,984 | 96.66% | 29.90x | 92.60% +/- 0.00% | 76.74% +/- 0.00% | 76.74% +/- 0.00% | 83.64% +/- 0.00% | 90.29% +/- 0.00% |
| ref_4_16_16 | (4, 16, 16) | 1,649,664 | 98.36% | 61.08x | 92.56% +/- 0.00% | 76.65% +/- 0.00% | 76.65% +/- 0.00% | 83.52% +/- 0.00% | 90.30% +/- 0.00% |
| tied_r24 | (4, 24, 24) | 2,497,536 | 97.52% | 40.34x | 92.51% +/- 0.00% | 76.51% +/- 0.00% | 76.51% +/- 0.00% | 83.45% +/- 0.00% | 90.20% +/- 0.00% |

## CSV and log cross-check

- All 9 configuration summaries match the CSV (allowing for rounded log values).

## Summary recorded in the text log

- `Smallest model within baseline tolerances (accuracy drop <= 0.0100, oil-spill IoU drop <= 0.0100): tied_r12 ranks=(4,12,12), 1,234,944 attention params (98.8% reduction, 81.59x smaller).`
- `No adjacent tied-rank oil-spill IoU drop exceeded 0.0100.`

Metric means and standard deviations come from the CSV. Tolerance and knee values are read from the log when available; otherwise the analyzer defaults are used.
