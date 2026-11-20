# ViT-S/16 Batch Size 8 Runs Analysis
## Oil Spill Segmentation with Tucker Attention Compression

**Date:** 2026-11-08 | **Dataset:** SAR (6,455 train, 1,615 validation) | **Encoder:** `WinKawaks/vit-small-patch16-224` (ViT-S/16)  
**Epochs:** 20 (stage 1 = 5 epochs) | **Batch Size:** 8 (assumed, same as the ViT-B runs; not recorded in the logs)

Source files: `out_baseline.txt`, `out_tucker.txt`, `out_tucker_unfreeze.txt` (this folder).

---

## Executive Summary

Three configurations were evaluated on SAR oil-spill segmentation:

1. **Baseline** — Full ViT attention (7.10M attention parameters), encoder frozen, decoder trained for 20 epochs
2. **Tucker (Frozen)** — Tucker-decomposed attention (0.32M parameters, 21.9× smaller), encoder frozen
3. **Tucker (Unfrozen)** — Same Tucker compression, last encoder layers fine-tuned from epoch 6 (number of unfrozen layers not recorded in the logs)

### Key Findings

- **Best validation F1:** Baseline (0.8887), then Tucker unfrozen (0.8838), then Tucker frozen (0.8798)
- **F1 gap vs baseline:** frozen −0.0089 (−1.0%), unfrozen −0.0049 (−0.6%)
- **Oil-spill IoU gap vs baseline:** frozen −0.0143 (−1.8%), unfrozen −0.0079 (−1.0%)
- **Tucker brings no speed benefit on ViT-S:** inference is ~4% *slower* (2.38 / 2.37 vs 2.28 ms per image)
- **Memory:** Tucker frozen saves only 8% (317 → 291 MB); Tucker unfrozen uses **63% more** than baseline (517 MB)
- **Training cost:** frozen +8% train time (+4.5% total), unfrozen +42% train time (+29% total)
- **Baseline is the best choice on this small encoder:** most accurate, fastest at inference and already very light (317 MB, 2.3 ms/image)

---

## Performance Metrics

### Best Validation Performance (per run, best-F1 epoch; epoch 17 for all three)

| Metric | Baseline | Tucker (Frozen) | Tucker (Unfrozen) | Δ Frozen−Base | Δ Unfrozen−Base |
|--------|----------|-----------------|-------------------|---------------|-----------------|
| **F1** | **0.8887** | 0.8798 | 0.8838 | −0.0089 | −0.0049 |
| **Mean F1** | **0.9232** | 0.9177 | 0.9200 | −0.0055 | −0.0032 |
| **Oil-Spill IoU** | **0.7997** | 0.7854 | 0.7918 | −0.0143 | −0.0079 |
| **Mean IoU** | **0.8593** | 0.8502 | 0.8540 | −0.0091 | −0.0053 |
| **Accuracy** | **0.9387** | 0.9351 | 0.9365 | −0.0036 | −0.0022 |
| **Precision** | 0.8732 | **0.8813** | 0.8747 | +0.0081 | +0.0015 |
| **Recall** | **0.9048** | 0.8784 | 0.8931 | −0.0264 | −0.0117 |
| **Loss (best epoch)** | **0.3588** | 0.3760 | 0.3662 | +0.0172 | +0.0074 |
| **Loss (epoch 20)** | **0.3531** | 0.3724 | 0.3629 | +0.0193 | +0.0098 |

### Comparison across encoder sizes (batch 8, best epoch)

| Config | ViT-S F1 / IoU | ViT-B F1 / IoU | ViT-L F1 / IoU |
|--------|----------------|----------------|----------------|
| Baseline | **0.8887** / **0.7997** | 0.8875 / 0.7977 | 0.8841 / 0.7922 |
| Tucker (Frozen) | 0.8798 / 0.7854 | **0.8836** / **0.7915** | 0.8727 / 0.7742 |
| Tucker (Unfrozen) | **0.8838** / **0.7918** | 0.8802 / 0.7861 | 0.8803 / 0.7861 |

| IoU gap vs baseline | ViT-S | ViT-B | ViT-L |
|---------------------|-------|-------|-------|
| Tucker frozen | −1.43 pt | −0.62 pt | −1.80 pt |
| Tucker unfrozen | −0.79 pt | −1.16 pt | −0.61 pt |
| Attention params (MHA → Tucker) | 7.10M → 0.32M (21.9×) | 28.35M → 0.62M (45.4×) | 100.8M → 1.65M (61×) |

- The baseline accuracy does **not** improve with encoder size: ViT-S is as good as (nominally better than) ViT-B and ViT-L, with differences within noise.
- The Tucker accuracy penalty does not follow encoder size (it is non-monotonic), which is consistent with these gaps being of the same order as run-to-run noise.

---

## Per-Epoch Dynamics

| Metric (oil-spill IoU) | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|------------------------|----------|-----------------|-------------------|
| Epoch 1 | 0.7065 | 0.7208 | 0.7207 |
| Mean epochs 1–5 | 0.7440 | 0.7376 | 0.7378 |
| Max over 20 epochs | **0.7997** | 0.7854 | 0.7918 |
| Mean epochs 16–20 | **0.7818** | 0.7697 | 0.7710 |
| Mean epochs 11–20 | **0.7775** | 0.7686 | 0.7698 |
| Std epochs 11–20 | 0.0177 | 0.0120 | 0.0129 |
| Min over epochs 2–20 | 0.7410 | 0.7188 | 0.7156 |
| Largest epoch-to-epoch swing | 0.0545 | 0.0390 | 0.0467 |
| IoU at epoch 20 | **0.7836** | 0.7602 | 0.7559 |

### Early Training (Epochs 1–5)
- Tucker runs start *higher* than the baseline (epoch 1 IoU 0.721 vs 0.707), then the baseline overtakes them around epoch 3–6.
- Tucker frozen and unfrozen are identical through epoch 5 (same configuration; only tiny numerical differences), as expected.

### Middle Training (Epochs 6–15)
- **Unfrozen run switches to fine-tuning at epoch 6**, without any visible immediate change: epoch 6–10 IoU values (0.767, 0.779, 0.733, 0.777, 0.754) are within ±0.01 of the frozen run's values (0.770, 0.779, 0.745, 0.777, 0.758). The loss is slightly lower from epoch 10 onward (0.3852 vs 0.3918 at epoch 10).
- All three runs oscillate strongly. Peaks and dips are **synchronized across runs** (peaks at epochs 9, 11, 17; dips at epochs 8, 10, 16), which suggests the same data ordering/seed and that the fluctuations are mostly epoch-specific (precision/recall swing) rather than caused by the configuration.

### Late Training (Epochs 16–20)
- **All runs peak at epoch 17** (IoU 0.7997 / 0.7854 / 0.7918), then drift down.
- **Epoch 20 is a dip for the Tucker runs** (IoU 0.7602 / 0.7559, recall ≈ 0.95, precision ≈ 0.79), versus 0.7836 for the baseline. Because the best checkpoint is selected on F1, this does not affect the reported best metrics, but the last-epoch Tucker models are 2.5 pt (frozen) and 3.6 pt (unfrozen) below their own best epoch.

### Robust comparison (averaged over epochs 11–20)
- Frozen Tucker: −0.89 pt IoU vs baseline
- Unfrozen Tucker: −0.77 pt IoU vs baseline
- Frozen vs unfrozen: ~0.1 pt apart, i.e. equivalent. The +0.64 pt unfrozen advantage at the best epoch is mostly an epoch-selection effect.

---

## Error Characteristics

### Confusion matrix at best epoch (pixels, rows = true, cols = predicted)

| Run | True negatives | False positives | False negatives | True positives |
|-----|----------------|-----------------|-----------------|----------------|
| Baseline | 56,234,719 | 2,880,742 | 2,087,076 | 19,831,703 |
| Tucker (Frozen) | 56,521,140 | 2,594,321 (−9.9%) | 2,665,299 (+27.7%) | 19,253,480 |
| Tucker (Unfrozen) | 56,310,779 | 2,804,682 (−2.6%) | 2,343,728 (+12.3%) | 19,575,051 |

### Precision vs. Recall trade-off
- At the best epoch, Tucker on ViT-S is **precision-oriented** (frozen: P = 0.881, R = 0.878): fewer false alarms, but 12–28% more missed oil pixels. This is the opposite of the ViT-B and ViT-L runs, where Tucker was recall-oriented.
- **Unfrozen Tucker is between the two:** P = 0.875, R = 0.893.
- Over epochs 11–20, all runs are recall-heavy on most epochs (average |P − R| = 0.115 baseline, 0.102 frozen, 0.102 unfrozen). The best-F1 epochs are the ones where precision and recall happen to be balanced.

---

## Resource Utilization

### Training Time

| Stage | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|-------|----------|-----------------|-------------------|
| Setup | 33.9 s | 32.1 s | 31.9 s |
| Training | 554.1 s | 596.1 s | 786.1 s |
| Data wait | 159.8 s | 93.4 s | 93.8 s |
| Eval | 185.3 s | 179.7 s | 178.6 s |
| **Total** | **774.1 s** | **808.5 s** | **997.3 s** |
| **Total vs baseline** | — | +4.5% | +28.8% |
| **Training vs baseline** | — | +7.6% | +41.9% |

- The baseline shows 66 s more data-wait time than the Tucker runs, so its timings are partly influenced by data loading; the frozen-vs-baseline training gap should be read with care.
- Unfrozen Tucker adds +42% training time due to backpropagation through the fine-tuned layers for 15 epochs.

### GPU Memory and Utilization

| Metric | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|--------|----------|-----------------|-------------------|
| Peak GPU Mem | 317 MB | **291 MB** | 517 MB |
| vs baseline | — | −8.4% | **+63.0%** |
| Mean GPU util | 53.5% | 51.7% | 50.3% |

- Memory savings from Tucker are small on ViT-S (−8%), versus −30% on ViT-B and −32% on ViT-L.
- GPU utilization is only ~50–54% (vs 88–96% for ViT-B/L): the small model is data-loading/CPU bound, which also limits any speed gain from a lighter attention block.

### Inference Performance

| Metric | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|--------|----------|-----------------|-------------------|
| Inference time (best epoch, 1615 images) | 3.65 s | 3.83 s | 3.85 s |
| Profiled ms per image | **2.28** | 2.38 | 2.37 |
| Speed vs baseline | — | 0.96× (+4.4% slower) | 0.96× (+4.1% slower) |

Attention is only a small part of the compute of ViT-S, and the Tucker factorized operations add extra small kernels, so the 21.9× attention-parameter reduction does not translate into faster inference.

### Efficiency by encoder size (batch 8)

| Metric | ViT-S | ViT-B | ViT-L |
|--------|-------|-------|-------|
| Baseline ms/image | 2.28 | 6.02 | 27.17 |
| Baseline peak memory | 317 MB | 688 MB | 2352 MB |
| Baseline total time | 774 s | 914 s | 2503 s |
| Tucker frozen: inference vs baseline | +4% (slower) | −11% | −16% |
| Tucker frozen: memory vs baseline | −8% | −30% | −32% |
| Tucker unfrozen: memory vs baseline | +63% | +47% | −19% |
| Tucker unfrozen: total time vs baseline | +29% | +29% | +16% |

The benefit of Tucker grows with encoder size; on ViT-S it is essentially absent.

---

## Statistical Caveats

1. **Single runs:** each configuration was trained once; no confidence intervals.
2. **Best-epoch selection on the validation set** is optimistic; all three runs peak at the same epoch (17), so the comparison is at least aligned, but the peak is a single noisy epoch.
3. **Epoch-level noise is as large as the differences:** swings of 4–6 pt IoU between adjacent epochs vs 0.8–1.4 pt gaps between configurations.
4. **No held-out test set:** rankings may not transfer to unseen data.
5. **Batch size and number of unfrozen layers** are not recorded in the logs and were assumed from the ViT-B commands.
6. **Timing noise:** data-wait time differs by 66 s between the baseline and the Tucker runs, which makes timing comparisons approximate.

**Recommendation:** run 2–3 seeds per configuration and compare averaged late-epoch metrics rather than the single best epoch.

---

## Recommendations

### For Production Deployment
- **Baseline** is the best option for ViT-S: best accuracy (F1 0.8887, IoU 0.7997), fastest inference (2.28 ms/image) and only 317 MB. Tucker brings no gain at this scale.
- **Tucker (Frozen)** only makes sense if memory is the sole constraint (−8%), at the price of −1.4 pt IoU.
- **Tucker (Unfrozen)** is not justified: +63% memory and +42% training time for ≈ 0.1 pt IoU over frozen on averaged late epochs.

### For Research
1. **Multi-seed validation** (see caveats).
2. **Larger encoders for Tucker:** the speed/memory gains scale with encoder size, so ViT-B/L remain the relevant targets for compression.
3. **Remove the data-loading bottleneck** (GPU util ≈ 50%) with more workers or caching before drawing conclusions on ViT-S timings.
4. **Learning-rate schedule / checkpoint averaging:** the late-epoch dips (epoch 20) suggest a decay schedule or averaging of the last epochs could stabilize all three runs.
5. **Rank sweep on ViT-S** with `rank_sweep.py`: ranks (4, 16, 16) may be too aggressive or too conservative for a 384-dim encoder.

---

## Conclusion

On ViT-S/16, **Tucker attention compression costs 0.6–1.0% F1 (1.0–1.8% IoU) and does not pay off**: no inference speedup (−4%), little memory saving when frozen (−8%), and a large memory and time overhead when unfrozen (+63% memory, +42% training time). The plain baseline is both the most accurate and the most efficient configuration at this model size. Combined with the ViT-B and ViT-L reports, Tucker becomes beneficial only as the encoder grows (−30% memory and −11 to −16% inference time from ViT-B upward).
