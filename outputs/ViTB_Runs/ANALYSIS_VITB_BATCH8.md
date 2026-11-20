# ViT-B/16 Batch Size 8 Runs Analysis
## Oil Spill Segmentation with Tucker Attention Compression

**Date:** 2026-11-08 | **Dataset:** SAR (6,455 train, 1,615 validation) | **Encoder:** `google/vit-base-patch16-224-in21k` (ViT-B/16)  
**Epochs:** 20 (stage 1 = 5 epochs) | **Batch Size:** 8 | **Workers:** 2

Source files: `out_baseline.txt`, `out_tucker.txt`, `out_tucker_unfreeze.txt` (this folder).

---

## Executive Summary

Three configurations were evaluated on SAR oil-spill segmentation:

1. **Baseline** — Full ViT attention (28.35M attention parameters), encoder frozen, decoder trained for 20 epochs
2. **Tucker (Frozen)** — Tucker-decomposed attention (0.62M parameters, 45.4× smaller), encoder frozen (`--tucker`)
3. **Tucker (Unfrozen)** — Same Tucker compression, last 5 encoder layers fine-tuned from epoch 6 (`--tucker --unfreeze-last 5`)

### Key Findings

- **Best validation F1:** Baseline (0.8875), then Tucker frozen (0.8836), then Tucker unfrozen (0.8802)
- **F1 gap vs baseline:** frozen −0.0039 (−0.4%), unfrozen −0.0073 (−0.8%)
- **Oil-spill IoU gap vs baseline:** frozen −0.0062 (−0.8%), unfrozen −0.0116 (−1.5%)
- **Memory:** Tucker frozen cuts peak GPU memory by ~30% (688 → 480 MB); Tucker unfrozen uses **47% more** than baseline (1014 MB)
- **Inference speed:** both Tucker variants are ~11% faster (5.33 / 5.35 vs 6.02 ms per image)
- **Training cost:** frozen Tucker costs the same as baseline (+1% train time); unfrozen adds +51% train time (+29% total)
- **Unfreezing does not help here:** at the best epoch the frozen variant is actually ahead, and averaged over the last epochs the two are equivalent (see Per-Epoch Dynamics)

---

## Performance Metrics

### Best Validation Performance (per run, best-F1 epoch)

| Metric | Baseline | Tucker (Frozen) | Tucker (Unfrozen) | Δ Frozen−Base | Δ Unfrozen−Base |
|--------|----------|-----------------|-------------------|---------------|-----------------|
| **F1** | **0.8875** | 0.8836 | 0.8802 | −0.0039 | −0.0073 |
| **Mean F1** | **0.9221** | 0.9189 | 0.9168 | −0.0032 | −0.0053 |
| **Oil-Spill IoU** | **0.7977** | 0.7915 | 0.7861 | −0.0062 | −0.0116 |
| **Mean IoU** | **0.8575** | 0.8519 | 0.8484 | −0.0056 | −0.0091 |
| **Accuracy** | **0.9376** | 0.9342 | 0.9328 | −0.0034 | −0.0048 |
| **Precision** | **0.8659** | 0.8470 | 0.8500 | −0.0189 | −0.0159 |
| **Recall** | 0.9101 | **0.9235** | 0.9127 | +0.0134 | +0.0026 |
| **Loss (best epoch)** | **0.3578** | 0.3667 | 0.3760 | +0.0089 | +0.0182 |
| **Loss (epoch 20)** | **0.3460** | 0.3667 | 0.3602 | +0.0207 | +0.0142 |
| **Best Epoch** | 14 | 20 | 13 | — | — |

Note: the frozen Tucker run reaches its best F1 at the last epoch (20), so it may not have fully converged.

### Comparison with the ViT-L (DINOv2-Large) batch 8 runs

| Config | ViT-B F1 | ViT-L F1 | ViT-B IoU | ViT-L IoU |
|--------|----------|----------|-----------|-----------|
| Baseline | **0.8875** | 0.8841 | **0.7977** | 0.7922 |
| Tucker (Frozen) | **0.8836** | 0.8727 | **0.7915** | 0.7742 |
| Tucker (Unfrozen) | 0.8802 | **0.8803** | 0.7861 | 0.7861 |

- ViT-B/16 is **at least as accurate as DINOv2-Large** on this task (differences ≤ 0.6 pt IoU for the baseline, within noise), while being far cheaper (6.0 vs 27.2 ms/image, 688 vs 2352 MB peak memory).
- The Tucker accuracy penalty is **smaller on ViT-B for the frozen variant** (−0.6 pt IoU vs −1.8 pt on ViT-L).
- The unfrozen variant behaves differently across encoders (ViT-B: worse than frozen; ViT-L: better than frozen). The setups also differ (`--unfreeze-last 5` here; the ViT-L report mentions the last 3 layers), so this should not be over-interpreted.

---

## Per-Epoch Dynamics

| Metric (oil-spill IoU) | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|------------------------|----------|-----------------|-------------------|
| Epoch 1 | 0.7430 | 0.7396 | 0.7388 |
| Max over 20 epochs | **0.7977** | 0.7915 | 0.7861 |
| Mean epochs 16–20 | **0.7831** | 0.7662 | 0.7697 |
| Mean epochs 11–20 | **0.7872** | 0.7745 | 0.7756 |
| Std epochs 11–20 | **0.0062** | 0.0141 | 0.0088 |
| Min over epochs 2–20 | 0.7300 | 0.6959 | 0.6940 |
| Largest epoch-to-epoch swing | 0.0478 | 0.0597 | 0.0606 |
| Epochs 11–20 with IoU < 0.76 | 0 | 2 (ep 17, 18) | 1 (ep 18) |

### Early Training (Epochs 1–5)
- All three start at the same level (IoU 0.739–0.743 at epoch 1; no cold-start gap as seen on ViT-L).
- Baseline reaches F1 0.8809 by epoch 5; Tucker runs reach 0.8722 / 0.8683.
- Epoch 2 is a dip for all runs (baseline IoU 0.7300, frozen 0.6959, unfrozen 0.6940): recall jumps above 0.95 while precision falls.

### Middle Training (Epochs 6–15)
- **Baseline:** already at F1 0.8873 at epoch 6 and stays between 0.8635 and 0.8875.
- **Tucker frozen:** climbs steadily from F1 0.8730 (epoch 6) to 0.8810 (epoch 14).
- **Tucker unfrozen:** switches to fine-tuning at epoch 6. The first fine-tuning epochs do not give a visible boost; epoch 7 is the lowest point of the run in this phase (F1 0.8378, IoU 0.7209). It then recovers to F1 0.8802 at epoch 13.
- **Epoch 7 dip is common to all three runs** (IoU 0.7598 / 0.7305 / 0.7209), with the same signature (recall ≥ 0.95, precision drop). Together with the epoch 2 dip, this suggests the same data ordering/seed is used across runs, and that the dips are linked to specific training steps rather than to the Tucker compression.

### Late Training (Epochs 16–20)
- **Baseline:** small oscillation (IoU 0.7772–0.7931), ends at F1 0.8846.
- **Tucker frozen:** unstable at epochs 16–18 (IoU 0.7657 → 0.7473 → 0.7520, precision down to 0.774) before recovering to its best value at epoch 20.
- **Tucker unfrozen:** same pattern at epoch 18 (IoU 0.7570), ends at F1 0.8783.

### Robust comparison (averaged over epochs 11–20)
The best-epoch ranking (frozen > unfrozen) depends on which single epoch is picked. Averaged over epochs 11–20:
- Frozen Tucker: −1.27 pt IoU vs baseline
- Unfrozen Tucker: −1.16 pt IoU vs baseline
- Frozen vs unfrozen: ~0.1 pt apart, i.e. equivalent.

---

## Error Characteristics

### Confusion matrix at best epoch (pixels, rows = true, cols = predicted)

| Run | True negatives | False positives | False negatives | True positives |
|-----|----------------|-----------------|-----------------|----------------|
| Baseline | 56,026,815 | 3,088,646 | 1,969,837 | 19,948,942 |
| Tucker (Frozen) | 55,460,518 | 3,654,943 (+18.3%) | 1,677,442 (−14.8%) | 20,241,337 |
| Tucker (Unfrozen) | 55,585,786 | 3,529,675 (+14.3%) | 1,913,884 (−2.8%) | 20,004,895 |

### Precision vs. Recall trade-off
- **Baseline** is the most balanced at the best epoch (P = 0.866, R = 0.910).
- **Tucker frozen** is recall-oriented (P = 0.847, R = 0.924): it detects more oil (fewer false negatives) at the price of ~18% more false alarms.
- **Tucker unfrozen** is close to baseline on recall (0.913) but still has +14% false positives (P = 0.850).
- Over epochs 11–20 the average |precision − recall| is 0.075 (baseline) vs 0.090 (frozen) and 0.089 (unfrozen): both Tucker variants swing more between over- and under-detection from epoch to epoch.

---

## Resource Utilization

### Training Time

| Stage | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|-------|----------|-----------------|-------------------|
| Setup | 39.6 s | 30.7 s | 31.1 s |
| Training | 576.3 s | 582.4 s | 868.6 s |
| Data wait | 108.4 s | 93.4 s | 95.5 s |
| Eval | 296.4 s | 273.7 s | 274.9 s |
| **Total** | **914.2 s** | **888.9 s** | **1176.2 s** |
| **Total vs baseline** | — | −2.8% | +28.7% |
| **Training vs baseline** | — | +1.1% | +50.7% |

- Frozen Tucker has essentially the same training cost as baseline (the encoder is frozen in both), and a slightly lower total time thanks to faster evaluation (−7.7%).
- Unfrozen Tucker spends 15 epochs backpropagating through the last 5 encoder layers, hence +51% training time.

### GPU Memory and Utilization

| Metric | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|--------|----------|-----------------|-------------------|
| Peak GPU Mem | 688 MB | **480 MB** | 1014 MB |
| vs baseline | — | **−30.3%** | **+47.2%** |
| Mean GPU util | 87.9% | 89.5% | 88.9% |

The unfrozen run needs gradients and optimizer states for the fine-tuned layers, plus activations kept for backpropagation, which is probably why it ends above the baseline (the baseline never backpropagates through the encoder).
Note that on ViT-L the unfrozen run stayed below baseline (1895 vs 2352 MB); on the much smaller ViT-B the training overhead dominates.

### Inference Performance

| Metric | Baseline | Tucker (Frozen) | Tucker (Unfrozen) |
|--------|----------|-----------------|-------------------|
| Inference time (best epoch, 1615 images) | 9.78 s | 8.62 s | 8.70 s |
| Profiled ms per image | 6.02 | 5.33 | 5.35 |
| Speedup | — | **1.13×** | **1.13×** |

Tucker removes 97.8% of the attention parameters (28.35M → 0.62M) but only gives ~11% faster inference, since the MLP blocks, patch embedding and decoder are unchanged.

---

## Statistical Caveats

1. **Single runs:** each configuration was trained once; no confidence intervals.
2. **Best-epoch selection on the validation set** is optimistic, and its ranking is unstable here (frozen Tucker peaks at epoch 20, unfrozen at 13, baseline at 14).
3. **Epoch-level noise is as large as the differences:** swings of 1–6 pt IoU between adjacent epochs vs 0.6–1.2 pt gaps between configurations.
4. **No held-out test set:** rankings may not transfer to unseen data.
5. **Mean-GPU-util / memory values** come from the training script's profiler and were not independently verified.

**Recommendation:** run 2–3 seeds per configuration and compare averaged late-epoch metrics rather than the single best epoch.

---

## Recommendations

### For Production Deployment
- **Tucker (Frozen)** is the best compromise on ViT-B: −0.4% F1 / −0.8% IoU, −30% memory, −11% inference time, no extra training cost. Prefer it if the recall-oriented behavior (more false alarms) is acceptable.
- **Tucker (Unfrozen)** is not justified in this setup: +51% training time and +47% memory over baseline for no clear accuracy gain over frozen.
- **Baseline** gives the best and most stable accuracy; at 688 MB and 6 ms/image it is already cheap, so Tucker's gain is modest on this encoder.

### For Research
1. **Multi-seed validation** (see caveats).
2. **Longer training** for frozen Tucker, whose best epoch is the last one.
3. **Fewer unfrozen layers or a lower fine-tuning learning rate** (e.g. `--unfreeze-last 2`/`3`, or LR warm-up at epoch 6) to test whether fine-tuning can help without the memory cost.
4. **Rank sweep on ViT-B** with `rank_sweep.py`, since the current ranks (4, 16, 16) give a small accuracy penalty and a 45× attention compression: higher compression may be affordable.
5. **Test-set evaluation** on a held-out split.

---

## Conclusion

**Tucker attention compression works on ViT-B/16 with a small accuracy cost.** Frozen Tucker loses 0.4% F1 (0.8% IoU) with a 45× smaller attention block, 30% less memory and 11% faster inference at no training-time cost. Fine-tuning the last 5 layers did not recover accuracy (and costs +51% training time and +47% memory vs baseline), so the benefit of unfreezing seen on ViT-L is not reproduced here. ViT-B/16 also matches DINOv2-Large accuracy on this dataset at a fraction of the cost.
