# Parallel Rank Sweep Guide

## Quick Start

The `rank_sweep.py` script has been updated to support parallel evaluation of different Tucker rank configurations. This can significantly speed up your experiments when using `--parallel`.

### Basic Usage

**Sequential (original behavior):**
```bash
python rank_sweep.py --seeds 3 --epochs 80
```

**Parallel (faster on multi-GPU systems):**
```bash
python rank_sweep.py --seeds 3 --epochs 80 --parallel
```

## What Gets Parallelized?

The rank sweep is divided into two phases:

### Phase A: Baseline & Reference (Sequential)
- Trains the baseline MHA configuration
- Trains 2 reference Tucker configurations
- Always runs sequentially for stability and determinism

### Phase B: Rank Sweep (Parallelizable)
- Trains multiple Tucker rank configurations
- Each configuration has its own set of seeds
- When `--parallel` is enabled, rank configurations are evaluated in parallel

## Performance Expectations

### Single GPU System
- **Without parallelization:** Trains configurations one-by-one
- **With parallelization:** Overlaps I/O and preprocessing, provides modest speedup
- **Recommendation:** Use `--parallel` if you have >= 2 rank configurations

### Multi-GPU System
- **Without parallelization:** Uses only one GPU at a time
- **With parallelization:** Distributes configurations across GPUs
- **Speedup:** Near-linear with number of GPUs (limited by GPU memory constraints)
- **Recommendation:** Always use `--parallel`

## Advanced Configuration

### Control Number of Workers

By default, the script detects your GPU count:
```bash
# Auto-detects GPU count (or 1 if CPU-only)
python rank_sweep.py --seeds 3 --epochs 80 --parallel
```

To manually set the number of workers:
```bash
# Use exactly 2 workers
python rank_sweep.py --seeds 3 --epochs 80 --parallel --parallel-workers 2

# Use 4 workers (even if you have 8 GPUs)
python rank_sweep.py --seeds 3 --epochs 80 --parallel --parallel-workers 4
```

### Combine with Other Options

```bash
# Parallel sweep with smaller subset for testing
python rank_sweep.py --seeds 2 --epochs 20 --parallel --subset 64

# Parallel sweep with custom model
python rank_sweep.py --seeds 3 --epochs 80 --parallel --model-name vit-b16

# Parallel sweep with custom output path
python rank_sweep.py --seeds 3 --epochs 80 --parallel --output my_results.csv
```

## Implementation Details

### How Parallelization Works

1. **Phase A (Sequential):** Baseline and reference configs run in the main process
2. **Phase B (Parallel):** Rank configurations are grouped into tasks
3. **Worker Pool:** Creates N worker processes (one per GPU by default)
4. **Task Distribution:** Each worker trains a rank configuration with all seeds
5. **Result Collection:** Results from all workers are combined into the final CSV

### Why Datasets Instead of DataLoaders?

To make the rank sweep parallelizable, the code:
- Passes **raw datasets** (train_set, val_set) to worker processes
- Each worker **creates its own DataLoaders** inside the training function
- This avoids trying to serialize PyTorch DataLoaders (which is problematic)

### Memory Considerations

- Each worker maintains its own model, optimizer, and data loaders
- On GPU systems, models run on the same GPU (GPU ID mapping may apply)
- Memory usage scales with `num_workers` × (model + batch)
- If you run out of memory, reduce `--batch-size` or `--parallel-workers`

## Troubleshooting

### Issue: "CUDA out of memory"
```bash
# Reduce batch size
python rank_sweep.py --seeds 3 --epochs 80 --parallel --batch-size 1

# Reduce number of workers
python rank_sweep.py --seeds 3 --epochs 80 --parallel --parallel-workers 1
```

### Issue: Script hangs or is slow
- Verify GPU utilization with `nvidia-smi` or equivalent
- Check if batch size is too small (increases overhead)
- Try reducing `--parallel-workers` to see if it helps

### Issue: Results differ between sequential and parallel runs
- This is **expected** due to different random seed initialization order
- Use the same `--seeds` value for fair comparison
- Parallelization does not affect individual config training, only order

## Performance Comparison Example

Assume you have 4 GPUs and want to evaluate 12 rank configurations with 3 seeds each:

**Sequential (without `--parallel`):**
- Total time ≈ 12 configs × 3 seeds × T_seed ≈ 12 × 3 × T_seed
- GPU utilization: ~25% (only 1 GPU used)

**Parallel with 4 workers (with `--parallel`):**
- Total time ≈ 3 configs per batch × 3 seeds × T_seed ≈ 3 × T_seed
- GPU utilization: ~100% (all 4 GPUs active)
- **Theoretical speedup: ~4x**

## Workflow Tips

### For Initial Testing
```bash
# Quick test with --smoke (always fast)
python rank_sweep.py --smoke

# Small parallel test with 2 configs, 1 seed, 5 epochs
python rank_sweep.py --seeds 1 --epochs 5 --parallel --parallel-workers 2
```

### For Production Runs
```bash
# Full parallel sweep
python rank_sweep.py --seeds 3 --epochs 80 --parallel

# Generate analysis
python analyze_output.py
```

### Resuming Interrupted Runs
- The script always overwrites the output CSV
- If interrupted, run the command again from scratch
- All results will be re-computed and written to the same CSV

## Summary

✅ **Use `--parallel` when:**
- You have multiple GPUs
- Your rank grid has many configurations
- You want to reduce total training time

❌ **Skip `--parallel` when:**
- You're debugging (easier to trace sequential execution)
- You have limited GPU memory
- You need perfectly reproducible output (though differences should be minimal)
