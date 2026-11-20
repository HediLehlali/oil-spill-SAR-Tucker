"""Measure training throughput (iterations/s) of each model at batch 8, bf16."""
import sys
import time

import torch

from seg_bench.models import MODELS, build


def main(keys):
    dev = torch.device("cuda")
    for key in keys:
        m = build(key).to(dev).train()
        m.set_class_weights(torch.tensor([1.0, 2.0], device=dev))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
        x = torch.randn(8, 3, 224, 224, device=dev)
        y = (torch.rand(8, 224, 224, device=dev) > 0.8).long()
        torch.cuda.reset_peak_memory_stats()
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m.training_loss(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
        print(f"{key}: {dt:.3f}s/iter -> {dt * 806 / 60:.1f} min/epoch, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB",
              flush=True)
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
