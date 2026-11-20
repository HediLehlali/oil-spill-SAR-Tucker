"""Smoke test: build each model, forward pass, and check full-rank Tucker reproduces the dense model."""
import sys
import time

import torch

from seg_bench.models import MODELS, build
from seg_bench.adapters import tuckerize
from seg_bench.tucker_core import clear_svd_cache


def main(keys):
    dev = torch.device("cuda")
    for key in keys:
        t0 = time.time()
        m = build(key).to(dev).eval()
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 224, 224, device=dev)
        with torch.no_grad():
            y = m(x)
        print(f"{key}: params={n/1e6:.2f}M out={tuple(y.shape)} build={time.time()-t0:.1f}s")
        if hasattr(m, "transfer_report"):
            print("  transfer:", m.transfer_report)
        stats = tuckerize(m, m.tucker_families, key, energy=0.999999)
        with torch.no_grad():
            y2 = m(x)
        err = (y - y2).abs().max().item() / (y.abs().max().item() + 1e-9)
        print(f"  full-rank tucker: layers={stats['n_layers']} dense={stats['dense_attn_params']/1e6:.2f}M "
              f"tucker={stats['tucker_attn_params']/1e6:.2f}M rel.err={err:.2e}")
        for level in ((1 / 3, 1 / 4), (1 / 2, 1 / 2), (1, 1)):
            s = tuckerize(build(key).to(dev).eval(), m.tucker_families, key, level=level)
            print(f"  level {level}: compression={s['compression']:.2f}x")
        del m
        clear_svd_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
