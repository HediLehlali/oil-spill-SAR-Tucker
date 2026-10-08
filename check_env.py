"""
check_env.py
============
Usage:
    python check_env.py
"""

import sys
import subprocess

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

# ── 1. Python ─────────────────────────────────────────
section("Python")
print(f"  Version : {sys.version}")

# ── 2. PyTorch ────────────────────────────────────────
section("PyTorch")
try:
    import torch
    print(f"  Version         : {torch.__version__}")
    print(f"  CUDA disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU détecté     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM totale     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Version CUDA    : {torch.version.cuda}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute cap.    : {cap[0]}.{cap[1]}")

        # Flash Attention dispo via SDPA ?
        # PyTorch >= 2.0 expose scaled_dot_product_attention
        has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        print(f"\n  scaled_dot_product_attention (Flash) : {'✓ disponible' if has_sdpa else '✗ absent (PyTorch < 2.0)'}")

        # Backends Flash Attention
        if has_sdpa and hasattr(torch.backends, "cuda"):
            fa  = torch.backends.cuda.flash_sdp_enabled()
            mem = torch.backends.cuda.mem_efficient_sdp_enabled()
            mat = torch.backends.cuda.math_sdp_enabled()
            print(f"  Flash SDP       : {'✓' if fa else '✗'}")
            print(f"  Mem-efficient   : {'✓' if mem else '✗'}")
            print(f"  Math SDP        : {'✓' if mat else '✗'}")
    else:
        print("\n  ⚠  CUDA non disponible.")
        print("     Les benchmarks tourneront sur CPU (résultats moins représentatifs).")
        print("     Pour activer CUDA, installe PyTorch avec :")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("  ✗ PyTorch non installé")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# ── 3. Dépendances optionnelles ───────────────────────
section("Dépendances")
deps = {
    "numpy"        : "numpy",
    "matplotlib"   : "matplotlib",
    "timm"         : "timm (modèles ViT pré-entraînés)",
    "transformers" : "transformers (HuggingFace)",
    "einops"       : "einops (Tucker Attention)",
    "jupyter"      : "jupyter (notebooks)",
}
missing = []
for pkg, label in deps.items():
    try:
        __import__(pkg)
        print(f"  ✓ {label}")
    except ImportError:
        print(f"  ✗ {label}  ← manquant")
        missing.append(pkg)

if missing:
    print(f"\n  Pour installer les manquants :")
    print(f"  pip install {' '.join(missing)}")

# ── 4. Test rapide GPU ────────────────────────────────
if torch.cuda.is_available():
    section("Test GPU (matmul 1024×1024)")
    import time
    device = torch.device("cuda")
    A = torch.randn(1024, 1024, device=device)
    B = torch.randn(1024, 1024, device=device)
    # Warmup
    _ = A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        C = A @ B
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 50 * 1000
    print(f"  Matmul 1024×1024 (moyenne 50 runs) : {elapsed:.3f} ms")
    print(f"  GPU opérationnel ✓")

section("Résumé")
if torch.cuda.is_available():
    print("  ✓ Setup complet")
else:
    print("  ⚠  Setup CPU uniquement — les scripts fonctionnent mais plus lentement")
    print("     Réinstalle PyTorch avec support CUDA pour les vrais benchmarks GPU")
print()
