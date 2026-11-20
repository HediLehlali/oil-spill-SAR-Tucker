"""Workflow schematics: architecture of each model, what Tucker touches, and what each regime trains."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_IN, C_ENC, C_ATT, C_FFN, C_DEC, C_OUT = "#ffffff", "#d9e7f7", "#f6b26b", "#d9d9d9", "#d9ead3", "#ffffff"
EDGE = "#3d3d8f"


def box(ax, x, y, w, h, color, lw=1.4, ls="-", edge=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.4",
                                fc=color, ec=edge, lw=lw, ls=ls))


def text(ax, x, y, s, size=8, weight="normal", color="#111111", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.25)


def arrow(ax, p, q, color=EDGE, ls="-", lw=1.4, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                       connectionstyle=f"arc3,rad={rad}"))


def stage_box(ax, x, y, w, h, title, sub, attn=None, ffn=None, color=C_ENC):
    box(ax, x, y, w, h, color)
    text(ax, x + w / 2, y + h - 2.6, title, 8.5, "bold")
    text(ax, x + w / 2, y + h - 6.6, sub, 7)
    if attn or ffn:
        mh = 5.6
        if attn:
            box(ax, x + 1, y + 1 + (mh + 1 if ffn else 0), w - 2, mh, C_ATT, lw=1.0)
            text(ax, x + w / 2, y + 1 + mh / 2 + (mh + 1 if ffn else 0), attn, 6.5)
        if ffn:
            box(ax, x + 1, y + 1, w - 2, mh, C_FFN, lw=1.0)
            text(ax, x + w / 2, y + 1 + mh / 2, ffn, 6.5)


def draw_model(spec: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 9.4))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 94)
    ax.axis("off")
    text(ax, 80, 91, f"{spec['title']} - workflow on SAR oil-spill segmentation", 14, "bold", EDGE)
    text(ax, 80, 87.4, spec["subtitle"], 8.5, color="#444444")

    enc, dec = spec["encoder"], spec["decoder"]
    in_w, gap = 13, 3.2
    ex0 = 3 + in_w + gap
    ew = (157 - ex0 - gap * (len(enc) - 1)) / len(enc)
    eh, ey = 24, 58
    box(ax, 3, ey + 6, in_w, 12, C_IN)
    text(ax, 3 + in_w / 2, ey + 12, spec["input"], 7.5, "bold")
    arrow(ax, (3 + in_w + 0.3, ey + 12), (ex0 - 0.3, ey + 12))
    enc_x = []
    for i, st in enumerate(enc):
        x = ex0 + i * (ew + gap)
        enc_x.append(x)
        stage_box(ax, x, ey, ew, eh, st[0], st[1], st[2] if len(st) > 2 else None,
                  st[3] if len(st) > 3 else None)
        if i:
            arrow(ax, (x - gap + 0.3, ey + 12), (x - 0.3, ey + 12))
    text(ax, 3, ey + eh + 2.6, "ENCODER", 8, "bold", EDGE, ha="left")

    nested = any(len(d) > 2 for d in dec)
    dh, dy = (22, 31) if nested else (18, 33)
    dw = ew
    dec_x = []
    for j, d in enumerate(dec):
        x = 157 - (j + 1) * dw - j * gap
        dec_x.append(x)
        stage_box(ax, x, dy, dw, dh, d[0], d[1], d[2] if len(d) > 2 else None,
                  d[3] if len(d) > 3 else None, color=C_DEC)
        if j:
            arrow(ax, (x + dw + gap - 0.3, dy + 9), (x + dw + 0.3, dy + 9))
    text(ax, 3, dy + dh + 2.6, "DECODER / HEAD", 8, "bold", EDGE, ha="left")
    arrow(ax, (enc_x[-1] + ew / 2, ey - 0.3), (dec_x[0] + dw / 2, dy + dh + 0.3), lw=1.6)
    for ei, dj, label in spec.get("skips", []):
        arrow(ax, (enc_x[ei] + ew / 2, ey - 0.3), (dec_x[dj] + dw / 2, dy + dh + 0.3),
              ls="--", color="#7f7f7f", lw=1.1, rad=0.0)
        text(ax, enc_x[ei] + ew / 2 + 4, ey - 4, label, 6.5, color="#555555")
    box(ax, 3, dy + 3, in_w, 12, C_OUT)
    text(ax, 3 + in_w / 2, dy + 9, spec["output"], 7.5, "bold")
    arrow(ax, (dec_x[-1] - 0.3, dy + 9), (3 + in_w + 0.3, dy + 9))

    # legend
    lx = 8
    for color, label in ((C_ATT, "Attention: Tucker-compressed (HOSVD)"), (C_FFN, "FFN / MLP: untouched"),
                         (C_ENC, "Pretrained encoder"), (C_DEC, "Decoder / head")):
        box(ax, lx, 24.2, 3.2, 2.6, color, lw=0.9)
        text(ax, lx + 4.5, 25.5, label, 7.5, ha="left")
        lx += 38

    # regimes
    ry, rh, rw = 2.5, 16.5, 49.5
    regimes = [
        ("BASELINE", "Full model fine-tuned on the oil-spill set\nAll encoder + decoder weights trained\n(dense attention, nothing compressed)", "#cfe2f3"),
        ("TUCKER - FROZEN", spec["frozen_note"], "#fce5cd"),
        ("TUCKER - UNFROZEN", spec["unfrozen_note"], "#d9ead3"),
    ]
    for k, (name, body, col) in enumerate(regimes):
        x = 3 + k * (rw + 2.5)
        box(ax, x, ry, rw, rh, col)
        text(ax, x + rw / 2, ry + rh - 2.4, name, 9, "bold", EDGE)
        text(ax, x + rw / 2, ry + 5.3, body, 7.3)
        if k < 2:
            arrow(ax, (x + rw + 0.2, ry + rh / 2), (x + rw + 2.3, ry + rh / 2))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


FROZEN = ("Start from the best baseline checkpoint\nReplace {n} by Tucker blocks\n"
          "(HOSVD of per-head Q.K^T and V.O tensors)\nNo training: compress, then inference only")
UNFROZEN = ("Start from the compressed model\nFine-tune ALL weights on the same data\nTucker factors: encoder LR;  other weights: LR / 10\n"
            "{extra}")

SWIN_ENC = [
    ("Patch embed 4x4", "C=96, LN\n56x56 tokens"),
    ("Stage 1: 2 blocks", "dim 96, 3 heads\n56x56, window 7", "W-/SW-MSA  3 heads", "MLP 96-384-96"),
    ("Stage 2: 2 blocks", "dim 192, 6 heads\n28x28 (merge /2)", "W-/SW-MSA  6 heads", "MLP 192-768-192"),
    ("Stage 3: 6 blocks", "dim 384, 12 heads\n14x14 (merge /2)", "W-/SW-MSA 12 heads", "MLP 384-1536-384"),
    ("Stage 4: 2 blocks", "dim 768, 24 heads\n7x7 (merge /2)", "W-/SW-MSA 24 heads", "MLP 768-3072-768"),
]

SPECS = {
    "segformer_b2": dict(
        title="SegFormer-B2", subtitle="MiT-B2 hierarchical encoder (4 stages, 16 blocks) + All-MLP decoder  |  27.4M params  |  HF: nvidia/segformer-b2-finetuned-ade-512-512",
        input="SAR image\n224x224\n(gray->RGB)",
        encoder=[
            ("Overlap patch embed", "7x7 conv, stride 4\n56x56, C=64"),
            ("Stage 1: 3 blocks", "dim 64, 1 head, SR 8\n56x56", "Efficient SR-attention 1h", "Mix-FFN 64-256-64\n(+3x3 DWConv)"),
            ("Stage 2: 4 blocks", "dim 128, 2 heads, SR 4\n28x28", "Efficient SR-attention 2h", "Mix-FFN 128-512-128"),
            ("Stage 3: 6 blocks", "dim 320, 5 heads, SR 2\n14x14", "Efficient SR-attention 5h", "Mix-FFN 320-1280-320"),
            ("Stage 4: 3 blocks", "dim 512, 8 heads, SR 1\n7x7", "Efficient SR-attention 8h", "Mix-FFN 512-2048-512"),
        ],
        decoder=[
            ("Per-scale linear", "C_i -> 768 (x4 maps)\nupsample to 1/4"),
            ("Concat + fuse", "4x768 -> 768\n1x1 conv + BN + ReLU"),
            ("Classifier", "1x1 conv -> 2 classes\nbilinear x4"),
        ],
        skips=[(1, 0, "stage 1"), (2, 0, "stage 2"), (3, 0, "stage 3")],
        output="mask 224x224\n(2 classes)",
        frozen_note=FROZEN.format(n="all 16 SR-attention blocks"),
        unfrozen_note=UNFROZEN.format(extra="Mix-FFN, SR conv, decoder all trainable"),
    ),
    "upernet_swin_t": dict(
        title="Swin-T + UperNet", subtitle="Swin-T backbone (12 blocks) + UperNet (PPM + FPN) head  |  58.9M params  |  HF: openmmlab/upernet-swin-tiny",
        input="SAR image\n224x224\n(gray->RGB)",
        encoder=SWIN_ENC,
        decoder=[
            ("PPM on stage 4", "pool 1/2/3/6 + conv\n(512 ch)"),
            ("FPN top-down", "lateral 1x1 + add\n3x3 conv per level"),
            ("Fuse levels", "upsample + concat 4x512\n3x3 conv + BN + ReLU"),
            ("Classifier", "1x1 conv -> 2 classes\nbilinear x4"),
        ],
        skips=[(1, 1, "C2"), (2, 1, "C3"), (3, 1, "C4")],
        output="mask 224x224\n(2 classes)",
        frozen_note=FROZEN.format(n="all 12 window-attention blocks"),
        unfrozen_note=UNFROZEN.format(extra="MLPs, patch merging and UperNet head trainable"),
    ),
    "mask2former_swin_t": dict(
        title="Mask2Former (Swin-T)", subtitle="Swin-T backbone + multi-scale deformable pixel decoder + masked-attention transformer decoder (100 queries)  |  47.4M params",
        input="SAR image\n224x224\n(gray->RGB)",
        encoder=SWIN_ENC,
        decoder=[
            ("Masked-attention decoder", "100 queries, 9 layers", "masked-cross + self attn (8h)", "FFN 256-2048-256"),
            ("Pixel decoder", "6 x MSDeformAttn encoder\n(1/8,1/16,1/32) + FPN to 1/4\n(deformable attn NOT compressed)"),
            ("Prediction heads", "class MLP (K+1) + mask embed\nmask = embed . mask features"),
            ("Semantic inference", "sum_q softmax(class) x sigmoid(mask)\n-> argmax"),
        ],
        skips=[(2, 1, "1/16"), (3, 1, "1/32"), (1, 1, "1/8")],
        output="mask 224x224\n(2 classes)",
        frozen_note=FROZEN.format(n="all 30 (12 Swin, 9 self, 9 masked-cross)"),
        unfrozen_note=UNFROZEN.format(extra="Hungarian + mask/dice/CE loss (native)"),
    ),
    "swin_unetr": dict(
        title="Swin-UNETR (2D)", subtitle="Swin encoder (stages 2-2-2-2, C=48) + UNETR conv decoder; MONAI SwinUNETR(spatial_dims=2), init from the 3D BTCV bundle  |  25.1M params",
        input="SAR image\n224x224\n(1 channel)",
        encoder=[
            ("Patch embed 2x2", "C=48\n112x112 tokens"),
            ("Stage 1: 2 blocks", "dim 48, 3 heads\n112x112, window 7", "W-/SW-MSA  3 heads", "MLP 48-192-48"),
            ("Stage 2: 2 blocks", "dim 96, 6 heads\n56x56", "W-/SW-MSA  6 heads", "MLP 96-384-96"),
            ("Stage 3: 2 blocks", "dim 192, 12 heads\n28x28", "W-/SW-MSA 12 heads", "MLP 192-768-192"),
            ("Stage 4: 2 blocks", "dim 384, 24 heads\n14x14", "W-/SW-MSA 24 heads", "MLP 384-1536-384"),
        ],
        decoder=[
            ("Bottleneck enc10", "residual conv block\n(768 ch, 7x7)"),
            ("UpBlock x4", "transposed conv x2 +\nconcat skip + res conv\n(384 -> 192 -> 96 -> 48)"),
            ("Output conv", "1x1 conv -> 2 classes"),
        ],
        skips=[(1, 1, "skip enc2"), (2, 1, "enc3"), (3, 1, "enc4")],
        output="mask 224x224\n(2 classes)",
        frozen_note=FROZEN.format(n="all 8 window-attention blocks"),
        unfrozen_note=UNFROZEN.format(extra="UNETR decoder trainable (random init)"),
    ),
    "transunet": dict(
        title="TransUNet (R50-ViT-B/16)", subtitle="ResNet-50 (BiT/GN) -> 12-layer ViT-B/16 bottleneck -> cascaded-upsampler U-Net decoder  |  105.3M params  |  timm vit_base_r50_s16_224.orig_in21k",
        input="SAR image\n224x224\n(gray->RGB)",
        encoder=[
            ("R50 stem", "7x7 conv s2, GN, pool\nskip 1/2 (64 ch)"),
            ("R50 stage 1-2", "1/4 (256 ch) skip\n1/8 (512 ch) skip"),
            ("R50 stage 3", "1/16, 1024 ch\n1x1 proj -> 768 + pos"),
            ("ViT-B x12 blocks", "196 tokens, dim 768", "MHSA 12 heads", "MLP 768-3072-768"),
        ],
        decoder=[
            ("conv_more", "3x3 conv 768 -> 512\n14x14"),
            ("Cascaded upsampler", "4 x [up x2, concat skip,\n2x conv3x3-BN-ReLU]\n256 -> 128 -> 64 -> 16"),
            ("Seg head", "3x3 conv -> 2 classes"),
        ],
        skips=[(0, 1, "skip 1/2"), (1, 1, "skips 1/4, 1/8")],
        output="mask 224x224\n(2 classes)",
        frozen_note=FROZEN.format(n="all 12 ViT attention blocks"),
        unfrozen_note=UNFROZEN.format(extra="R50, MLPs and decoder all trainable"),
    ),
}


def draw_tucker_attention(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 62)
    ax.axis("off")
    text(ax, 80, 59, "Tucker attention: what is compressed (and what stays dense)", 14, "bold", EDGE)
    steps = [
        ("1. Dense MHA (per layer)", "H heads: Wq_h, Wk_h, Wv_h, Wo_h\n4 d^2 parameters", C_ATT),
        ("2. Fold per head", "W[h] = Wq_h Wk_h^T   (pre-softmax)\nWt[h] = (Wv_h Wo_h)^T   (post-softmax)\n-> two (H x d x d) tensors", "#fce5cd"),
        ("3. Truncated HOSVD", "W  ~ C x1 U1 x2 U2 x3 U3\nWt ~ Ct x1 Ut1 x2 Ut2 x3 Ut3\nranks (r1, r2, r3) = (a.H, b.d_head, b.d_head)", "#fce5cd"),
        ("4. Tucker block", "q = X U2, k = X U3, v = X Ut3\nsoftmax(q G k^T) v -> Gt -> Ut2 (+bias)\nshared factors across heads", C_ATT),
    ]
    w = 35
    for i, (t, b, c) in enumerate(steps):
        x = 3 + i * (w + 5.5)
        box(ax, x, 22, w, 28, c)
        text(ax, x + w / 2, 45, t, 9, "bold")
        text(ax, x + w / 2, 34, b, 7.6)
        if i:
            arrow(ax, (x - 5.2, 36), (x - 0.4, 36))
    box(ax, 3, 3, 72, 13, C_FFN)
    text(ax, 39, 9.5, "After attention (NOT compressed here): residual + LN, FFN/MLP (~2/3 of encoder params),\nPatch merging, decoder / head. The output projection Wo is folded into step 2 (post-softmax tensor).", 7.6)
    box(ax, 85, 3, 72, 13, C_DEC)
    text(ax, 121, 9.5, "Exact corrections vs the ViT study: query bias folded into a per-head key bias,\nvalue bias into the output bias; relative-position bias, shifted-window mask,\nsequence reduction and positional embeddings are kept as-is.", 7.6)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    out = Path("outputs/Seg5_Runs/figures")
    out.mkdir(parents=True, exist_ok=True)
    for key, spec in SPECS.items():
        print(draw_model(spec, out / f"workflow_{key}.png"))
    print(draw_tucker_attention(out / "tucker_attention_workflow.png"))


if __name__ == "__main__":
    main()
