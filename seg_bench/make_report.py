"""Assemble outputs/Seg5_Runs/REPORT.md from the run summaries, CSVs and hand-written narrative files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seg_bench.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "Seg5_Runs"
ORDER = ["segformer_b2", "upernet_swin_t", "mask2former_swin_t", "swin_unetr", "transunet"]
PRETTY = {k: v.pretty for k, v in MODELS.items()}


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def load(key):
    d = OUT / key
    if not (d / "summary.json").is_file():
        return None
    s = json.loads((d / "summary.json").read_text())
    if "tucker_unfrozen" not in s:
        return None
    return dict(s=s, base=pd.read_csv(d / "baseline_epochs.csv"), unf=pd.read_csv(d / "tucker_unfrozen_epochs.csv"),
                dir=d, ffn=json.loads((d / "ffn_probe.json").read_text()) if (d / "ffn_probe.json").is_file() else None)


def narrative(name):
    p = OUT / "narrative" / f"{name}.md"
    return p.read_text(encoding="utf-8").strip() + "\n" if p.is_file() else ""


METRICS = [("f1", "F1"), ("mean_f1", "Mean F1"), ("oil_spill_iou", "Oil-spill IoU"), ("mean_iou", "Mean IoU"),
           ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall")]


def metrics_table(r):
    b, f, u = r["s"]["baseline"]["metrics"], r["s"]["tucker_frozen"]["metrics"], r["s"]["tucker_unfrozen"]["metrics"]
    lines = ["| Metric (%) | Baseline | Tucker frozen | Tucker unfrozen | Δ frozen | Δ unfrozen |",
             "|---|---|---|---|---|---|"]
    for k, n in METRICS:
        lines.append(f"| {n} | {pct(b[k])} | {pct(f[k])} | {pct(u[k])} | {100 * (f[k] - b[k]):+.2f} | {100 * (u[k] - b[k]):+.2f} |")
    return "\n".join(lines)


def late_table(r):
    k = 3
    bl, ul = r["base"].tail(k), r["unf"].tail(k)
    lines = [f"| Last-{k}-epoch mean (%) | Baseline | Tucker unfrozen | Δ |", "|---|---|---|---|"]
    for m, n in (("f1", "F1"), ("oil_spill_iou", "Oil-spill IoU"), ("accuracy", "Accuracy"),
                 ("precision", "Precision"), ("recall", "Recall")):
        lines.append(f"| {n} | {pct(bl[m].mean())} | {pct(ul[m].mean())} | {100 * (ul[m].mean() - bl[m].mean()):+.2f} |")
    sd_b, sd_u = r["base"].tail(5)["oil_spill_iou"].std(), r["unf"].tail(5)["oil_spill_iou"].std()
    lines.append(f"| Std of IoU, last 5 epochs | {pct(sd_b)} | {pct(sd_u)} | |")
    return "\n".join(lines)


def sweep_table(r):
    s = r["s"]
    base = s["baseline"]
    lines = ["| Config | Attention params | Attention compression | Total params (M) | Oil-spill IoU (%) | F1 (%) | GFLOPs | ms/img (bf16) |",
             "|---|---|---|---|---|---|---|---|",
             f"| Baseline (dense) | {s['sweep'][next(iter(s['sweep']))]['dense_attn_params'] / 1e6:.2f}M | 1.0x | "
             f"{base['profile']['params_m']:.2f} | {pct(base['metrics']['oil_spill_iou'])} | {pct(base['metrics']['f1'])} | "
             f"{base['profile']['gflops']:.1f} | {base['profile']['ms_per_img_bf16']:.2f} |"]
    for name, v in s["sweep"].items():
        star = " **(fine-tuned)**" if name == s["headline_level"] else ""
        lines.append(f"| Frozen {name}{star} | {v['tucker_attn_params'] / 1e6:.3f}M | {v['compression']:.1f}x | "
                     f"{v['profile']['params_m']:.2f} | {pct(v['metrics']['oil_spill_iou'])} | {pct(v['metrics']['f1'])} | "
                     f"{v['profile']['gflops']:.1f} | {v['profile']['ms_per_img_bf16']:.2f} |")
    return "\n".join(lines)


def resource_table(r):
    s = r["s"]
    cols = [("Baseline", s["baseline"]), ("Tucker frozen", dict(s["tucker_frozen"], train_time_s=0, train_peak_mem_mb=float("nan"))),
            ("Tucker unfrozen", s["tucker_unfrozen"])]
    rows = [("Parameters (M)", lambda c: f"{c['profile']['params_m']:.2f}"),
            ("GFLOPs / image (fwd, 224²)", lambda c: f"{c['profile']['gflops']:.1f}"),
            ("Latency bf16 (ms/img, batch 8)", lambda c: f"{c['profile']['ms_per_img_bf16']:.2f}"),
            ("Latency fp32 (ms/img, batch 8)", lambda c: f"{c['profile']['ms_per_img_fp32']:.2f}"),
            ("Inference peak memory, bf16 (MB)", lambda c: f"{c['profile']['infer_peak_mem_mb_bf16']:.0f}"),
            ("Training time (s)", lambda c: f"{c['train_time_s']:.0f}" if c["train_time_s"] else "-"),
            ("Training peak memory (MB)", lambda c: "-" if c["train_peak_mem_mb"] != c["train_peak_mem_mb"] else f"{c['train_peak_mem_mb']:.0f}")]
    lines = ["| Metric | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for n, fn in rows:
        lines.append(f"| {n} | " + " | ".join(fn(c) for _, c in cols) + " |")
    return "\n".join(lines)


def confusion_table(r):
    s = r["s"]
    lines = ["| Run | TN | FP | FN | TP |", "|---|---|---|---|---|"]
    for n, k in (("Baseline", "baseline"), ("Tucker frozen", "tucker_frozen"), ("Tucker unfrozen", "tucker_unfrozen")):
        (tn, fp), (fn, tp) = s[k]["metrics"]["confusion_matrix"]
        lines.append(f"| {n} | {tn:,} | {fp:,} | {fn:,} | {tp:,} |")
    return "\n".join(lines)


def ffn_table(r):
    if not r["ffn"]:
        return ""
    lines = ["| FFN scheme (post-hoc on baseline, no retraining) | FFN param ratio | Oil-spill IoU (%) | F1 (%) |", "|---|---|---|---|"]
    names = {"none": "none (reference)", "matrix": "per-layer low-rank (Tucker-2 / SVD)", "stacked": "cross-layer Tucker-3 (shared factors)"}
    for v in r["ffn"].values():
        lines.append(f"| {names[v['scheme']]} | {v['ffn_param_ratio']:.2f} | {pct(v['oil_spill_iou'])} | {pct(v['f1'])} |")
    return "\n".join(lines)


def summary_table(results):
    lines = ["| Model | Params (M) | Attn share | Attn compression | Baseline IoU | Frozen IoU | Unfrozen IoU | Δ unfrozen | Baseline F1 | Unfrozen F1 | ms/img base→Tucker (bf16) | GFLOPs base→Tucker |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    arch = json.loads((OUT / "architecture_analysis.json").read_text()) if (OUT / "architecture_analysis.json").is_file() else {}
    for key, r in results.items():
        s = r["s"]
        b, f, u = (s[k]["metrics"] for k in ("baseline", "tucker_frozen", "tucker_unfrozen"))
        share = f"{100 * arch[key]['categories']['attention'] / arch[key]['total']:.0f}%" if key in arch else "-"
        lines.append(
            f"| {PRETTY[key]} | {s['baseline']['profile']['params_m']:.1f} | {share} | {s['tucker_frozen']['compression']:.1f}x | "
            f"{pct(b['oil_spill_iou'])} | {pct(f['oil_spill_iou'])} | {pct(u['oil_spill_iou'])} | {100 * (u['oil_spill_iou'] - b['oil_spill_iou']):+.2f} | "
            f"{pct(b['f1'])} | {pct(u['f1'])} | {s['baseline']['profile']['ms_per_img_bf16']:.1f} → {s['tucker_unfrozen']['profile']['ms_per_img_bf16']:.1f} | "
            f"{s['baseline']['profile']['gflops']:.1f} → {s['tucker_unfrozen']['profile']['gflops']:.1f} |")
    return "\n".join(lines)


def arch_table():
    p = OUT / "architecture_analysis.json"
    if not p.is_file():
        return ""
    arch = json.loads(p.read_text())
    lines = ["| Model | Total (M) | Attention | FFN / MLP | Other encoder | Decoder / head | FFN SVD param-ratio @95% / @90% energy | Cross-layer Tucker-3 ratio @95% / @90% |",
             "|---|---|---|---|---|---|---|---|"]
    for key in ORDER:
        if key not in arch:
            continue
        a = arch[key]
        c = a["categories"]
        sh = lambda n: f"{c.get(n, 0) / 1e6:.2f}M ({100 * c.get(n, 0) / a['total']:.0f}%)"
        lines.append(f"| {a['pretty']} | {a['total'] / 1e6:.1f} | {sh('attention')} | {sh('ffn')} | {sh('encoder_other')} | {sh('decoder_head')} | "
                     f"{a['ffn_matrix_svd']['0.95']:.2f} / {a['ffn_matrix_svd']['0.9']:.2f} | "
                     f"{a['ffn_stacked_tucker']['0.95']:.2f} / {a['ffn_stacked_tucker']['0.9']:.2f} |")
    return "\n".join(lines)


def model_section(key, r):
    s, rel = r["s"], f"{key}"
    extra = ""
    if "extra_unfrozen" in s:
        lines = ["| Extra unfrozen run | Compression | Oil-spill IoU (%) | F1 (%) |", "|---|---|---|---|"]
        for level, v in s["extra_unfrozen"].items():
            lines.append(f"| {level} | {s['sweep'][level]['compression']:.1f}x | {pct(v['metrics']['oil_spill_iou'])} | {pct(v['metrics']['f1'])} |")
        extra = "\n**Additional fine-tuning at other Tucker levels**\n\n" + "\n".join(lines) + "\n"
    ffn = ffn_table(r)
    return f"""## {PRETTY[key]}

![workflow](figures/workflow_{key}.png)

{narrative(key + '_arch')}
### Results (best epoch by oil-spill IoU, single seed)

Headline Tucker level: **{s['headline_level']}** — attention {s['tucker_frozen']['compression']:.1f}× smaller
({s['tucker_frozen']['dense_attn_params'] / 1e6:.2f}M → {s['tucker_frozen']['tucker_attn_params'] / 1e6:.2f}M parameters, {s['tucker_frozen']['n_layers']} attention blocks).

{metrics_table(r)}

{late_table(r)}

![curves]({rel}/comparison_curves.png)

**Rank sweep (Tucker frozen = compress + inference only)**

{sweep_table(r)}
{extra}
**Resources**

{resource_table(r)}

**Confusion matrix at best epoch (pixels)**

{confusion_table(r)}

{('**FFN probe (what happens if the feed-forward weights are compressed instead)**' + chr(10) + chr(10) + ffn) if ffn else ''}

{narrative(key)}
"""


def main():
    results = {k: r for k in ORDER if (r := load(k)) is not None}
    head = narrative("00_intro")
    body = "\n".join(model_section(k, r) for k, r in results.items())
    report = f"""{head}
## Cross-model summary

{summary_table(results)}

{narrative('01_cross_model')}
## Architecture inventory: where the parameters are

{arch_table()}

{narrative('02_ffn')}
![tucker attention](figures/tucker_attention_workflow.png)

{body}
{narrative('99_conclusion')}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("report written", len(report))


if __name__ == "__main__":
    main()
