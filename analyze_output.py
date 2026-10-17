"""Create a Markdown report from a SAR rank-sweep CSV and training log."""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "results_sar_rank_sweep.csv"
DEFAULT_LOG = ROOT / "outputs" / "out_ranksweep2.txt"
DEFAULT_REPORT = ROOT / "analysis_report.md"

METRICS = (
    "accuracy",
    "oil_spill_iou",
    "mean_iou_no_background",
    "precision",
    "recall",
)
REQUIRED_COLUMNS = (
    "tag",
    "r1",
    "r2",
    "r3",
    "n_seeds",
    "attn_params",
    "attention_reduction_pct",
    "compression_ratio",
    *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")),
)


@dataclass(frozen=True)
class ConfigResult:
    tag: str
    r1: int | None
    r2: int | None
    r3: int | None
    n_seeds: int
    attn_params: int
    attention_reduction_pct: float
    compression_ratio: float
    accuracy_mean: float
    accuracy_std: float
    oil_spill_iou_mean: float
    oil_spill_iou_std: float
    mean_iou_no_background_mean: float
    mean_iou_no_background_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float


@dataclass(frozen=True)
class LogInfo:
    device: str | None
    train_samples: int | None
    validation_samples: int | None
    classes: int | None
    configurations: dict[str, dict[str, float | int]]
    accuracy_tolerance: float | None
    iou_tolerance: float | None
    knee_threshold: float | None
    summary_lines: tuple[str, ...]


def _required_value(row: dict[str, str | None], column: str, line: int) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise ValueError(f"CSV row {line} has no value for {column!r}")
    return value.strip()


def _number(row: dict[str, str | None], column: str, line: int) -> float:
    value = _required_value(row, column, line)
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {line} has invalid number {value!r} in {column!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"CSV row {line} has non-finite value in {column!r}")
    return number


def _integer(row: dict[str, str | None], column: str, line: int) -> int:
    value = _required_value(row, column, line)
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {line} has invalid integer {value!r} in {column!r}"
        ) from exc
    return number


def read_results(csv_path: Path) -> list[ConfigResult]:
    """Read and validate the rank-sweep result rows."""
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        results: list[ConfigResult] = []
        seen_tags: set[str] = set()
        for line, row in enumerate(reader, start=2):
            tag = _required_value(row, "tag", line)
            if tag in seen_tags:
                raise ValueError(f"CSV contains duplicate configuration tag {tag!r}")
            seen_tags.add(tag)

            ranks: list[int | None] = []
            for column in ("r1", "r2", "r3"):
                value = row.get(column)
                ranks.append(
                    None if value is None or not value.strip() else _integer(row, column, line)
                )
            if tag == "MHA":
                if any(rank is not None for rank in ranks):
                    raise ValueError("MHA baseline must not have Tucker ranks in the CSV")
            elif any(rank is None for rank in ranks):
                raise ValueError(f"CSV row {line} is missing Tucker ranks for {tag!r}")

            values: dict[str, float] = {}
            for metric in METRICS:
                for statistic in ("mean", "std"):
                    column = f"{metric}_{statistic}"
                    values[column] = _number(row, column, line)
                    if statistic == "std" and values[column] < 0:
                        raise ValueError(f"CSV row {line} has negative standard deviation")

            n_seeds = _integer(row, "n_seeds", line)
            attn_params = _integer(row, "attn_params", line)
            if n_seeds < 1 or attn_params < 1:
                raise ValueError(f"CSV row {line} has invalid seed or parameter count")

            results.append(
                ConfigResult(
                    tag=tag,
                    r1=ranks[0],
                    r2=ranks[1],
                    r3=ranks[2],
                    n_seeds=n_seeds,
                    attn_params=attn_params,
                    attention_reduction_pct=_number(
                        row, "attention_reduction_pct", line
                    ),
                    compression_ratio=_number(row, "compression_ratio", line),
                    **values,
                )
            )

    if not results:
        raise ValueError("CSV contains no result rows")
    return results


def _decode_log(data: bytes, log_path: Path) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data:
        null_ratio = data.count(b"\x00") / len(data)
        if null_ratio > 0.1:
            encoding = "utf-16-be" if data[0] == 0 else "utf-16-le"
            return data.decode(encoding)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Cannot decode log as UTF-8 or UTF-16: {log_path}"
        ) from exc


def read_log(log_path: Path) -> LogInfo:
    """Extract run metadata and per-configuration summaries from the log."""
    text = _decode_log(log_path.read_bytes(), log_path)
    device_match = re.search(r"^\[device\]\s+training on:\s*(.+)$", text, re.MULTILINE)
    data_match = re.search(
        r"^\[data\]\s+(\d+)\s+train\s*/\s*(\d+)\s+validation samples;\s*(\d+) classes",
        text,
        re.MULTILINE,
    )
    config_pattern = re.compile(
        r"^\[(?P<tag>[^\]]+)\]\s+ranks=.*?params=(?P<params>[\d,]+)"
        r"\s+reduction=(?P<reduction>[\d.]+)%\s+accuracy=(?P<accuracy>[\d.]+)"
        r"\s+oil_iou=(?P<iou>[\d.]+)\s+mean_iou_no_bg=(?P<mean_iou>[\d.]+)\r?$",
        re.MULTILINE,
    )
    configurations: dict[str, dict[str, float | int]] = {}
    for match in config_pattern.finditer(text):
        configurations[match["tag"]] = {
            "params": int(match["params"].replace(",", "")),
            "reduction": float(match["reduction"]),
            "accuracy": float(match["accuracy"]),
            "oil_spill_iou": float(match["iou"]),
            "mean_iou_no_background": float(match["mean_iou"]),
        }

    tolerance_match = re.search(
        r"accuracy drop <=\s*(\d+(?:\.\d+)?),\s*"
        r"oil-spill IoU drop <=\s*(\d+(?:\.\d+)?)",
        text,
    )
    knee_match = re.search(
        r"(?:First tied-rank IoU drop >|No adjacent tied-rank oil-spill IoU drop exceeded)"
        r"\s*(\d+(?:\.\d+)?)",
        text,
    )
    summary_lines = tuple(
        line.strip()
        for line in text.splitlines()
        if line.startswith("Smallest model within baseline tolerances")
        or line.startswith("No Tucker configuration met both baseline metric tolerances")
        or line.startswith("First tied-rank IoU drop")
        or line.startswith("No adjacent tied-rank oil-spill IoU drop")
    )
    return LogInfo(
        device=device_match.group(1).strip() if device_match else None,
        train_samples=int(data_match[1]) if data_match else None,
        validation_samples=int(data_match[2]) if data_match else None,
        classes=int(data_match[3]) if data_match else None,
        configurations=configurations,
        accuracy_tolerance=float(tolerance_match[1]) if tolerance_match else None,
        iou_tolerance=float(tolerance_match[2]) if tolerance_match else None,
        knee_threshold=float(knee_match[1]) if knee_match else None,
        summary_lines=summary_lines,
    )


def smallest_acceptable(
    results: list[ConfigResult],
    baseline: ConfigResult,
    accuracy_tolerance: float,
    iou_tolerance: float,
) -> ConfigResult | None:
    eligible = [
        result
        for result in results
        if result.tag != "MHA"
        and result.attn_params < baseline.attn_params
        and result.accuracy_mean >= baseline.accuracy_mean - accuracy_tolerance
        and result.oil_spill_iou_mean
        >= baseline.oil_spill_iou_mean - iou_tolerance
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda result: (
            result.attn_params,
            -result.oil_spill_iou_mean,
            -result.accuracy_mean,
        ),
    )


def tied_rank_knee(
    results: list[ConfigResult], threshold: float
) -> tuple[ConfigResult, ConfigResult, float] | None:
    tied = [result for result in results if result.tag.startswith("tied_")]
    tied.sort(key=lambda result: result.attn_params, reverse=True)
    for larger, smaller in zip(tied, tied[1:]):
        drop = larger.oil_spill_iou_mean - smaller.oil_spill_iou_mean
        if drop > threshold:
            return larger, smaller, drop
    return None


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _metric_cell(mean: float, std: float) -> str:
    return f"{_percent(mean)} +/- {_percent(std)}"


def _ranks(result: ConfigResult) -> str:
    if result.r1 is None:
        return "MHA"
    return f"({result.r1}, {result.r2}, {result.r3})"


def _cross_check(results: list[ConfigResult], log: LogInfo) -> list[str]:
    if not log.configurations:
        return ["No per-configuration summary lines were found in the text log."]

    csv_by_tag = {result.tag: result for result in results}
    log_tags = set(log.configurations)
    csv_tags = set(csv_by_tag)
    issues: list[str] = []
    missing_from_log = sorted(csv_tags - log_tags)
    missing_from_csv = sorted(log_tags - csv_tags)
    if missing_from_log:
        issues.append(f"Missing from log: {', '.join(missing_from_log)}.")
    if missing_from_csv:
        issues.append(f"Missing from CSV: {', '.join(missing_from_csv)}.")

    for tag in sorted(csv_tags & log_tags):
        result = csv_by_tag[tag]
        logged = log.configurations[tag]
        checks = (
            ("attention parameters", result.attn_params, logged["params"], 0),
            (
                "attention reduction",
                result.attention_reduction_pct,
                logged["reduction"],
                0.051,
            ),
            ("accuracy", result.accuracy_mean, logged["accuracy"], 0.000051),
            (
                "oil-spill IoU",
                result.oil_spill_iou_mean,
                logged["oil_spill_iou"],
                0.000051,
            ),
            (
                "non-background mean IoU",
                result.mean_iou_no_background_mean,
                logged["mean_iou_no_background"],
                0.000051,
            ),
        )
        for name, csv_value, log_value, tolerance in checks:
            if abs(csv_value - log_value) > tolerance:
                issues.append(
                    f"{tag}: {name} differs (CSV {csv_value}, log {log_value})."
                )

    if not issues:
        return [
            f"All {len(csv_tags)} configuration summaries match "
            "the CSV (allowing for rounded log values)."
        ]
    return issues


def build_report(
    results: list[ConfigResult],
    log: LogInfo,
    csv_path: Path,
    log_path: Path,
    accuracy_tolerance: float,
    iou_tolerance: float,
    knee_threshold: float,
) -> str:
    baselines = [result for result in results if result.tag == "MHA"]
    if len(baselines) != 1:
        raise ValueError(f"Expected exactly one MHA baseline; found {len(baselines)}")
    baseline = baselines[0]
    best_iou = max(results, key=lambda result: result.oil_spill_iou_mean)
    best_accuracy = max(results, key=lambda result: result.accuracy_mean)
    acceptable = smallest_acceptable(
        results, baseline, accuracy_tolerance, iou_tolerance
    )
    knee = tied_rank_knee(results, knee_threshold)
    seeds = sorted({result.n_seeds for result in results})

    lines = [
        "# SAR Tucker Rank Sweep Analysis",
        "",
        f"- Results CSV: `{csv_path}`",
        f"- Training log: `{log_path}`",
        f"- Configurations: {len(results)}",
        f"- Seeds per configuration: {', '.join(map(str, seeds))}",
    ]
    if log.device:
        lines.append(f"- Training device: {log.device}")
    if log.train_samples is not None and log.validation_samples is not None:
        lines.append(
            f"- Data split: {log.train_samples:,} train / "
            f"{log.validation_samples:,} validation samples"
            + (f"; {log.classes} classes" if log.classes is not None else "")
        )

    lines.extend(
        [
            "",
            "## Key findings",
            "",
            (
                f"- MHA baseline: accuracy {_percent(baseline.accuracy_mean)}, "
                f"oil-spill IoU {_percent(baseline.oil_spill_iou_mean)}, "
                f"{baseline.attn_params:,} attention parameters."
            ),
            (
                f"- Highest oil-spill IoU: **{best_iou.tag}** "
                f"({_percent(best_iou.oil_spill_iou_mean)}; "
                f"{best_iou.attention_reduction_pct:.1f}% attention-parameter "
                "reduction)."
            ),
            (
                f"- Highest pixel accuracy: **{best_accuracy.tag}** "
                f"({_percent(best_accuracy.accuracy_mean)})."
            ),
            (
                f"- Baseline tolerances used: accuracy drop <= "
                f"{accuracy_tolerance:.4f}, oil-spill IoU drop <= "
                f"{iou_tolerance:.4f} (absolute metric fractions)."
            ),
        ]
    )
    if acceptable:
        lines.append(
            f"- Smallest configuration within both tolerances: **{acceptable.tag}** "
            f"ranks {_ranks(acceptable)}, {acceptable.attn_params:,} parameters, "
            f"{acceptable.attention_reduction_pct:.2f}% reduction "
            f"({acceptable.compression_ratio:.2f}x smaller)."
        )
    else:
        lines.append("- No Tucker configuration met both baseline tolerances.")
    if knee:
        larger, smaller, drop = knee
        lines.append(
            f"- First tied-rank IoU drop greater than {knee_threshold:.4f}: "
            f"{larger.tag} ({_percent(larger.oil_spill_iou_mean)}) to "
            f"{smaller.tag} ({_percent(smaller.oil_spill_iou_mean)}), "
            f"a {_percent(drop)} drop."
        )
    else:
        lines.append(
            f"- No adjacent tied-rank IoU drop exceeded {knee_threshold:.4f}."
        )

    lines.extend(
        [
            "",
            "## Configuration metrics",
            "",
            "| Configuration | Ranks | Attention params | Reduction | Compression | "
            "Accuracy (mean +/- std) | Oil-spill IoU (mean +/- std) | "
            "Mean IoU, no background | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(
        results, key=lambda row: row.oil_spill_iou_mean, reverse=True
    ):
        lines.append(
            f"| {result.tag} | {_ranks(result)} | {result.attn_params:,} | "
            f"{result.attention_reduction_pct:.2f}% | "
            f"{result.compression_ratio:.2f}x | "
            f"{_metric_cell(result.accuracy_mean, result.accuracy_std)} | "
            f"{_metric_cell(result.oil_spill_iou_mean, result.oil_spill_iou_std)} | "
            f"{_metric_cell(result.mean_iou_no_background_mean, result.mean_iou_no_background_std)} | "
            f"{_metric_cell(result.precision_mean, result.precision_std)} | "
            f"{_metric_cell(result.recall_mean, result.recall_std)} |"
        )

    lines.extend(["", "## CSV and log cross-check", ""])
    lines.extend(f"- {item}" for item in _cross_check(results, log))
    if log.summary_lines:
        lines.extend(["", "## Summary recorded in the text log", ""])
        lines.extend(f"- `{line}`" for line in log.summary_lines)
    lines.extend(
        [
            "",
            "Metric means and standard deviations come from the CSV. "
            "Tolerance and knee values are read from the log when available; "
            "otherwise the analyzer defaults are used.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SAR Tucker rank-sweep results and write a Markdown report."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="results CSV")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="training log")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown report path (default: analysis_report.md)",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.01,
        help="accuracy tolerance if not recorded in the log (default: 0.01)",
    )
    parser.add_argument(
        "--max-iou-drop",
        type=float,
        default=0.01,
        help="oil-spill IoU tolerance if not recorded in the log (default: 0.01)",
    )
    parser.add_argument(
        "--knee-thresh",
        type=float,
        default=0.01,
        help="tied-rank IoU knee threshold if not recorded in the log (default: 0.01)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Results CSV does not exist: {args.csv}")
    if not args.log.is_file():
        raise FileNotFoundError(f"Training log does not exist: {args.log}")
    if min(args.max_accuracy_drop, args.max_iou_drop, args.knee_thresh) < 0:
        raise ValueError("Metric tolerances and knee threshold must be non-negative")

    results = read_results(args.csv)
    log = read_log(args.log)
    report = build_report(
        results=results,
        log=log,
        csv_path=args.csv,
        log_path=args.log,
        accuracy_tolerance=(
            log.accuracy_tolerance
            if log.accuracy_tolerance is not None
            else args.max_accuracy_drop
        ),
        iou_tolerance=(
            log.iou_tolerance if log.iou_tolerance is not None else args.max_iou_drop
        ),
        knee_threshold=(
            log.knee_threshold
            if log.knee_threshold is not None
            else args.knee_thresh
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
