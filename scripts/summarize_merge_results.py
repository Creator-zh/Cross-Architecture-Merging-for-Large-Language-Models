#!/usr/bin/env python
"""Summarize the five-variant medical/Thai/Malay merge-only evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.dfop.task_presets import TASK_PRESETS, get_task_preset  # noqa: E402


DEFAULT_VARIANTS = ("target", "source", "hot", "dfop_attn", "dfop_full")
METRIC_PRIORITY = {
    "xquad_th": ("f1,none", "f1", "exact_match,none", "exact_match"),
    "mgsm_direct_th": (
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
    ),
    "mgsm_native_cot_th": (
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
    ),
}
DEFAULT_METRIC_PRIORITY = (
    "acc,none",
    "acc_norm,none",
    "exact_match,none",
    "f1,none",
    "acc",
    "acc_norm",
    "exact_match",
    "f1",
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=_csv, default=["medical", "thai", "malay"])
    parser.add_argument("--variants", type=_csv, default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation_results" / "merge_only",
    )
    parser.add_argument("--scope", choices=("primary", "extended", "all"), default="primary")
    parser.add_argument("--output-prefix", default="merge_only")
    return parser


def _expected_benchmarks(task: str, scope: str) -> tuple[str, ...]:
    preset = get_task_preset(task)
    if scope == "primary":
        return preset.primary_eval_tasks
    if scope == "extended":
        return preset.extended_eval_tasks
    return preset.primary_eval_tasks + preset.extended_eval_tasks


def _latest_result(directory: Path) -> Path:
    metrics = directory / "metrics.json"
    if metrics.is_file():
        return metrics
    candidates = list(directory.rglob("results*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results*.json or metrics.json under {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _pick_metric(benchmark: str, values: dict) -> tuple[str, float]:
    priorities = METRIC_PRIORITY.get(benchmark, DEFAULT_METRIC_PRIORITY)
    for metric in priorities:
        value = values.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return metric, float(value)
    available = sorted(key for key, value in values.items() if isinstance(value, (int, float)))
    raise ValueError(f"No supported metric for {benchmark}; numeric metrics={available}")


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}"


def _markdown_summary(
    tasks: list[str],
    variants: list[str],
    macro: dict[tuple[str, str], float],
) -> str:
    lines = [
        "# Merge-only comparison",
        "",
        "Scores are percentages. Each domain macro is the unweighted mean of its configured benchmarks.",
        "Medical, Thai, and Malay are intentionally not averaged together.",
        "",
        "| Domain | " + " | ".join(variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]
    for task in tasks:
        cells = [_format_percent(macro.get((task, variant))) for variant in variants]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = set(args.tasks) - set(TASK_PRESETS)
    if unknown:
        raise SystemExit(f"Unknown tasks: {sorted(unknown)}")

    rows = []
    macro: dict[tuple[str, str], float] = {}
    failures = []
    for task in args.tasks:
        expected = _expected_benchmarks(task, args.scope)
        if not expected:
            failures.append(f"No {args.scope} benchmarks configured for {task}")
            continue
        for variant in args.variants:
            directory = args.eval_root / task / variant / args.scope
            try:
                result_file = _latest_result(directory)
                payload = json.loads(result_file.read_text(encoding="utf-8"))
                results = payload["results"]
                scores = []
                for benchmark in expected:
                    if benchmark not in results:
                        raise KeyError(f"Missing benchmark {benchmark}")
                    metric, score = _pick_metric(benchmark, results[benchmark])
                    scores.append(score)
                    rows.append(
                        {
                            "domain": task,
                            "variant": variant,
                            "benchmark": benchmark,
                            "metric": metric,
                            "score": score,
                            "score_percent": 100.0 * score,
                            "result_file": str(result_file),
                        }
                    )
                macro[(task, variant)] = sum(scores) / len(scores)
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"{task}:{variant}: {error}")

    args.eval_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.eval_root / f"{args.output_prefix}_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "domain",
                "variant",
                "benchmark",
                "metric",
                "score",
                "score_percent",
                "result_file",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.eval_root / f"{args.output_prefix}_summary.md"
    markdown_path.write_text(
        _markdown_summary(args.tasks, args.variants, macro), encoding="utf-8"
    )
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    if failures:
        print("Incomplete results:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
