#!/usr/bin/env python
"""Evaluate legacy FFN and circuit-attention variants against target."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.aci.presets import ACI_SAFE_PRESETS  # noqa: E402
from scripts.evaluate_aci_ablations import (  # noqa: E402
    build_parser,
    evaluate_experiments,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(
        eval_root=REPOSITORY_ROOT / "evaluation_results" / "aci_safe"
    )
    return evaluate_experiments(
        parser.parse_args(argv),
        ACI_SAFE_PRESETS,
        method="aci_safe_circuit_evaluation",
        manifest_name="safe_manifest.json",
        csv_name="aci_safe_scores.csv",
        markdown_name="aci_safe_summary.md",
        title="ACI legacy FFN and circuit-attention vs target",
    )


if __name__ == "__main__":
    raise SystemExit(main())
