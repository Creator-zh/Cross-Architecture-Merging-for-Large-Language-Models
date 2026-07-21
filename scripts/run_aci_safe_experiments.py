#!/usr/bin/env python
"""Run legacy FFN and circuit-attention experiments with compatible run names."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.aci.presets import ACI_SAFE_PRESETS  # noqa: E402
from scripts.run_aci_ablations import build_parser, run_experiments  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = __doc__
    return run_experiments(
        parser.parse_args(argv),
        ACI_SAFE_PRESETS,
        method="aci_safe_circuit_experiments",
        manifest_name="safe_launch_manifest.json",
        log_subdirectory="safe",
    )


if __name__ == "__main__":
    raise SystemExit(main())
