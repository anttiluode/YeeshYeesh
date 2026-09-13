from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from yeesh.experiment import ExperimentConfig, run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the boundary-programmed dynamics V0 experiment."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/v0_canonical.json"),
        help="Path for the JSON receipt.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=128,
        help="Number of matched random-location and random-order controls.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = replace(ExperimentConfig(), random_repeats=args.random_repeats)
    receipt = run_experiment(config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    for name in ("wave", "diffusion"):
        result = receipt[name]
        held = result["heldout"]
        print(f"{name}: optimized schedule {result['optimized_schedule']}")
        print(
            "  held-out target energy "
            f"free={held['free']['target_energy']:.8g} "
            f"optimized={held['optimized']['target_energy']:.8g} "
            f"fixed={held['fixed']['target_energy']:.8g}"
        )
        print(
            "  held-out selectivity "
            f"free={held['free']['selectivity']:.6f} "
            f"optimized={held['optimized']['selectivity']:.6f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
