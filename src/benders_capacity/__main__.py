from __future__ import annotations

import argparse
import json

from .experiment import comparison_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multi-cut Benders decomposition with the extensive-form MILP."
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = comparison_report(
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
