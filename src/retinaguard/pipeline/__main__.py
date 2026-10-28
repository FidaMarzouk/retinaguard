"""Materialise every asset and run every asset check: ``python -m retinaguard.pipeline``.

Uses the Python API rather than ``dagster asset materialize``: the CLI's selection parser needs a
newer ANTLR runtime than Hydra allows.
"""

import sys

import dagster as dg

from retinaguard.pipeline.assets import DataPaths, asset_checks, assets


def main() -> int:
    result = dg.materialize([*assets, *asset_checks], resources={"data": DataPaths()})
    failed = [c.check_name for c in result.get_asset_check_evaluations() if not c.passed]
    for name in failed:
        print(f"asset check failed: {name}", file=sys.stderr)
    return 0 if result.success and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
