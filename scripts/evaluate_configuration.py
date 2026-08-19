"""Numerically enumerate low points of a finite-decimal configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from square_riesz.evaluate import evaluate_continuous
from square_riesz.io import load_coordinates_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("--grid-size", type=int, default=129)
    parser.add_argument("--low-seeds", type=int, default=64)
    args = parser.parse_args()

    coordinates = load_coordinates_csv(args.coordinates)
    result = evaluate_continuous(
        coordinates, grid_size=args.grid_size, low_seed_count=args.low_seeds
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
