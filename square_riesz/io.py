"""Input/output helpers for finite-decimal source configurations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .potential import as_sources


def load_coordinates_csv(path: str | Path) -> NDArray[np.float64]:
    """Load a CSV containing named ``x`` and ``y`` columns."""

    coordinate_path = Path(path)
    with coordinate_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "x" not in rows[0] or "y" not in rows[0]:
        raise ValueError(f"{coordinate_path} must contain x and y columns")
    return as_sources([[float(row["x"]), float(row["y"])] for row in rows])
