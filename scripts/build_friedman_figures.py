"""Render guideline-compatible GIFs for the Friedman light-intensity board.

The live replacement-image dimensions were measured on 2026-08-19. Rendering
is offline and deterministic; this script performs no network access and sends
nothing.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

try:
    from scripts.build_friedman_submission import PROJECT_ROOT, RECORDS
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from build_friedman_submission import PROJECT_ROOT, RECORDS


DEFAULT_OUTPUT = PROJECT_ROOT / "submission" / "friedman_20260819" / "figures"
REFERENCE_SIZES = {
    3: (230, 230),
    5: (230, 230),
    29: (230, 230),
    30: (230, 225),
    31: (230, 227),
    32: (230, 228),
    33: (230, 229),
    34: (230, 230),
    35: (230, 231),
}


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _fraction(exact: dict[str, str]) -> Fraction:
    return Fraction(int(exact["numerator"]), int(exact["denominator"]))


def _minimum_markers(
    n: int, certificate: dict[str, object]
) -> list[tuple[float, float]]:
    if n == 3:
        # The frozen 19-digit literal coordinates have two exact darkest corners.
        return [(0.0, 0.0), (1.0, 0.0)]
    witness = certificate["upper_witness"]
    point = witness["point"]
    return [(float(_fraction(point[0])), float(_fraction(point[1])))]


def _render_one(
    n: int,
    coordinates: np.ndarray,
    minimum: float,
    markers: list[tuple[float, float]],
    output: Path,
) -> None:
    width, height = REFERENCE_SIZES[n]
    extent = 0.085
    resolution = 561
    axis = np.linspace(-extent, 1.0 + extent, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    field = np.zeros_like(grid_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        for source_x, source_y in coordinates:
            squared_distance = (grid_x - source_x) ** 2 + (grid_y - source_y) ** 2
            field += 1.0 / squared_distance

    contour_levels = minimum * np.array(
        [1.0025, 1.01, 1.03, 1.08, 1.18, 1.40, 1.80, 2.40]
    )
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    plot = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    plot.set_aspect("equal", adjustable="box")
    plot.set_xlim(-extent, 1.0 + extent)
    plot.set_ylim(-extent, 1.0 + extent)
    plot.axis("off")
    plot.contour(
        grid_x,
        grid_y,
        field,
        levels=contour_levels,
        colors="black",
        linewidths=0.55,
        antialiased=True,
    )
    plot.plot(
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        color="black",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=3,
    )
    plot.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=30,
        c="#ff0000",
        edgecolors="none",
        clip_on=False,
        zorder=5,
    )
    marker_array = np.asarray(markers, dtype=float)
    plot.scatter(
        marker_array[:, 0],
        marker_array[:, 1],
        s=30,
        c="#0000ff",
        edgecolors="none",
        clip_on=False,
        zorder=6,
    )
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(rgba[:, :, :3], mode="RGB")
    if image.size != (width, height):
        raise ValueError(f"renderer produced {image.size}, expected {(width, height)}")
    # Fix the four semantic colors explicitly. The remaining slots are a gray
    # ramp for antialiased contours; an adaptive palette can otherwise omit a
    # lone blue marker or slightly perturb saturated red.
    palette_values = [
        255, 255, 255,  # white background
        0, 0, 0,        # black contours and square
        255, 0, 0,      # red sources
        0, 0, 255,      # blue minima
    ]
    for index in range(252):
        # Avoid duplicating the explicit black and white entries above.
        gray = 1 + round(251 * index / 251)
        palette_values.extend((gray, gray, gray))
    rgb = np.asarray(image, dtype=np.uint16)
    luminance = np.rint(
        0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ).astype(np.int16)
    luminance = np.clip(luminance, 1, 252)
    indices = (luminance + 3).astype(np.uint8)
    indices[np.all(rgb >= 250, axis=2)] = 0
    indices[np.all(rgb <= 5, axis=2)] = 1
    red = (rgb[:, :, 0] >= 160) & (rgb[:, :, 0] >= 2 * rgb[:, :, 1]) & (
        rgb[:, :, 0] >= 2 * rgb[:, :, 2]
    )
    blue = (rgb[:, :, 2] >= 160) & (rgb[:, :, 2] >= 2 * rgb[:, :, 0]) & (
        rgb[:, :, 2] >= 2 * rgb[:, :, 1]
    )
    indices[red] = 2
    indices[blue] = 3
    palette = Image.fromarray(indices, mode="P")
    palette.putpalette(palette_values)
    palette.save(output, format="GIF", optimize=False)


def build(output_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for record in RECORDS:
        candidate = _load(project_root / record.candidate)
        certificate = _load(project_root / record.spectral)
        coordinates = np.asarray(candidate["coordinates"], dtype=float)
        upper = _fraction(certificate["upper_witness"]["value"])
        output = output_dir / f"{record.n}.gif"
        _render_one(
            record.n,
            coordinates,
            float(upper),
            _minimum_markers(record.n, certificate),
            output,
        )
        with Image.open(output) as image:
            if image.size != REFERENCE_SIZES[record.n] or image.format != "GIF":
                raise ValueError(f"invalid replacement image: {output}")
        rendered.append({"n": record.n, "file": output.name, "size": list(REFERENCE_SIZES[record.n])})
    return {"board_checked_date": "2026-08-19", "rendered": rendered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
