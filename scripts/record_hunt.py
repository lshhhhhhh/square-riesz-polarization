"""Overlap CUDA candidate generation with CPU continuous-domain evaluation.

The GPU never waits for SciPy polishing: after each population search, its
top candidates are written to disk and submitted to a process pool while the
next CUDA job starts.  Every GPU job and CPU evaluation is atomically persisted
so an interrupted hunt can resume without discarding completed work.

This program discovers and ranks candidates.  A reported hit is provisional
until its literal decimal coordinates pass both exact full-square certifiers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from square_riesz.evaluate import evaluate_continuous


BASELINES: dict[int, float] = {
    3: 7.56838964002969,
    4: 17.729,
    5: 22.063083016036777,
    6: 29.791,
    9: 57.38839577734211,
    13: 96.34710139024597,
}
_UPSTREAM_RESULTS = PROJECT_ROOT / "upstream" / "certificates" / "data" / "certified-results.csv"
if _UPSTREAM_RESULTS.exists():
    with _UPSTREAM_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_value = int(row["n"])
            if n_value <= 35:
                # To claim a genuine improvement, exceed the incumbent's
                # rigorous upper witness, not merely its rounded lower target.
                BASELINES[n_value] = float(row["rigorous_upper_witness"])

# Locally certified configurations supersede older public incumbents.  These
# are upper-witness values, so a provisional hit must improve the configuration
# itself rather than merely its rounded certified lower target.
BASELINES.update(
    {
        3: 7.56838964002969,
        5: 22.063083016036777,
        29: 282.85692528612702,
        30: 285.3456853298848,
        31: 305.29836691152415,
        32: 317.2038189278292,
        33: 330.5954794801076,
        34: 337.906235823038,
        35: 347.195722331293,
    }
)


def _load_incumbent(n: int, configuration_root: Path) -> list[list[float]]:
    local_path = PROJECT_ROOT / "data" / "candidates" / f"n{n:02d}_hunt_best.json"
    if local_path.exists():
        payload = json.loads(local_path.read_text(encoding="utf-8"))
        if int(payload.get("n", -1)) == n:
            coordinates = [
                [float(x), float(y)] for x, y in payload["coordinates"]
            ]
            if len(coordinates) == n:
                return coordinates
    path = configuration_root / f"n{n:02d}" / "coordinates.csv"
    if not path.exists():
        raise FileNotFoundError(f"no incumbent coordinates for N={n}: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        coordinates = [
            [float(row["x"]), float(row["y"])] for row in csv.DictReader(handle)
        ]
    if len(coordinates) != n:
        raise ValueError(f"incumbent N={n} has {len(coordinates)} coordinates")
    return coordinates


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _evaluate_candidate(task: dict[str, Any]) -> dict[str, Any]:
    """CPU worker entry point; intentionally imports no CUDA code."""

    started = perf_counter()
    coordinates = np.asarray(task["coordinates"], dtype=np.float64)
    evaluation = evaluate_continuous(
        coordinates,
        grid_size=int(task["evaluation_grid"]),
        low_seed_count=max(64, 10 * int(task["n"])),
        active_tolerance=2e-6,
    )
    minimum = float(evaluation.minimum)
    baseline = float(task["baseline"])
    return {
        **task,
        "continuous_evaluation": evaluation.to_dict(),
        "continuous_minimum": minimum,
        "baseline": baseline,
        "margin_over_baseline": minimum - baseline,
        "provisional_hit": minimum > baseline,
        "cpu_elapsed_seconds": perf_counter() - started,
        "finished_at": _utc_now(),
        "rigor": "numerical_only_pending_literal_decimal_exact_certificate",
    }


def _run_gpu_job(
    *,
    n: int,
    seed: int,
    batch: int,
    steps: int,
    grid_size: int,
    chunk: int,
    top_k: int,
    learning_rate: float,
    start_temperature: float,
    end_temperature: float,
    initialization: str,
    incumbent_coordinates: list[list[float]] | None,
    jitter: float,
    extra_test_points: list[list[float]] | None,
) -> dict[str, Any]:
    import torch

    from square_riesz.torch_engine import (
        optimize_population_softmin,
        random_configurations,
        unit_square_grid,
    )

    device = torch.device("cuda")
    if initialization == "random":
        initial = random_configurations(
            batch, n, seed=seed, device=device, dtype=torch.float32
        )
    else:
        if incumbent_coordinates is None:
            raise ValueError("warm initialization requires incumbent coordinates")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        incumbent = torch.tensor(
            incumbent_coordinates, device=device, dtype=torch.float32
        )
        scale = torch.linspace(
            0.05, 1.0, batch, device=device, dtype=torch.float32
        ).reshape(batch, 1, 1)
        noise = torch.randn(
            (batch, n, 2), generator=generator, device=device, dtype=torch.float32
        )
        initial = torch.clamp(incumbent[None, :, :] + jitter * scale * noise, 0.0, 1.0)
        initial[0] = incumbent
        if initialization == "mixed":
            random_count = max(1, batch // 8)
            initial[-random_count:] = torch.rand(
                (random_count, n, 2),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
    grid = unit_square_grid(grid_size, device=device, dtype=torch.float32)
    if extra_test_points:
        grid = torch.cat(
            (
                grid,
                torch.tensor(extra_test_points, device=device, dtype=torch.float32),
            ),
            dim=0,
        )
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    search = optimize_population_softmin(
        initial,
        grid,
        steps=steps,
        learning_rate=learning_rate,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        point_chunk_size=chunk,
        report_every=max(1, steps // 8),
    )
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    order = torch.argsort(search.sampled_minima, descending=True)
    indices = order[: min(top_k, len(order))].cpu().tolist()
    candidates = []
    for rank, index in enumerate(indices, start=1):
        candidates.append(
            {
                "rank": rank,
                "population_index": index,
                "sampled_minimum": float(search.sampled_minima[index].item()),
                "coordinates": search.configurations[index]
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist(),
            }
        )
    return {
        "schema_version": 1,
        "stage": "gpu_candidate_generation",
        "n": n,
        "seed": seed,
        "batch": batch,
        "steps": steps,
        "grid_size": grid_size,
        "chunk": chunk,
        "top_k": top_k,
        "learning_rate": learning_rate,
        "start_temperature": start_temperature,
        "end_temperature": end_temperature,
        "initialization": initialization,
        "jitter": jitter,
        "extra_test_point_count": len(extra_test_points or ()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "gpu_elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "history": list(search.history),
        "candidates": candidates,
        "finished_at": _utc_now(),
    }


def _job_id(n: int, seed: int) -> str:
    return f"n{n:02d}_seed{seed}"


def _evaluation_id(job_id: str, rank: int) -> str:
    return f"{job_id}_rank{rank:03d}"


def _parse_n_values(text: str) -> list[int]:
    result = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--n-values must be a nonempty unique CSV list")
    missing = [n for n in result if n not in BASELINES]
    if missing:
        raise argparse.ArgumentTypeError(f"no curated baseline for N={missing}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=_parse_n_values, default="4,5,6,9,13")
    parser.add_argument("--seeds-per-n", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=2026081900)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--evaluation-grid", type=int, default=129)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--end-temperature", type=float, default=0.015)
    parser.add_argument(
        "--initialization",
        choices=("random", "incumbent-jitter", "mixed"),
        default="random",
    )
    parser.add_argument("--jitter", type=float, default=0.025)
    parser.add_argument(
        "--incumbent-root",
        type=Path,
        default=PROJECT_ROOT / "upstream" / "certificates" / "data" / "configurations",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.seeds_per_n < 1 or args.batch < 1 or args.steps < 1 or args.top_k < 1:
        raise SystemExit("seed, batch, step, and top-k counts must be positive")
    if args.cpu_workers < 1:
        raise SystemExit("--cpu-workers must be positive")
    if args.jitter < 0:
        raise SystemExit("--jitter must be nonnegative")

    incumbents: dict[int, list[list[float]]] = {}
    incumbent_witnesses: dict[int, list[list[float]]] = {}
    if args.initialization != "random":
        incumbents = {
            n: _load_incumbent(n, args.incumbent_root) for n in args.n_values
        }
        for n, coordinates in incumbents.items():
            incumbent_evaluation = evaluate_continuous(
                np.asarray(coordinates, dtype=np.float64),
                grid_size=max(129, args.evaluation_grid),
                low_seed_count=max(96, 6 * n),
                active_tolerance=2e-6,
            )
            incumbent_witnesses[n] = [
                list(item.point) for item in incumbent_evaluation.local_minima
            ]

    # Prevent each SciPy worker from spawning its own full BLAS thread team.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    output_dir = args.output_dir.resolve()
    gpu_dir = output_dir / "gpu"
    evaluated_dir = output_dir / "evaluated"
    event_log = output_dir / "events.jsonl"
    best_path = output_dir / "best.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_or_resumed_at": _utc_now(),
        "n_values": args.n_values,
        "baselines": {str(n): BASELINES[n] for n in args.n_values},
        "seeds_per_n": args.seeds_per_n,
        "base_seed": args.base_seed,
        "batch": args.batch,
        "steps": args.steps,
        "grid_size": args.grid_size,
        "top_k": args.top_k,
        "evaluation_grid": args.evaluation_grid,
        "cpu_workers": args.cpu_workers,
        "initialization": args.initialization,
        "jitter": args.jitter,
        "incumbent_root": str(args.incumbent_root.resolve()),
        "incumbent_witness_counts": {
            str(n): len(points) for n, points in incumbent_witnesses.items()
        },
        "gpu": torch.cuda.get_device_name(0),
        "claim_boundary": "hits_are_numerical_until_both_exact_certifiers_pass",
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    best: dict[int, dict[str, Any]] = {}
    if best_path.exists():
        loaded_best = json.loads(best_path.read_text(encoding="utf-8"))
        best = {int(key): value for key, value in loaded_best.items()}

    futures: dict[Future[dict[str, Any]], tuple[Path, str]] = {}

    def submit_gpu_payload(
        pool: ProcessPoolExecutor, job_id: str, payload: dict[str, Any]
    ) -> None:
        for candidate in payload["candidates"]:
            evaluation_id = _evaluation_id(job_id, int(candidate["rank"]))
            path = evaluated_dir / f"{evaluation_id}.json"
            if path.exists():
                continue
            task = {
                "schema_version": 1,
                "stage": "cpu_continuous_evaluation",
                "evaluation_id": evaluation_id,
                "job_id": job_id,
                "n": int(payload["n"]),
                "seed": int(payload["seed"]),
                "rank": int(candidate["rank"]),
                "population_index": int(candidate["population_index"]),
                "sampled_minimum": float(candidate["sampled_minimum"]),
                "coordinates": candidate["coordinates"],
                "evaluation_grid": args.evaluation_grid,
                "baseline": BASELINES[int(payload["n"])],
                "gpu_result": str((gpu_dir / f"{job_id}.json").resolve()),
            }
            futures[pool.submit(_evaluate_candidate, task)] = (path, evaluation_id)

    def collect_finished(*, wait_for_all: bool = False) -> None:
        selected = list(as_completed(futures)) if wait_for_all else [f for f in futures if f.done()]
        for future in selected:
            path, evaluation_id = futures.pop(future)
            result = future.result()
            _atomic_json(path, result)
            n = int(result["n"])
            if n not in best or result["continuous_minimum"] > best[n]["continuous_minimum"]:
                best[n] = result
                _atomic_json(best_path, {str(key): value for key, value in sorted(best.items())})
            event = {
                "event": "cpu_evaluation_complete",
                "at": _utc_now(),
                "evaluation_id": evaluation_id,
                "n": n,
                "minimum": result["continuous_minimum"],
                "margin": result["margin_over_baseline"],
                "provisional_hit": result["provisional_hit"],
                "pending_cpu": len(futures),
            }
            _append_event(event_log, event)
            print(json.dumps(event), flush=True)

    # Round-robin N ordering gives early information for every target.
    jobs = []
    for replicate in range(args.seeds_per_n):
        for n_index, n in enumerate(args.n_values):
            seed = args.base_seed + replicate * 100 + n_index
            jobs.append((n, seed))

    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
        # Resume CPU work from any completed GPU payloads first.
        for path in sorted(gpu_dir.glob("*.json")):
            submit_gpu_payload(pool, path.stem, json.loads(path.read_text(encoding="utf-8")))

        for job_number, (n, seed) in enumerate(jobs, start=1):
            job_id = _job_id(n, seed)
            gpu_path = gpu_dir / f"{job_id}.json"
            if gpu_path.exists():
                collect_finished()
                continue
            event = {
                "event": "gpu_job_start",
                "at": _utc_now(),
                "job": job_number,
                "job_count": len(jobs),
                "job_id": job_id,
                "n": n,
                "seed": seed,
                "pending_cpu": len(futures),
            }
            _append_event(event_log, event)
            print(json.dumps(event), flush=True)
            payload = _run_gpu_job(
                n=n,
                seed=seed,
                batch=args.batch,
                steps=args.steps,
                grid_size=args.grid_size,
                chunk=args.chunk,
                top_k=args.top_k,
                learning_rate=args.learning_rate,
                start_temperature=args.start_temperature,
                end_temperature=args.end_temperature,
                initialization=args.initialization,
                incumbent_coordinates=incumbents.get(n),
                jitter=args.jitter,
                extra_test_points=incumbent_witnesses.get(n),
            )
            _atomic_json(gpu_path, payload)
            submit_gpu_payload(pool, job_id, payload)
            event = {
                "event": "gpu_job_complete",
                "at": _utc_now(),
                "job_id": job_id,
                "n": n,
                "gpu_seconds": payload["gpu_elapsed_seconds"],
                "sampled_best": payload["candidates"][0]["sampled_minimum"],
                "peak_allocated_mib": payload["peak_allocated_mib"],
                "pending_cpu": len(futures),
            }
            _append_event(event_log, event)
            print(json.dumps(event), flush=True)
            collect_finished()

        collect_finished(wait_for_all=True)

    summary = {
        "event": "hunt_complete",
        "at": _utc_now(),
        "elapsed_seconds": perf_counter() - started,
        "best": {str(key): value for key, value in sorted(best.items())},
    }
    _atomic_json(output_dir / "summary.json", summary)
    _append_event(event_log, {key: value for key, value in summary.items() if key != "best"})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
