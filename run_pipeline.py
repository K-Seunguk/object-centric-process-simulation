from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

from simulation_input_builder import DATASETS, build_simulation_input
from simulation.src.configuration_manager import ConfigurationManager
from simulation.src.object_graph_generator import ObjectGraphGenerator
from simulation.src.ocel_logger import sync_object_relationships_from_graph
from simulation.src.simulation_engine import SimPySimulationEngine


BASE_DIR = Path(__file__).resolve().parent


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _datasets(selected: str) -> Iterable[str]:
    return DATASETS if selected == "all" else (selected,)


def build_input(dataset: str, *, data_dir: Path, input_dir: Path, noise: float) -> Path:
    print(f"[ANALYZE] {dataset}")
    sim_input = build_simulation_input(dataset, data_dir=data_dir, noise=noise)
    input_path = input_dir / f"simulation_input_{dataset}.json"
    _write_json(input_path, sim_input)
    print(f"[INPUT] {input_path}")
    return input_path


def run_simulation(
    dataset: str,
    *,
    input_path: Path,
    output_dir: Path,
    start_time: str,
    seed: int,
    case_count: int,
    progress_every: int,
    heartbeat_seconds: float,
) -> Path:
    output_path = output_dir / f"simulated_{dataset}.json"
    config = ConfigurationManager().from_file(
        input_path,
        start_time=start_time,
        seed=seed,
        case_count=case_count,
        progress_every=progress_every,
        heartbeat_seconds=heartbeat_seconds,
    )
    prepared = ObjectGraphGenerator(config).prepare()

    started_at = perf_counter()
    try:
        SimPySimulationEngine(config, prepared).run()
    finally:
        print(f"[TIME] {dataset} simulation elapsed: {perf_counter() - started_at:.3f}s")

    sync_object_relationships_from_graph(
        log=prepared.log,
        graph=prepared.graph,
        object_rel_targets=config.object_rel_targets,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.log.dump(str(output_path))
    print(f"[OUTPUT] {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full OCEL-to-simulation pipeline without intermediate analysis artifacts."
    )
    parser.add_argument("dataset", nargs="?", default="all", help="om, p2p, logistics, or all")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "simulation" / "input")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "simulation" / "output")
    parser.add_argument("--start-time", default="2040-01-01T00:00:00+00:00")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-count", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument(
        "--reuse-input",
        action="store_true",
        help="Skip analysis when simulation/input/simulation_input_<dataset>.json already exists.",
    )
    args = parser.parse_args()

    for dataset in _datasets(args.dataset):
        input_path = args.input_dir / f"simulation_input_{dataset}.json"
        if args.reuse_input and input_path.exists():
            print(f"[INPUT] Reusing {input_path}")
        else:
            input_path = build_input(
                dataset,
                data_dir=args.data_dir,
                input_dir=args.input_dir,
                noise=args.noise,
            )
        run_simulation(
            dataset,
            input_path=input_path,
            output_dir=args.output_dir,
            start_time=args.start_time,
            seed=args.seed,
            case_count=args.case_count,
            progress_every=args.progress_every,
            heartbeat_seconds=args.heartbeat_seconds,
        )


if __name__ == "__main__":
    main()
