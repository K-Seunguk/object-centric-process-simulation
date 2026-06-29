from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .configuration_manager import ConfigurationManager
from .object_graph_generator import ObjectGraphGenerator
from .ocel_logger import sync_object_relationships_from_graph
from .simulation_engine import SimPySimulationEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case-count", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args()

    config = ConfigurationManager().from_file(
        args.sim_input,
        start_time=args.start_time,
        seed=args.seed,
        case_count=args.case_count,
        progress_every=args.progress_every,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    prepared = ObjectGraphGenerator(config).prepare()

    started_at = perf_counter()
    try:
        SimPySimulationEngine(config, prepared).run()
    finally:
        print(f"[TIME] Simulation elapsed: {perf_counter() - started_at:.3f} seconds")

    sync_object_relationships_from_graph(
        log=prepared.log,
        graph=prepared.graph,
        object_rel_targets=config.object_rel_targets,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.log.dump(str(out_path))


if __name__ == "__main__":
    main()
