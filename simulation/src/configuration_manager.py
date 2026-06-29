from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Set

from .process_model_builder import PetriNet


def require(d: dict, path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"Missing required key: {path}")
        cur = cur[key]
    return cur


def parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("start_time must include timezone offset, e.g. 2040-01-01T00:00:00+00:00")
    return dt


@dataclass(frozen=True)
class SimulationConfig:
    sim_input: dict
    net: PetriNet
    start_time: datetime
    seed: int
    case_count: int
    progress_every: int = 0
    heartbeat_seconds: float = 0.0
    rng: Random = field(compare=False, repr=False, hash=False, default_factory=Random)

    @property
    def branch_prob(self) -> Dict[str, Dict[str, float]]:
        decision_points = self.sim_input.get("decision_points")
        if isinstance(decision_points, dict):
            return decision_points.get("branch_probabilities", {})
        return {}

    @property
    def event_duration(self) -> Dict[str, Dict[str, float]]:
        return require(self.sim_input, "performance.event_duration_distribution")


    @property
    def resource_event_duration(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        return self.sim_input.get("resources", {}).get("resource_event_duration_distribution", {})

    @property
    def event_relationships(self) -> Dict[str, Dict[str, List[str]]]:
        return require(self.sim_input, "events").get("event_object_roles", {})

    @property
    def event_participants(self) -> Dict[str, List[str]]:
        rels = self.event_relationships
        configured = require(self.sim_input, "events").get("event_participants")
        if configured is not None:
            return configured
        return {activity: sorted(type_quals.keys()) for activity, type_quals in rels.items()}

    @property
    def resource_object_types(self) -> Set[str]:
        return set(require(self.sim_input, "objects.resource_object_types"))

    @property
    def reusable_object_types(self) -> Set[str]:
        return set(require(self.sim_input, "objects.reusable_object_types"))

    @property
    def event_role_to_resources(self) -> Dict[str, Dict[str, List[str]]]:
        return require(self.sim_input, "resources.event_resource_pools")

    @property
    def object_rel_targets(self) -> Dict[str, List[str]]:
        return require(self.sim_input, "relations.object_relation_targets")

    @property
    def event_object_iteration(self) -> List[Dict[str, Any]]:
        events = self.sim_input.get("events", {})
        raw = events.get("event_object_iteration", []) if isinstance(events, dict) else []
        return raw if isinstance(raw, list) else []

    @property
    def reusable_source_event_targets(self) -> List[Dict[str, Any]]:
        events = self.sim_input.get("events", {})
        raw = events.get("event_object_iteration", []) if isinstance(events, dict) else []
        reusable: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            for entry in raw:
                if str(entry.get("iteration_type") or "").upper() == "REUSABLE_SOURCE_EVENT_TARGET":
                    copied = dict(entry)
                    reusable.append(copied)
        return reusable

    @property
    def aggregation_config(self) -> Dict[str, Dict[str, Any]]:
        decision_points = self.sim_input.get("decision_points", {})
        if isinstance(decision_points, dict) and "sequential_aggregation" in decision_points:
            return decision_points.get("sequential_aggregation", {})
        return {}

    @property
    def deferred_generation_config(self) -> Dict[str, Dict[str, Any]]:
        return self.sim_input.get("performance", {}).get("deferred_arrival_distribution", {})

    @property
    def arrival_distribution(self) -> Dict[str, Dict[str, Any]]:
        return require(self.sim_input, "performance.arrival_distribution")

    def start_object_types(self) -> List[str]:
        gen = self.arrival_distribution
        flagged = [
            otype
            for otype, cfg in gen.items()
            if isinstance(cfg, dict) and "is_top_level" in cfg
        ]
        if flagged:
            out = sorted(
                otype
                for otype in flagged
                if bool(gen.get(otype, {}).get("is_top_level"))
            )
        else:
            out = sorted(gen.keys())
        if not out:
            raise ValueError("No top-level start object types found in performance.arrival_distribution.")
        return out

    def start_places(self) -> List[str]:
        gen = require(self.sim_input, "performance.arrival_distribution")
        out: List[str] = []
        for otype in self.start_object_types():
            configured = gen.get(otype, {}).get("start_place")
            if configured and configured in self.net.places:
                out.append(configured)
            else:
                inferred = f"{otype}_source"
                if inferred not in self.net.places:
                    raise ValueError(f"Inferred start_place '{inferred}' not found in net.places.")
                out.append(inferred)
        return out

    def exponential_lambda(self, obj_type: str) -> float:
        gen = self.arrival_distribution
        cfg = gen.get(obj_type)
        if cfg is None:
            raise ValueError(f"Missing required key: performance.arrival_distribution['{obj_type}']")
        dist = cfg.get("dist", "exponential")
        if dist != "exponential":
            raise ValueError(f"Only 'exponential' generation is supported. Got {dist!r} for {obj_type!r}.")
        lam = float(require(cfg, "lambda"))
        if lam <= 0:
            raise ValueError(f"arrival_distribution['{obj_type}']['lambda'] must be > 0.")
        return lam

    def sample_exponential_iat_seconds(self, lam: float) -> float:
        u = self.rng.random()
        if u <= 0.0:
            raise RuntimeError("Random draw u was 0.0; cannot compute exponential inter-arrival time.")
        return -math.log(u) / lam


class ConfigurationManager:
    REQUIRED_TOP_LEVEL = ("process_model", "objects", "relations", "resources", "events", "performance")

    @staticmethod
    def load_json(path: str | Path) -> dict:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def from_file(
        self,
        path: str | Path,
        *,
        start_time: str | datetime,
        seed: int,
        case_count: int,
        progress_every: int = 0,
        heartbeat_seconds: float = 0.0,
    ) -> SimulationConfig:
        sim_input = self.load_json(path)
        for key in self.REQUIRED_TOP_LEVEL:
            require(sim_input, key)

        parsed_start = parse_datetime(start_time) if isinstance(start_time, str) else start_time
        rng = Random(seed)
        net = PetriNet.from_sim_input(sim_input)
        return SimulationConfig(
            sim_input=sim_input,
            net=net,
            start_time=parsed_start,
            seed=seed,
            case_count=case_count,
            progress_every=max(0, progress_every),
            heartbeat_seconds=max(0.0, heartbeat_seconds),
            rng=rng,
        )
