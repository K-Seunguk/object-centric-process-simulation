from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple

from .configuration_manager import SimulationConfig
from .object_state_store import Marking, Token
from .ocel_logger import OCEL2Log
from .object_graph_store import Graph
from .id_factory import IdFactory


Arrival = Tuple[datetime, str, List[Tuple[str, str]], str]
DeferredArrival = Tuple[datetime, str, List[Tuple[str, str]]]


@dataclass
class PreparedSimulation:
    log: OCEL2Log
    id_factory: IdFactory
    graph: Graph
    initial_marking: Marking
    arrivals: List[Arrival]
    deferred_arrivals: List[DeferredArrival]
    case_oids_map: Dict[str, Set[str]]


class ObjectGraphGenerator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def _root_type_for_reusable_start(self, reusable_type: str, net_object_types: Set[str]) -> Optional[str]:
        candidates = [
            target_type
            for target_type in self.config.object_rel_targets.get(reusable_type, [])
            if target_type in net_object_types
            and target_type not in self.config.reusable_object_types
            and target_type not in self.config.resource_object_types
        ]
        if not candidates:
            return None
        return sorted(candidates)[0]

    def _reusable_start_oid(
        self,
        *,
        graph: Graph,
        log: OCEL2Log,
        graph_case_root: str,
        root_oid: str,
        reusable_type: str,
    ) -> str:
        attached = graph.children(root_oid, reusable_type)
        if attached:
            return sorted(attached)[0]

        pool = graph.id_pools.get(reusable_type, [])
        if not pool:
            raise ValueError(f"Missing reusable object pool for start type '{reusable_type}'.")
        reusable_oid = self.config.rng.choice(pool)
        log.create_object(reusable_type, oid=reusable_oid)
        graph.register_existing(graph_case_root, reusable_type, reusable_oid)
        graph.add_edge(graph_case_root, root_oid, reusable_oid)
        return reusable_oid

    def prepare(self) -> PreparedSimulation:
        cfg = self.config
        sim_input = cfg.sim_input
        net = cfg.net

        log = OCEL2Log()
        id_factory = IdFactory(
            width=5,
            rng=cfg.rng,
            id_pools=sim_input.get("objects", {}).get("id_pools", {}),
        )
        graph = Graph(sim_input=sim_input, log=log, rng=cfg.rng, idf=id_factory)
        initial_marking = Marking()
        source_places = net.get_start_places_for_types()

        for otype, oids in id_factory.id_pools.items():
            start_place = source_places.get(otype)
            for oid in oids:
                if oid not in graph.oid_type:
                    log.create_object(otype, oid=oid)
                    graph.register_existing(case_root=None, otype=otype, oid=oid)
                if otype in cfg.resource_object_types and start_place:
                    initial_marking.add(start_place, Token(oid=oid, last_label=None))

        start_object_types = cfg.start_object_types()
        start_places = cfg.start_places()
        start_place_by_type = dict(zip(start_object_types, start_places))
        lam_by_type = {otype: cfg.exponential_lambda(otype) for otype in start_object_types}
        lam_total = sum(lam_by_type.values())
        types_sorted = sorted(start_object_types)
        net_object_types = {p.object_type for p in net.places.values() if p.object_type is not None}
        lifecycle_events = sim_input.get("objects", {}).get("object_lifecycle_events", {})
        arrival_distribution = cfg.arrival_distribution
        top_level_arrival_types = set(start_object_types)
        has_top_level_flags = any(
            isinstance(arrival_cfg, dict) and "is_top_level" in arrival_cfg
            for arrival_cfg in arrival_distribution.values()
        )
        if has_top_level_flags:
            candidate_non_top_initiate_types = {
                otype
                for otype, arrival_cfg in arrival_distribution.items()
                if isinstance(arrival_cfg, dict)
                and arrival_cfg.get("is_top_level") is False
            }
        else:
            candidate_non_top_initiate_types = {
                otype
                for otype, roles in lifecycle_events.items()
                if roles.get("INITIATE") and otype not in top_level_arrival_types
            }
        non_top_initiate_types = {
            otype
            for otype in candidate_non_top_initiate_types
            if otype not in cfg.resource_object_types
            and otype not in cfg.reusable_object_types
            and otype in net_object_types
        }

        arrivals: List[Arrival] = []
        deferred_arrivals: List[DeferredArrival] = []
        case_oids_map: Dict[str, Set[str]] = {}

        current_time = cfg.start_time
        build_started_at = perf_counter()
        for index in range(cfg.case_count):
            case_id = f"case_{index + 1:05d}"
            current_time += timedelta(seconds=cfg.sample_exponential_iat_seconds(lam_total))

            weights = [lam_by_type[otype] for otype in types_sorted]
            chosen_type = cfg.rng.choices(types_sorted, weights=weights, k=1)[0]
            arrival_type = chosen_type
            root_type = chosen_type

            if chosen_type in cfg.reusable_object_types:
                root_type = self._root_type_for_reusable_start(chosen_type, net_object_types)
                if root_type is None:
                    raise ValueError(
                        f"Start object type '{chosen_type}' is reusable, but no non-reusable child root type "
                        "was found in relations.object_relation_targets."
                    )

            root_oid = id_factory.next_object_id(root_type)
            log.create_object(root_type, oid=root_oid)

            graph_case_root = root_oid
            graph.build_case(root_type=root_type, root_oid=root_oid, case_root_oid=graph_case_root)
            all_case_objects = graph.get_real_case_objects_by_type(graph_case_root)

            root_start_place = start_place_by_type.get(arrival_type) or f"{arrival_type}_source"
            arrival_oid = root_oid
            if arrival_type in cfg.reusable_object_types:
                arrival_oid = self._reusable_start_oid(
                    graph=graph,
                    log=log,
                    graph_case_root=graph_case_root,
                    root_oid=root_oid,
                    reusable_type=arrival_type,
                )
            arrivals.append((current_time, case_id, [(root_start_place, arrival_oid)], graph_case_root))

            for otype in sorted(non_top_initiate_types):
                start_place = source_places.get(otype) or f"{otype}_source"
                if start_place not in net.places:
                    continue
                oids = sorted(all_case_objects.get(otype, []))
                if not oids:
                    continue
                arrival_cfg: Dict[str, Any] = arrival_distribution.get(otype, {})
                lam = float(arrival_cfg.get("lambda", 0.0) or 0.0) if isinstance(arrival_cfg, dict) else 0.0
                non_top_arrival_time = current_time
                for oid in oids:
                    if lam > 0:
                        non_top_arrival_time += timedelta(seconds=cfg.sample_exponential_iat_seconds(lam))
                    deferred_arrivals.append((non_top_arrival_time, case_id, [(start_place, oid)]))

            deferred_time = current_time
            for otype, deferred_cfg in cfg.deferred_generation_config.items():
                if otype == chosen_type:
                    continue
                oids = all_case_objects.get(otype, [])
                if not oids:
                    continue
                start_place = (
                    deferred_cfg.get("start_place")
                    or start_place_by_type.get(otype)
                    or source_places.get(otype)
                )
                lam = float(deferred_cfg.get("lambda", 0.0) or 0.0)
                if not start_place or lam <= 0:
                    continue
                for oid in oids:
                    deferred_time += timedelta(seconds=cfg.sample_exponential_iat_seconds(lam))
                    deferred_arrivals.append((deferred_time, case_id, [(start_place, oid)]))

            total_case_oids: Set[str] = set()
            for otype, oids in all_case_objects.items():
                if (
                    otype not in cfg.resource_object_types
                    and otype not in cfg.reusable_object_types
                    and otype in net_object_types
                ):
                    total_case_oids.update(oids)
            case_oids_map[case_id] = total_case_oids

            built_count = index + 1
            if cfg.progress_every > 0 and (
                built_count == cfg.case_count or built_count % cfg.progress_every == 0
            ):
                elapsed = perf_counter() - build_started_at
                print(f"[BUILD] prepared={built_count}/{cfg.case_count} elapsed={elapsed:.1f}s", flush=True)

        graph.apply_global_assignment_relations()
        global_assignment_source_types = {src for src, _tgt in graph.global_assignment_relations}

        case_oids_map = {}
        for case_id, _arrival_time, case_root in [
            (case_id, arrival_time, case_root)
            for arrival_time, case_id, _tokens, case_root in arrivals
        ]:
            total_case_oids = set()
            for otype, oids in graph.get_real_case_objects_by_type(case_root).items():
                if (
                    otype not in cfg.resource_object_types
                    and otype not in cfg.reusable_object_types
                    and otype in net_object_types
                    and otype not in global_assignment_source_types
                ):
                    total_case_oids.update(oids)
            case_oids_map[case_id] = total_case_oids

        return PreparedSimulation(
            log=log,
            id_factory=id_factory,
            graph=graph,
            initial_marking=initial_marking,
            arrivals=arrivals,
            deferred_arrivals=deferred_arrivals,
            case_oids_map=case_oids_map,
        )
