from __future__ import annotations

from datetime import datetime, timedelta
from heapq import heappush
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

import simpy

from .configuration_manager import SimulationConfig
from .object_graph_generator import PreparedSimulation
from .object_state_store import Token, _Completion
from .resource_scheduler import ResourceScheduler
from .runtime_context import RuntimeContext


class SimPyRuntimeContext(RuntimeContext):
    """
    Runtime context whose DES clock is driven by SimPy.

    Transition selection/completion semantics are provided by the shared runtime
    context. This class drives the clock with SimPy processes.
    """

    def attach_resource_scheduler(self, scheduler: ResourceScheduler) -> None:
        self.resource_scheduler = scheduler
        self._busy_until = scheduler.busy_until

    def _required_resource_types(self, event_label: str) -> List[str]:
        return self.resource_scheduler.required_resource_types(event_label)

    def _try_acquire_resources(
        self,
        now: datetime,
        event_label: str,
    ) -> Tuple[Optional[List[Tuple[str, str]]], Optional[datetime]]:
        return self.resource_scheduler.try_acquire(now, event_label)

    def _set_resources_busy(self, reserved: List[Tuple[str, str]], busy_until: datetime) -> None:
        self.resource_scheduler.set_busy(reserved, busy_until)
        for rtype, rid in reserved:
            self.log.create_object(rtype, oid=rid)
            self.graph.register_shared(rtype, rid)

    def run(
        self,
        *,
        arrivals: List[Tuple],
        deferred_arrivals: List[Tuple[datetime, str, List[Tuple[str, str]]]],
        case_oids_map: Dict[str, Set[str]],
        expected_case_count: int,
    ) -> Dict[str, int]:
        self._simpy_env = simpy.Environment()
        self._simpy_change = self._simpy_env.event()
        self._simpy_active = 0
        self._simpy_arrivals_done = 0
        self._simpy_deferred_done = 0
        self._simpy_arrival_total = len(arrivals)
        self._simpy_deferred_total = len(deferred_arrivals)

        self._case_oids_map = case_oids_map
        self._cid_to_root_oid = {}

        true_sinks = self.net.get_true_sink_places()
        self._terminal_places_set = set(true_sinks)
        self._terminal_object_types = {self.net.places[p].object_type for p in true_sinks}

        print(f"[INIT] Discovered Structural Sinks: {true_sinks}", flush=True)
        print(f"[INIT] Terminal Object Types: {sorted(self._terminal_object_types)}", flush=True)

        self.case_state.reset()
        self._bind_case_state()
        self._last_progress_count = 0
        self._progress_started_at = perf_counter()
        self._last_heartbeat_at = self._progress_started_at

        arrivals_sorted = sorted(arrivals, key=lambda x: x[0])
        deferred_sorted = sorted(deferred_arrivals, key=lambda x: x[0])
        self._simpy_arrived_roots: List[str] = []

        for entry in arrivals_sorted:
            self._simpy_env.process(self._arrival_process(entry))
        for entry in deferred_sorted:
            self._simpy_env.process(self._deferred_arrival_process(entry))

        self._simpy_env.process(self._scheduler_process(expected_case_count))
        self._simpy_env.run()
        return {root: 1 for root in self._simpy_arrived_roots}

    def _simpy_now_datetime(self) -> datetime:
        return self.start_time + timedelta(seconds=float(self._simpy_env.now))

    def _seconds_from_start(self, when: datetime) -> float:
        return max(0.0, (when - self.start_time).total_seconds())

    def _notify_simpy(self) -> None:
        if self._simpy_change.triggered:
            return
        self._simpy_change.succeed()
        self._simpy_change = self._simpy_env.event()

    def _arrival_process(self, entry: Tuple) -> object:
        when, cid, init_toks = entry[0], entry[1], entry[2]
        root_oid = entry[3] if len(entry) > 3 else None
        yield self._simpy_env.timeout(self._seconds_from_start(when))

        if root_oid is None:
            root_oid = self._resolve_case_root_for_event([oid for _, oid in init_toks])
        self._cid_to_root_oid[cid] = root_oid
        self._simpy_arrived_roots.append(cid)
        self._arrive_one_case(cid, init_toks)
        self._simpy_arrivals_done += 1
        self._notify_simpy()

    def _deferred_arrival_process(self, entry: Tuple[datetime, str, List[Tuple[str, str]]]) -> object:
        when, cid, toks = entry
        yield self._simpy_env.timeout(self._seconds_from_start(when))

        if cid in self._cid_to_root_oid:
            self._arrive_one_case(cid, toks)
        self._simpy_deferred_done += 1
        self._notify_simpy()

    def _scheduler_process(self, expected_case_count: int) -> object:
        while True:
            now = self._simpy_now_datetime()
            done_count = self._refresh_completed_cases(self._simpy_arrived_roots)
            self._maybe_print_progress(done_count, expected_case_count, len(self._simpy_arrived_roots), now)
            self._maybe_print_heartbeat(done_count, expected_case_count, len(self._simpy_arrived_roots), now)

            if self._simpy_is_finished(done_count):
                print(f"[SUCCESS] All {len(self._simpy_arrived_roots)} arrived cases completed.")
                return

            progressed = False
            while self._start_one_simpy(now):
                progressed = True
                now = self._simpy_now_datetime()

            if progressed:
                yield self._simpy_env.timeout(0)
                continue

            next_resource_time = self._next_resource_availability(now)
            if next_resource_time is not None:
                delay = max(0.0, (next_resource_time - now).total_seconds())
                yield self._simpy_env.timeout(delay)
                continue

            if self._simpy_no_future_work():
                self._raise_deadlock(now)

            change = self._simpy_change
            yield change

    def _start_one_simpy(self, now: datetime) -> bool:
        enabled_selections = self.synchronization_controller.enabled_selections()
        enabled = sorted(enabled_selections.keys())
        if not enabled:
            return False

        startable: List[str] = []
        reservations: Dict[str, List[Tuple[str, str]]] = {}
        for tid in enabled:
            label = self.net.transitions[tid].label
            if label is None:
                startable.append(tid)
                reservations[tid] = []
                continue
            reserved, _ = self._try_acquire_resources(now, label)
            if reserved is not None:
                startable.append(tid)
                reservations[tid] = reserved

        if not startable:
            return False

        tid = self.synchronization_controller.choose_transition(startable)
        transition = self.net.transitions[tid]
        selection = enabled_selections.get(tid)
        if selection is None:
            return False

        consumed = []
        for place, token in selection:
            try:
                consumed_token = self.marking.pop_matching(
                    place,
                    lambda candidate, oid=token.oid, case_id=token.case_id: (
                        candidate.oid == oid and candidate.case_id == case_id
                    ),
                )
            except RuntimeError as exc:
                available = [
                    (candidate.oid, candidate.case_id)
                    for candidate in self.marking.tokens(place)
                ]
                selected = [
                    (selected_place, selected_token.oid, selected_token.case_id)
                    for selected_place, selected_token in selection
                ]
                raise RuntimeError(
                    f"Failed to consume token for transition {tid} "
                    f"label={transition.label!r} place={place!r} "
                    f"target={(token.oid, token.case_id)!r} "
                    f"selection={selected!r} available_at_place={available!r}"
                ) from exc
            consumed.append((place, consumed_token))

        label = transition.label
        reserved = reservations.get(tid, [])
        if label and reserved is None:
            for place, token in consumed:
                self.marking.add(place, token)
            return False

        fallback_duration = self._duration_seconds(label) if label else 0.0
        if label and reserved:
            duration_seconds = self.resource_scheduler.sample_duration_seconds(label, reserved, fallback_duration)
        else:
            duration_seconds = fallback_duration
        done_time = now + timedelta(seconds=duration_seconds)
        if reserved:
            self._set_resources_busy(reserved, busy_until=done_time)

        self._seq += 1
        completion = _Completion(
            time=done_time,
            seq=self._seq,
            tid=tid,
            start_time=now,
            consumed=consumed,
            reserved_resources=reserved,
        )
        self._simpy_active += 1
        self._simpy_env.process(self._completion_process(completion, duration_seconds))
        return True

    def _completion_process(self, completion: _Completion, delay: float) -> object:
        yield self._simpy_env.timeout(max(0.0, delay))
        heappush(self._q, completion)
        self._complete_one()
        self._simpy_active -= 1
        self._notify_simpy()

    def _next_resource_availability(self, now: datetime) -> Optional[datetime]:
        next_avail: Optional[datetime] = None
        for tid in sorted(self.synchronization_controller.enabled_selections()):
            label = self.net.transitions[tid].label
            if not label:
                continue
            _, available_at = self._try_acquire_resources(now, label)
            if available_at and available_at > now:
                if next_avail is None or available_at < next_avail:
                    next_avail = available_at
        return next_avail

    def _simpy_is_finished(self, done_count: int) -> bool:
        return (
            self._simpy_arrivals_done >= self._simpy_arrival_total
            and self._simpy_deferred_done >= self._simpy_deferred_total
            and self._simpy_active == 0
            and done_count >= len(self._simpy_arrived_roots)
        )

    def _simpy_no_future_work(self) -> bool:
        return (
            self._simpy_arrivals_done >= self._simpy_arrival_total
            and self._simpy_deferred_done >= self._simpy_deferred_total
            and self._simpy_active == 0
        )

    def _raise_deadlock(self, now: datetime) -> None:
        print("\n--- DEADLOCK DIAGNOSTICS: STUCK TOKENS ---")
        for place, bucket in self.marking.tokens_by_place.items():
            if bucket:
                print(f"Place '{place}': {[token.oid for token in bucket]}")
        print("--- DEADLOCK DIAGNOSTICS: CASE COMPLETION ---")
        for cid in self._simpy_arrived_roots[:10]:
            case_root = self._cid_to_root_oid.get(cid, cid)
            case_oids = sorted(self._case_oids_map.get(cid, set()))
            born = self._born_oids_by_case.get(case_root, set())
            done = self._done_oids_by_case.get(case_root, set())
            missing_born = [oid for oid in case_oids if oid not in born]
            active = [
                (oid, self.graph.type_of(oid), self.marking.get_object_location(oid, case_id=cid))
                for oid in case_oids
                if self.marking.get_object_location(oid, case_id=cid)
            ]
            print(
                f"Case '{cid}' root='{case_root}': "
                f"case_oids={len(case_oids)} born={len(born)} done={len(done)} "
                f"missing_born={missing_born[:10]} active={active[:20]}"
            )
            for oid, otype, loc in active[:20]:
                if otype in {'Vehicle', 'Container'}:
                    parents = [(p, self.graph.type_of(p)) for p in self.graph.parents(oid)]
                    children = [(ch, self.graph.type_of(ch)) for ch in self.graph.children(oid)]
                    progress = {
                        str(k): sorted(v)[:10]
                        for k, v in self._sync_progress.items()
                        if oid in k or any(x == oid for x in v)
                    }
                    print(f"  oid={oid} type={otype} loc={loc} parents={parents[:10]} children={children[:10]} progress={progress}")
        raise RuntimeError(f"Simulation Deadlock at {now.isoformat()}")


class SimPySimulationEngine:
    def __init__(self, config: SimulationConfig, prepared: PreparedSimulation) -> None:
        self.config = config
        self.prepared = prepared

    def _build_aggregation_config_from_event_object_iteration(self) -> Dict[str, Dict[str, object]]:
        cfg = self.config
        net = cfg.net
        aggregation: Dict[str, Dict[str, object]] = dict(cfg.aggregation_config)

        mode_by_type = {"CREATE": "produce", "INCORPORATE": "consume", "LOOP": "consume"}
        for entry in cfg.event_object_iteration:
            iteration_type = str(entry.get("iteration_type") or "").upper()
            mode = mode_by_type.get(iteration_type)
            if mode is None:
                continue
            event_type = str(entry.get("event_type") or "")
            source_type = str(entry.get("source_object_type") or "")
            target_type = str(entry.get("target_object_type") or "")
            distribution = entry.get("target_count_distribution") or {"1": 1.0}
            if not event_type or not source_type or not target_type:
                continue

            for tid, transition in net.transitions.items():
                if transition.label != event_type:
                    continue
                source_post_places = [
                    place
                    for place in sorted(net.post.get(tid, set()))
                    if net.places[place].object_type == source_type
                ]
                source_pre_places = [
                    place
                    for place in sorted(net.pre.get(tid, set()))
                    if net.places[place].object_type == source_type
                ]
                loop_back_place = (source_pre_places or source_post_places or [None])[0]
                if loop_back_place is None:
                    continue
                aggregation[tid] = {
                    "mode": mode,
                    "parent_type": source_type,
                    "child_type": target_type,
                    "activity": event_type,
                    "loop_back_place": loop_back_place,
                    "per_iteration_distribution": distribution,
                }
        return aggregation

    def build_runtime(self) -> SimPyRuntimeContext:
        cfg = self.config
        prepared = self.prepared
        aggregation_config = self._build_aggregation_config_from_event_object_iteration()
        runtime = SimPyRuntimeContext(
            net=cfg.net,
            log=prepared.log,
            graph=prepared.graph,
            marking=prepared.initial_marking,
            rng=cfg.rng,
            start_time=cfg.start_time,
            branch_prob=cfg.branch_prob,
            event_duration=cfg.event_duration,
            event_participants=cfg.event_participants,
            event_relationships=cfg.event_relationships,
            resource_object_types=cfg.resource_object_types,
            event_role_to_resources=cfg.event_role_to_resources,
            object_rel_targets=cfg.object_rel_targets,
            aggregation_config=aggregation_config,
            reusable_source_event_targets=cfg.reusable_source_event_targets,
            progress_every=cfg.progress_every,
            heartbeat_seconds=cfg.heartbeat_seconds,
        )
        scheduler = ResourceScheduler(
            event_relationships=cfg.event_relationships,
            event_participants=cfg.event_participants,
            resource_object_types=cfg.resource_object_types,
            event_role_to_resources=cfg.event_role_to_resources,
            resource_event_duration=cfg.resource_event_duration,
            rng=cfg.rng,
        )
        runtime.attach_resource_scheduler(scheduler)
        return runtime

    def run(self) -> Dict[str, int]:
        runtime = self.build_runtime()
        return runtime.run(
            arrivals=self.prepared.arrivals,
            deferred_arrivals=self.prepared.deferred_arrivals,
            case_oids_map=self.prepared.case_oids_map,
            expected_case_count=self.config.case_count,
        )
