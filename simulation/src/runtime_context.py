# runtime_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from random import Random
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple


from .object_state_store import CaseLifecycleState, Marking, Token, _Completion
from .ocel_logger import OCELRelationshipBuilder
from .process_model_builder import PetriNet
from .synchronization_controller import SynchronizationController
from .transition_executor import TransitionExecutor


@dataclass
class RuntimeContext:
    net: PetriNet
    log: object      # OCEL2Log
    graph: object    # Graph
    marking: Marking
    rng: Random
    start_time: datetime

    branch_prob: Dict[str, Dict[str, float]]                 # simulation.BRANCH_PROB
    event_duration: Dict[str, Dict[str, float]]              # simulation.EVENT_DURATION
    event_participants: Dict[str, List[str]]                 # ocel.EVENT_PARTICIPANTS (legacy; auto-optional for non-resources)
    event_relationships: Dict[str, Dict[str, List[str]]]     # ocel.EVENT_RELATIONSHIPS: activity -> object_type -> qualifiers
    resource_object_types: Set[str]                          # objects.RESOURCE_OBJECT_TYPES
    event_role_to_resources: Dict[str, Dict[str, List[str]]] # resources.EVENT_ROLE_TO_RESOURCES
    object_rel_targets: Dict[str, List[str]]                 # relations.OBJECT_REL_TARGETS
    aggregation_config: Dict[str, Dict[str, str]]            # simulation.SEQUENTIAL_AGGREGATION
    reusable_source_event_targets: List[Dict[str, object]] = field(default_factory=list)
    progress_every: int = 0                                  # Print progress every N completed cases; 0 disables it.
    heartbeat_seconds: float = 0.0                           # Print liveness updates every N wall-clock seconds; 0 disables it.

    _q: List[_Completion] = field(default_factory=list)
    _seq: int = 0
    _sync_progress: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)

    _case_oids_map: Dict[str, Set[str]] = field(default_factory=dict)
    _cid_to_root_oid: Dict[str, Optional[str]] = field(default_factory=dict, init=False)
    _oid_case_cache: Dict[str, Optional[str]] = field(default_factory=dict, init=False)
    _true_terminal_places: Set[str] = field(default_factory=set, init=False)
    _terminal_object_types: Set[str] = field(default_factory=set, init=False)
    _done_oids_by_case: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    _born_oids_by_case: Dict[str, Set[str]] = field(default_factory=dict, init=False)
    _completed_cases: Set[str] = field(default_factory=set, init=False)
    _cases_to_check: Set[str] = field(default_factory=set, init=False)
    case_state: CaseLifecycleState = field(default_factory=CaseLifecycleState, init=False)
    _last_progress_count: int = field(default=0, init=False)
    _progress_started_at: Optional[float] = field(default=None, init=False)
    _last_heartbeat_at: Optional[float] = field(default=None, init=False)

    _place_to_outgoing: Dict[str, List[str]] = field(default_factory=dict)
    _first_visible_labels: Dict[str, Set[str]] = field(default_factory=dict)
    _aggregation_required_by_tid: Dict[str, bool] = field(default_factory=dict, init=False)
    _children_by_type_cache: Dict[Tuple[str, Optional[str]], Tuple[str, ...]] = field(default_factory=dict, init=False)
    _related_by_type_cache: Dict[Tuple[str, str, str], Set[str]] = field(default_factory=dict, init=False)
    _iteration_targets_by_source: Dict[Tuple[str, str, str], Tuple[str, ...]] = field(default_factory=dict, init=False)
    _terminal_graph_oids_cache: Optional[Tuple[str, ...]] = field(default=None, init=False)
    _progress_terminal_graph_oids_cache: Optional[Tuple[str, ...]] = field(default=None, init=False)
    _iteration_distribution_cache: Dict[int, Tuple[List[int], List[float]]] = field(default_factory=dict, init=False)
    synchronization_controller: SynchronizationController = field(init=False)
    transition_executor: TransitionExecutor = field(init=False)

    def __post_init__(self) -> None:
        self._bind_case_state()
        self.synchronization_controller = SynchronizationController(self)
        self.transition_executor = TransitionExecutor(self)
        self._place_to_outgoing = {p: [] for p in self.net.places.keys()}
        for tid, pre_places in self.net.pre.items():
            for p in pre_places:
                self._place_to_outgoing[p].append(tid)
        for p in self._place_to_outgoing:
            self._place_to_outgoing[p].sort()

        # Identify true terminal places (no outgoing arcs to any transition)
        all_places = set(self.net.places.keys())
        all_sources = set()
        for tids in self.net.pre.values():
            all_sources.update(tids) # These are places used as inputs to transitions
        
        self._true_terminal_places = all_places - all_sources
        
        # Identify object types that HAVE at least one true terminal place
        self._terminal_object_types = {self.net.places[p].object_type for p in self._true_terminal_places}

        self._loop_exit_place_to_cfg = {}
        for tid, cfg in self.aggregation_config.items():
            ep = cfg.get("exit_place")
            if ep:
                self._loop_exit_place_to_cfg[ep] = (tid, cfg)

        self._precompute_first_visible_labels()
        self._precompute_iteration_targets()
        self._aggregation_required_by_tid = {
            tid: self._compute_aggregation_required_transition(tid)
            for tid in self.net.transitions
        }
        self._reusable_source_event_target_by_key = self._build_reusable_source_event_target_index()

    def _build_reusable_source_event_target_index(self) -> Dict[Tuple[str, str, str], Dict[str, object]]:
        index: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        for entry in self.reusable_source_event_targets or []:
            event_type = str(entry.get("event_type") or "")
            source_type = str(entry.get("source_object_type") or "")
            target_type = str(entry.get("target_object_type") or "")
            if event_type and source_type and target_type:
                index[(event_type, source_type, target_type)] = dict(entry)
        return index

    def _reusable_source_event_target_config(self, event_label: str, source_type: str, target_type: str) -> Optional[Dict[str, object]]:
        return getattr(self, "_reusable_source_event_target_by_key", {}).get((event_label, source_type, target_type))

    def _sample_reusable_source_event_target_count(self, cfg: Dict[str, object]) -> int:
        distribution = cfg.get("target_count_distribution") or {"1": 1.0}
        values: List[int] = []
        weights: List[float] = []
        if isinstance(distribution, dict):
            for raw_value, raw_weight in distribution.items():
                try:
                    value = int(str(raw_value).split(":")[-1])
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if value > 0 and weight > 0:
                    values.append(value)
                    weights.append(weight)
        if not values:
            return 1
        return max(1, int(self.rng.choices(values, weights=weights, k=1)[0]))

    def _bind_case_state(self) -> None:
        self._done_oids_by_case = self.case_state.done_oids_by_case
        self._born_oids_by_case = self.case_state.born_oids_by_case
        self._completed_cases = self.case_state.completed_cases
        self._cases_to_check = self.case_state.cases_to_check

    # ---------------- basic ----------------
    def _children_of_type(self, oid: str, object_type: Optional[str] = None) -> Tuple[str, ...]:
        key = (oid, object_type)
        cached = self._children_by_type_cache.get(key)
        if cached is None:
            cached = tuple(sorted(self.graph.children(oid, object_type)))
            self._children_by_type_cache[key] = cached
        return cached

    def _precompute_iteration_targets(self) -> None:
        self._iteration_targets_by_source = {}
        oid_type = getattr(self.graph, "oid_type", {})
        oid_case_roots = getattr(self.graph, "oid_case_roots", {})

        for transition_id, cfg in self.aggregation_config.items():
            source_type = cfg.get("parent_type")
            target_type = cfg.get("child_type")
            if not source_type or not target_type:
                continue

            mode = cfg.get("mode", "consume")
            for source_oid, actual_type in oid_type.items():
                if actual_type != source_type:
                    continue

                targets = set(self._directional_related_oids(source_oid, target_type))
                if not targets and mode == "produce":
                    for case_root in oid_case_roots.get(source_oid, set()):
                        targets.update(self.graph.get_all_related_oids(source_oid, target_type, case_root))

                if targets:
                    self._iteration_targets_by_source[(transition_id, source_oid, target_type)] = tuple(sorted(targets))

    def _iteration_targets_for(
        self,
        transition_id: str,
        source_oid: str,
        target_type: str,
    ) -> Tuple[str, ...]:
        return self._iteration_targets_by_source.get((transition_id, source_oid, target_type), ())

    def _is_wildcard_oid(self, oid: str) -> bool:
        # 그래프 레지스트리에 없는 oid면 wildcard
        if hasattr(self.graph, "oid_type") and oid not in self.graph.oid_type:
            return True

        # 만약 해당 객체가 스스로 케이스 루트라면 (예: 시작 객체가 Reusable 타입), 와일드카드로 보지 않음
        if hasattr(self.graph, "oid_case_roots"):
            roots = self.graph.oid_case_roots.get(oid, set())
            if len(roots) == 1 and next(iter(roots)) == oid:
                return False

        t = self.graph.type_of(oid)

        if hasattr(self.graph, "reusable_types") and t in self.graph.reusable_types:
            return True
        if t in self.resource_object_types:
            return True
        return False

    def _case_root_of_if_unique(self, oid: str) -> Optional[str]:
        if self._is_wildcard_oid(oid):
            return None
        roots = None
        if hasattr(self.graph, "oid_case_roots"):
            roots = self.graph.oid_case_roots.get(oid)
        if not roots or len(roots) != 1:
            return None
        return next(iter(roots))

    def _resolve_case_root_for_event(self, oids: List[str]) -> Optional[str]:
        # Use cache for single OID resolution
        if len(oids) == 1:
            oid = oids[0]
            if oid in self._oid_case_cache:
                return self._oid_case_cache[oid]

        found_roots = set()
        for oid in oids:
            if self._is_wildcard_oid(oid):
                continue
            r = self._case_root_of_if_unique(oid)
            if r:
                found_roots.add(r)
            else:
                # fall back to search
                if hasattr(self.graph, "related_oids_for_type_undirected"):
                    # find descendants that are case roots
                    q = deque([oid])
                    visited = {oid}
                    while q:
                        curr = q.popleft()
                        cr = self._case_root_of_if_unique(curr)
                        if cr:
                            found_roots.add(cr); break # optimization: one root is enough for our goals
                        
                        # expand
                        neighbors = set()
                        if hasattr(self.graph, "children"):
                            neighbors.update(self.graph.children(curr))
                        if hasattr(self.graph, "parents"):
                            neighbors.update(self.graph.parents(curr))

                        for neighbor in neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                        if found_roots: break
        
        res = sorted(list(found_roots))[0] if found_roots else None
        if len(oids) == 1:
            self._oid_case_cache[oids[0]] = res
        return res

    def _duration_seconds(self, label: str) -> float:
        m = self.event_duration.get(label)
        if not m:
            return 0.0

        mean = m.get("mean")
        std = m.get("std")
        if mean is None or std is None:
            return 0.0

        mean = float(mean)
        std = float(std)

        for _ in range(1000):
            x = self.rng.gauss(mean, std)
            if x > 0:
                return float(x)

        return 0.0

    def _sample_iteration_batch_size(self, cfg: Dict[str, object]) -> int:
        cache_key = id(cfg)
        cached = self._iteration_distribution_cache.get(cache_key)
        if cached is None:
            distribution = cfg.get("per_iteration_distribution") or {"1": 1.0}
            values: List[int] = []
            weights: List[float] = []
            if isinstance(distribution, dict):
                for raw_value, raw_weight in distribution.items():
                    try:
                        value = int(str(raw_value).split(":")[-1])
                        weight = float(raw_weight)
                    except (TypeError, ValueError):
                        continue
                    if value > 0 and weight > 0:
                        values.append(value)
                        weights.append(weight)
            cached = (values, weights)
            self._iteration_distribution_cache[cache_key] = cached
        values, weights = cached
        if not values:
            return 1
        return max(1, int(self.rng.choices(values, weights=weights, k=1)[0]))

    # ---------------- tau -> first visible ----------------
    def _precompute_first_visible_labels(self) -> None:
        self._first_visible_labels = {}
        for tid, tr in self.net.transitions.items():
            if tr.label is not None:
                self._first_visible_labels[tid] = {tr.label}
                continue

            seen_places: Set[str] = set()
            seen_trans: Set[str] = {tid}
            queue: List[str] = list(self.net.post.get(tid, set()))
            found: Set[str] = set()

            while queue:
                p = queue.pop(0)
                if p in seen_places:
                    continue
                seen_places.add(p)

                for nxt_tid in self._place_to_outgoing.get(p, []):
                    if nxt_tid in seen_trans:
                        continue
                    seen_trans.add(nxt_tid)

                    nxt_tr = self.net.transitions[nxt_tid]
                    if nxt_tr.label is not None:
                        found.add(nxt_tr.label)
                        continue

                    for p2 in self.net.post.get(nxt_tid, set()):
                        if p2 not in seen_places:
                            queue.append(p2)

            self._first_visible_labels[tid] = found

    def _first_visible_labels_for(self, tid: str) -> Set[str]:
        cached = self._first_visible_labels.get(tid)
        if cached is not None and len(cached) > 0:
            return set(cached)

        tr = self.net.transitions[tid]
        if tr.label is not None:
            self._first_visible_labels[tid] = {tr.label}
            return {tr.label}

        # on-demand BFS
        seen_places: Set[str] = set()
        seen_trans: Set[str] = {tid}
        queue: List[str] = list(self.net.post.get(tid, set()))
        found: Set[str] = set()

        while queue:
            p = queue.pop(0)
            if p in seen_places:
                continue
            seen_places.add(p)

            for nxt_tid in self._place_to_outgoing.get(p, []):
                if nxt_tid in seen_trans:
                    continue
                seen_trans.add(nxt_tid)

                nxt_tr = self.net.transitions[nxt_tid]
                if nxt_tr.label is not None:
                    found.add(nxt_tr.label)
                    continue

                for p2 in self.net.post.get(nxt_tid, set()):
                    if p2 not in seen_places:
                        queue.append(p2)

        self._first_visible_labels[tid] = found
        return set(found)

    # ---------------- helper: terminal-only tau candidate ----------------
    def _is_terminal_only_path_from_transition(self, tid: str) -> bool:
        """
        Returns True if from this transition you can reach a terminal place without encountering a visible label.
        Used to map 'no visible label' => virtual label 'Complete Case'.
        """
        terminal_places = getattr(self, "_terminal_places_set", None) or self._true_terminal_places
        if not terminal_places:
            return False

        seen_places: Set[str] = set()
        seen_trans: Set[str] = {tid}
        queue: List[str] = list(self.net.post.get(tid, set()))

        while queue:
            p = queue.pop(0)
            if p in seen_places:
                continue
            seen_places.add(p)

            if p in terminal_places:
                return True

            for nxt_tid in self._place_to_outgoing.get(p, []):
                if nxt_tid in seen_trans:
                    continue
                seen_trans.add(nxt_tid)

                nxt_tr = self.net.transitions[nxt_tid]
                if nxt_tr.label is not None:
                    return False

                for p2 in self.net.post.get(nxt_tid, set()):
                    if p2 not in seen_places:
                        queue.append(p2)

        return False

    def _try_select_aggregation_consumption(self, tid: str) -> Optional[List[Tuple[str, Token]]]:
        return self.synchronization_controller.try_select_aggregation_consumption(tid)

    def _compute_aggregation_required_transition(self, tid: str) -> bool:
        return self.synchronization_controller.is_aggregation_required_transition(tid)

    def _is_aggregation_required_transition(self, tid: str) -> bool:
        if tid in self._aggregation_required_by_tid:
            return self._aggregation_required_by_tid[tid]
        return self._compute_aggregation_required_transition(tid)

    # ---------------- case-consistent enabling/consumption ----------------
    def _is_compatible(self, tok: Token, root: str) -> bool:
        r = self._case_root_of_if_unique(tok.oid)
        return (r is None) or (r == root)

    def _is_related(self, oid1: str, oid2: str) -> bool:
        """
        Determines if two objects are semantically related using LCA isolation.
        Logic:
        1) Vertical: One is an ancestor of the other.
        2) Local Horizontal: Share a direct parent or direct child.
        3) Distant Horizontal: Share a common ancestor that is NOT just the case root,
           unless the case root is their direct/only possible common parent.
        """
        if oid1 == oid2: return True
        r1 = self._case_root_of_if_unique(oid1)
        r2 = self._case_root_of_if_unique(oid2)
        if (r1 is None) or (r1 != r2):
            return False
            
        # 1. Vertical Check
        anc1 = self.graph.get_all_ancestors(oid1)
        if oid2 in anc1: return True
        anc2 = self.graph.get_all_ancestors(oid2)
        if oid1 in anc2: return True
        
        # 2. Shared Neighbors (Direct)
        p1 = set(self.graph.parents(oid1))
        p2 = set(self.graph.parents(oid2))
        if p1 & p2: return True
        
        c1 = set(self.graph.children(oid1))
        c2 = set(self.graph.children(oid2))
        if c1 & c2: return True
        
        # 3. LCA Isolation (Distant)
        lcas = self.graph.get_lowest_common_ancestors(oid1, oid2)
        if not lcas: return False
        
        # If any LCA is NOT the case root, they are related via an intermediate structure.
        if any(anc_oid != r1 for anc_oid in lcas):
            return True
            
        # If the ONLY LCA is the case root:
        # We only allow this if at least one of them has NO other parents (i.e., it's a direct child of the root).
        # This allows P2P (PR -> Quotation) but blocks Logistics (CO -> TD -> Vehicle).
        if not p1 or not p2:
            return True
            
        return False

    def _directional_related_oids(self, anchor_oid: str, target_type: str) -> Set[str]:
        """Return directly connected objects of target_type following configured type directions."""
        try:
            anchor_type = self.graph.type_of(anchor_oid)
        except Exception:
            return set()

        related: Set[str] = set()
        if target_type in self.object_rel_targets.get(anchor_type, []):
            related.update(self.graph.children(anchor_oid, target_type))
        if anchor_type in self.object_rel_targets.get(target_type, []):
            related.update(self.graph.parents(anchor_oid, target_type))

        if not related:
            related.update(self.graph.children(anchor_oid, target_type))
            related.update(self.graph.parents(anchor_oid, target_type))
        return related

    def _get_related_of_type(self, anchor_oid: str, target_type: str, case_root: str) -> Set[str]:
        """Find objects by configured type-direction graph relation; case_root is only fallback context."""
        key = (anchor_oid, target_type, case_root)
        cached = self._related_by_type_cache.get(key)
        if cached is None:
            cached = set(self._directional_related_oids(anchor_oid, target_type))
            if not cached and case_root is not None:
                cached = set(self.graph.get_all_related_oids(anchor_oid, target_type, case_root))
            self._related_by_type_cache[key] = cached
        return set(cached)

    def _select_consumption_for_tid(self, tid: str) -> Optional[List[Tuple[str, Token]]]:
        return self.synchronization_controller.select_consumption(tid)


    def _complete_one(self) -> datetime:
        return self.transition_executor.complete_one()

    def _candidate_tids_from_marking(self) -> List[str]:
        return self.synchronization_controller.candidate_tids_from_marking()

    def _enabled_selections(self) -> Dict[str, List[Tuple[str, Token]]]:
        return self.synchronization_controller.enabled_selections()

    def _enabled_tids(self) -> List[str]:
        return self.synchronization_controller.enabled_tids()

    def _conflict_places(self, enabled_tids: List[str]) -> List[str]:
        return self.synchronization_controller.conflict_places(enabled_tids)

    def _seq_children_still_pending(
        self,
        *,
        parent_oid: str,
        case_root: Optional[str],
        child_type: str,
    ) -> bool:
        return self.synchronization_controller.seq_children_still_pending(
            parent_oid=parent_oid,
            case_root=case_root,
            child_type=child_type,
        )

    def _choose_seq_branch_override(
        self,
        *,
        place_id: str,
        candidates: List[str],
    ) -> Optional[str]:
        return self.synchronization_controller.choose_seq_branch_override(
            place_id=place_id,
            candidates=candidates,
        )

    # ---------------- BRANCH_PROB ----------------
    def _choose_transition(self, enabled_tids: List[str]) -> str:
        return self.synchronization_controller.choose_transition(enabled_tids)

    # ---------------- cross-type activation (UNDIRECTED multi-hop) ----------------
    def _collect_related_oids_for_output(
        self,
        *,
        consumed_oids: List[str],
        out_type: str,
    ) -> Tuple[List[str], bool]:
        if not consumed_oids:
            return [], False

        case_root = self._resolve_case_root_for_event(consumed_oids)
        if case_root is None:
            return [], False

        # 1. Undirected BFS search (preferred: relational proximity)
        all_related = set()
        for anchor in consumed_oids:
            found = self.graph.related_oids_for_type_undirected(
                anchor_oid=anchor,
                target_type=out_type,
                case_root=case_root,
                stop_expand_at_reusable=True,
                max_hops=10,
            )
            all_related.update(found)

        # 2. Fallback: Case-wide retrieval (ensures no orphans are missed)
        if not all_related:
            case_index = self.graph.case_objects_by_type.get(case_root, {})
            all_related = set(case_index.get(out_type, set()))

        # 3. Filter: already active or done (idempotent production)
        active_oids = self.marking.all_oids()
        done_oids = self._done_oids_by_case.get(case_root, set())
        
        final_oids = [oid for oid in sorted(all_related) 
                     if oid not in active_oids and oid not in done_oids]
        
        return final_oids, bool(all_related)

    # ---------------- OCEL relationships ----------------
    def _qualifier_for_event_object(self, event_label: str, oid: str, default: str) -> str:
        return OCELRelationshipBuilder(self).qualifier_for_event_object(event_label, oid, default)

    def _build_event_relationships(
        self,
        event_label: str,
        consumed_oids: List[str],
        produced_oids: List[str],
        reserved_resources: List[Tuple[str, str]],
        case_root: str,
    ) -> List[Dict[str, str]]:
        return OCELRelationshipBuilder(self).build_event_relationships(
            event_label=event_label,
            consumed_oids=consumed_oids,
            produced_oids=produced_oids,
            reserved_resources=reserved_resources,
            case_root=case_root,
        )

    def _update_object_relationships_from_event(self, rels: List[Dict[str, str]]) -> None:
        OCELRelationshipBuilder(self).update_object_relationships_from_event(rels)

    # ---------------- case lifecycle ----------------
    def _arrive_one_case(self, case_id: str, initial_tokens: List[Tuple[str, str]]) -> None:
        for sp, oid in initial_tokens:
            self.marking.add(sp, Token(oid=oid, case_id=case_id, last_label=None))
            self._register_born_oid(oid, case_id=case_id)
        self.case_state.request_check(case_id)

    def _case_roots_for_oid(self, oid: str, fallback_case_root: Optional[str]) -> Set[str]:
        roots = set(getattr(self.graph, "oid_case_roots", {}).get(oid, set()))
        if not roots and fallback_case_root is not None:
            roots.add(fallback_case_root)
        return roots

    def _register_born_oid(self, oid: str, *, case_id: Optional[str] = None, case_root: Optional[str] = None) -> None:
        if case_root is None and case_id is not None:
            case_root = self._cid_to_root_oid.get(case_id)
        for root in self._case_roots_for_oid(oid, case_root):
            self.case_state.mark_born(root, oid)

    def _mark_done_oid(self, oid: str, *, case_root: Optional[str] = None) -> None:
        for root in self._case_roots_for_oid(oid, case_root):
            self.case_state.mark_done(root, oid)

    def is_case_completed(self, cid: str) -> bool:
        """
        A case is completed if ALL objects belonging to this case ID
        have matched their lifecycles (reached sinks or finished).
        Uses _terminal_places_set which is initialized in run().
        """
        case_oids = self._case_oids_map.get(cid, set())
        if not case_oids: return False
        
        terminal_set = getattr(self, "_terminal_places_set", set())
        case_root = self._cid_to_root_oid.get(cid, cid)
        born_oids = self._born_oids_by_case.get(case_root, set())
        for oid in case_oids:
            if oid not in born_oids:
                return False
            p = self.marking.get_object_location(oid)
            if p and p not in terminal_set:
                # Still active in an intermediate place.
                return False
        
        return True

    def _is_terminal_location(self, place_id: Optional[str]) -> bool:
        if place_id is None:
            return False
        terminal_set = getattr(self, "_terminal_places_set", set())
        return place_id in terminal_set or str(place_id).endswith("_sink")

    def _terminal_graph_oids(self) -> Tuple[str, ...]:
        if self._terminal_graph_oids_cache is not None:
            return self._terminal_graph_oids_cache

        terminal_types = set(getattr(self, "_terminal_object_types", set()))
        dummy_oids = set(getattr(self.graph, "dummy_oids", set()))
        self._terminal_graph_oids_cache = tuple(
            sorted(
                oid
                for oid, otype in getattr(self.graph, "oid_type", {}).items()
                if otype in terminal_types
                and oid not in dummy_oids
                and otype not in self.resource_object_types
            )
        )
        return self._terminal_graph_oids_cache

    def _leaf_terminal_object_types(self) -> Set[str]:
        terminal_types = set(getattr(self, "_terminal_object_types", set()))
        if not terminal_types:
            return set()

        def reaches_terminal_descendant(source_type: str) -> bool:
            seen: Set[str] = set()
            stack = list(self.object_rel_targets.get(source_type, []))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                if current in terminal_types and current != source_type:
                    return True
                stack.extend(self.object_rel_targets.get(current, []))
            return False

        leaf_types = {
            object_type
            for object_type in terminal_types
            if not reaches_terminal_descendant(object_type)
        }
        return leaf_types or terminal_types

    def _progress_terminal_graph_oids(self) -> Tuple[str, ...]:
        if self._progress_terminal_graph_oids_cache is not None:
            return self._progress_terminal_graph_oids_cache

        progress_types = self._leaf_terminal_object_types()
        dummy_oids = set(getattr(self.graph, "dummy_oids", set()))
        self._progress_terminal_graph_oids_cache = tuple(
            sorted(
                oid
                for oid, otype in getattr(self.graph, "oid_type", {}).items()
                if otype in progress_types
                and oid not in dummy_oids
                and otype not in self.resource_object_types
            )
        )
        return self._progress_terminal_graph_oids_cache

    def _graph_oid_completion_counts(self, graph_oids: Tuple[str, ...]) -> Tuple[int, int]:
        if not graph_oids:
            return 0, 0

        done_any = set()
        for oids in self._done_oids_by_case.values():
            done_any.update(oids)

        completed = 0
        for oid in graph_oids:
            place_id = self.marking.get_object_location(oid)
            if self._is_terminal_location(place_id) or oid in done_any:
                completed += 1
        return completed, len(graph_oids)

    def _progress_terminal_graph_completion_counts(self) -> Tuple[int, int]:
        return self._graph_oid_completion_counts(self._progress_terminal_graph_oids())

    def _terminal_graph_completion_counts(self) -> Tuple[int, int]:
        return self._graph_oid_completion_counts(self._terminal_graph_oids())

    def _all_terminal_graph_objects_completed(self) -> bool:
        completed, total = self._terminal_graph_completion_counts()
        return total > 0 and completed >= total


    def all_arrived_cases_completed(self, arrived_roots: List[str]) -> bool:
        return bool(arrived_roots) and self._all_terminal_graph_objects_completed()

    def _refresh_completed_cases(self, arrived_roots: List[str]) -> int:
        self.case_state.cases_due_for_check(arrived_roots)
        completed_terminal, total_terminal = self._progress_terminal_graph_completion_counts()
        if total_terminal <= 0:
            return len(self._completed_cases)

        if completed_terminal < total_terminal:
            arrived_count = len(arrived_roots)
            if arrived_count <= 0:
                return 0
            progress_count = int(arrived_count * completed_terminal / total_terminal)
            return min(progress_count, arrived_count - 1)

        for cid in arrived_roots:
            self.case_state.mark_completed(cid)
        return len(self._completed_cases)

    def _maybe_print_progress(self, done_count: int, expected_case_count: int, arrived_count: int, now: datetime) -> None:
        if self.progress_every <= 0:
            return
        if done_count <= self._last_progress_count:
            return
        if done_count < expected_case_count and done_count - self._last_progress_count < self.progress_every:
            return

        elapsed = perf_counter() - self._progress_started_at if self._progress_started_at is not None else 0.0
        terminal_done, terminal_total = self._progress_terminal_graph_completion_counts()
        terminal_pct = (100.0 * terminal_done / terminal_total) if terminal_total else 0.0
        print(
            "[PROGRESS] "
            f"completed={done_count}/{expected_case_count} "
            f"terminal_progress={terminal_done}/{terminal_total}({terminal_pct:.1f}%) "
            f"arrived={arrived_count}/{expected_case_count} "
            f"sim_time={now.isoformat()} "
            f"queue={len(self._q)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        self._last_progress_count = done_count
        self._last_heartbeat_at = perf_counter()

    def _maybe_print_heartbeat(self, done_count: int, expected_case_count: int, arrived_count: int, now: datetime) -> None:
        if self.heartbeat_seconds <= 0:
            return
        if self._progress_started_at is None:
            return

        current = perf_counter()
        last = self._last_heartbeat_at if self._last_heartbeat_at is not None else self._progress_started_at
        if current - last < self.heartbeat_seconds:
            return

        elapsed = current - self._progress_started_at
        terminal_done, terminal_total = self._progress_terminal_graph_completion_counts()
        terminal_pct = (100.0 * terminal_done / terminal_total) if terminal_total else 0.0
        print(
            "[HEARTBEAT] "
            f"completed={done_count}/{expected_case_count} "
            f"terminal_progress={terminal_done}/{terminal_total}({terminal_pct:.1f}%) "
            f"arrived={arrived_count}/{expected_case_count} "
            f"sim_time={now.isoformat()} "
            f"queue={len(self._q)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        self._last_heartbeat_at = current
