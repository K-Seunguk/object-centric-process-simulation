from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .object_state_store import Token


@dataclass
class SynchronizationController:
    """
    Coordinates graph-consistent token selection, aggregation-aware enabling,
    and branch choices for enabled transitions.
    """

    runtime: object

    def candidate_tids_from_marking(self) -> List[str]:
        rt = self.runtime
        candidate_tids: Set[str] = set()
        for place_id, bucket in rt.marking.tokens_by_place.items():
            if not bucket:
                continue
            candidate_tids.update(rt._place_to_outgoing.get(place_id, []))
        return sorted(candidate_tids)

    def enabled_selections(self) -> Dict[str, List[Tuple[str, Token]]]:
        rt = self.runtime
        enabled: Dict[str, List[Tuple[str, Token]]] = {}
        for tid in self.candidate_tids_from_marking():
            selection = self.select_consumption(tid)
            if selection is not None:
                enabled[tid] = selection
        return enabled

    def enabled_tids(self) -> List[str]:
        return sorted(self.enabled_selections().keys())

    def _transition_can_reach_activity(self, transition_id: str, activity: str) -> bool:
        rt = self.runtime
        cache = getattr(rt, "_can_reach_activity_cache", None)
        if cache is None:
            cache = {}
            setattr(rt, "_can_reach_activity_cache", cache)

        key = (transition_id, activity)
        if key in cache:
            return cache[key]

        transition = rt.net.transitions[transition_id]
        if transition.label == activity:
            cache[key] = True
            return True

        seen_places: Set[str] = set()
        seen_transitions: Set[str] = {transition_id}
        queue: List[str] = list(rt.net.post.get(transition_id, set()))

        while queue:
            place_id = queue.pop(0)
            if place_id in seen_places:
                continue
            seen_places.add(place_id)

            for next_tid in rt._place_to_outgoing.get(place_id, []):
                if next_tid in seen_transitions:
                    continue
                seen_transitions.add(next_tid)

                next_transition = rt.net.transitions[next_tid]
                if next_transition.label == activity:
                    cache[key] = True
                    return True

                for next_place_id in rt.net.post.get(next_tid, set()):
                    if next_place_id not in seen_places:
                        queue.append(next_place_id)

        cache[key] = False
        return False

    def _blocked_by_pending_iteration(
        self,
        transition_id: str,
        selection: List[Tuple[str, Token]],
    ) -> bool:
        rt = self.runtime
        firsts = rt._first_visible_labels_for(transition_id)
        if not firsts and rt.net.transitions[transition_id].label:
            firsts = {rt.net.transitions[transition_id].label}

        for seq_tid, cfg in rt.aggregation_config.items():
            parent_type = cfg.get("parent_type")
            child_type = cfg.get("child_type")
            activity = cfg.get("activity")
            if not parent_type or not child_type or not activity:
                continue
            if activity in firsts:
                continue
            if self._transition_can_reach_activity(transition_id, activity):
                continue

            for place_id, tok in selection:
                if rt.net.places[place_id].object_type != parent_type:
                    continue
                case_root = rt._cid_to_root_oid.get(tok.case_id) if tok.case_id else None
                if not case_root:
                    continue
                if rt._case_root_of_if_unique(tok.oid) != case_root:
                    continue
                if self.seq_children_still_pending(
                    parent_oid=tok.oid,
                    case_root=case_root,
                    child_type=child_type,
                    transition_id=seq_tid,
                ):
                    return True
        return False

    def select_consumption(self, transition_id: str) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        pre_places = sorted(rt.net.pre.get(transition_id, set()))
        if not pre_places:
            return None

        seq_cfg = rt.aggregation_config.get(transition_id)
        if seq_cfg:
            selected = self._select_sequential_consumption(transition_id, pre_places, seq_cfg)
            if selected is not None and not self._blocked_by_pending_iteration(transition_id, selected):
                return selected
            return None

        aggregation_consumed = rt._try_select_aggregation_consumption(transition_id)
        if aggregation_consumed is not None:
            if self._blocked_by_pending_iteration(transition_id, aggregation_consumed):
                return None
            return aggregation_consumed
        if rt._is_aggregation_required_transition(transition_id):
            return None

        selected = self._select_standard_consumption(pre_places)
        if selected is not None and self._blocked_by_pending_iteration(transition_id, selected):
            return None
        return selected

    def _select_sequential_consumption(
        self,
        transition_id: str,
        pre_places: List[str],
        seq_cfg: Dict[str, str],
    ) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        mode = seq_cfg.get("mode", "consume")
        parent_type = seq_cfg["parent_type"]
        child_type = seq_cfg["child_type"]

        parent_places = [p for p in pre_places if rt.net.places[p].object_type == parent_type]
        child_places = [p for p in pre_places if rt.net.places[p].object_type == child_type]

        if mode == "produce":
            if not parent_places:
                return None
            for parent_place in parent_places:
                for parent_tok in rt.marking.tokens(parent_place):
                    case_id = parent_tok.case_id
                    case_root = rt._cid_to_root_oid.get(case_id) if case_id else None
                    if not case_root:
                        continue

                    remaining = [
                        place_id
                        for place_id in pre_places
                        if place_id != parent_place and rt.net.places[place_id].object_type != child_type
                    ]
                    chosen = [(parent_place, parent_tok)]
                    for remaining_place in remaining:
                        match = next(
                            (
                                tok for tok in rt.marking.tokens(remaining_place)
                                if rt._is_compatible(tok, case_root)
                            ),
                            None,
                        )
                        if match is None:
                            chosen = None
                            break
                        chosen.append((remaining_place, match))
                    if chosen:
                        return chosen
            return None

        if not parent_places or not child_places:
            return None

        for parent_place in parent_places:
            for parent_tok in rt.marking.tokens(parent_place):
                case_id = parent_tok.case_id
                case_root = rt._cid_to_root_oid.get(case_id) if case_id else None
                if not case_root:
                    continue

                all_children = set(rt._iteration_targets_for(transition_id, parent_tok.oid, child_type))
                if not all_children:
                    continue

                done = rt._sync_progress.get((parent_tok.oid, transition_id, child_type), set())
                unprocessed = all_children - done
                if not unprocessed:
                    return [(parent_place, parent_tok)]

                available_child_oids: Set[str] = set()
                for child_place in child_places:
                    available_child_oids.update(unprocessed & rt.marking.oids(child_place))
                if not available_child_oids:
                    continue

                batch_size = rt._sample_iteration_batch_size(seq_cfg)
                selected_child_oids = sorted(available_child_oids)[:batch_size]
                chosen = [(parent_place, parent_tok)]
                used_child_places: Set[str] = set()
                for child_oid in selected_child_oids:
                    child_pair = None
                    for child_place in sorted(child_places):
                        child_tok = rt.marking.first_token_for_oid(child_place, child_oid)
                        if child_tok is not None:
                            child_pair = (child_place, child_tok)
                            break
                    if child_pair is not None:
                        chosen.append(child_pair)
                        used_child_places.add(child_pair[0])

                if len(chosen) <= 1:
                    continue

                remaining = [
                    place_id
                    for place_id in pre_places
                    if place_id != parent_place and place_id not in used_child_places
                ]
                for remaining_place in remaining:
                    if rt.net.places[remaining_place].object_type == child_type:
                        continue
                    match = next(
                        (
                            tok for tok in rt.marking.tokens(remaining_place)
                            if rt._is_compatible(tok, case_root)
                        ),
                        None,
                    )
                    if match is None:
                        chosen = None
                        break
                    chosen.append((remaining_place, match))
                if chosen:
                    return chosen
        return None

    def _select_standard_consumption(self, pre_places: List[str]) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        resource_tokens: Dict[str, List[Token]] = {}
        non_resource_places: List[str] = []
        for place_id in pre_places:
            place_type = rt.net.places[place_id].object_type
            if place_type in rt.resource_object_types:
                resource_tokens[place_id] = rt.marking.tokens(place_id)
            else:
                non_resource_places.append(place_id)

        if not non_resource_places:
            return None

        first_case_place = non_resource_places[0]
        seen_case_ids: Set[str] = set()
        for first_tok in rt.marking.tokens(first_case_place):
            case_id = first_tok.case_id
            if case_id is None or case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)

            chosen: List[Tuple[str, Token]] = []
            for place_id in pre_places:
                place_type = rt.net.places[place_id].object_type
                if place_type in rt.resource_object_types:
                    if not resource_tokens[place_id]:
                        chosen = []
                        break
                    chosen.append((place_id, resource_tokens[place_id][0]))
                else:
                    tok = rt.marking.first_token_for_case(place_id, case_id)
                    if tok is None:
                        chosen = []
                        break
                    chosen.append((place_id, tok))
            if chosen:
                return chosen

        return None

    def try_select_aggregation_consumption(self, tid: str) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        tr = rt.net.transitions[tid]
        if tr.label is None:
            return None

        if tid in rt.aggregation_config:
            return None

        pre_places = sorted(rt.net.pre.get(tid, set()))
        if not pre_places:
            return None

        post_places = sorted(rt.net.post.get(tid, set()))

        parent_produced = self._select_when_parent_is_produced(pre_places, post_places)
        if parent_produced is not None:
            return parent_produced

        if len(pre_places) >= 2:
            parent_present = self._select_when_parent_is_present(pre_places)
            if parent_present is not None:
                return parent_present

        return None

    def is_aggregation_required_transition(self, tid: str) -> bool:
        rt = self.runtime
        tr = rt.net.transitions[tid]
        if tr.label is None:
            return False

        pre_places = sorted(rt.net.pre.get(tid, set()))
        if not pre_places:
            return False

        if tid in rt.aggregation_config:
            return False

        pre_types = {
            rt.net.places[p].object_type
            for p in pre_places
            if rt.net.places[p].object_type
        }
        post_places = sorted(rt.net.post.get(tid, set()))
        post_types = {
            rt.net.places[p].object_type
            for p in post_places
            if rt.net.places[p].object_type
        }

        for parent_type in post_types:
            for child_type in pre_types:
                if parent_type != child_type and rt.graph.has_relation(parent_type, child_type):
                    return True

        for parent_type in pre_types:
            for child_type in pre_types:
                if parent_type != child_type and rt.graph.has_relation(parent_type, child_type):
                    return True

        return False

    def _select_when_parent_is_produced(
        self,
        pre_places: List[str],
        post_places: List[str],
    ) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        if len(pre_places) != 1:
            return None

        input_place = pre_places[0]
        child_type = rt.net.places[input_place].object_type
        if not child_type:
            return None

        out_types = {
            rt.net.places[place_id].object_type
            for place_id in post_places
            if rt.net.places[place_id].object_type
        }
        for parent_type in out_types:
            if parent_type == child_type or not rt.graph.has_relation(parent_type, child_type):
                continue

            tokens = rt.marking.tokens(input_place)
            if not tokens:
                continue

            if (parent_type, child_type) in getattr(rt.graph, "global_assignment_relations", set()):
                tok_by_oid = {tok.oid: tok for tok in tokens}
                done_any: Set[str] = set()
                for done_oids in rt._done_oids_by_case.values():
                    done_any.update(done_oids)

                parent_oids = [
                    oid
                    for oid, otype in sorted(getattr(rt.graph, "oid_type", {}).items())
                    if otype == parent_type
                    and oid not in getattr(rt.graph, "dummy_oids", set())
                    and rt.marking.get_object_location(oid) is None
                    and oid not in done_any
                ]
                for parent_oid in parent_oids:
                    all_children = set(rt._children_of_type(parent_oid, child_type))
                    if all_children and all_children.issubset(tok_by_oid.keys()):
                        return [(input_place, tok_by_oid[child_oid]) for child_oid in sorted(all_children)]
                continue

            by_case: Dict[Optional[str], List[Token]] = {}
            for tok in tokens:
                root = rt._case_root_of_if_unique(tok.oid)
                by_case.setdefault(root, []).append(tok)

            for root, case_tokens in by_case.items():
                if root is None:
                    continue

                tok_by_oid = {tok.oid: tok for tok in case_tokens}
                present_by_parent: Dict[str, Set[str]] = {}
                for child_oid in tok_by_oid:
                    parents = rt.graph.parents(child_oid, parent_type)
                    for parent_oid in parents:
                        if root in rt.graph.oid_case_roots.get(parent_oid, set()):
                            present_by_parent.setdefault(parent_oid, set()).add(child_oid)

                for parent_oid, present_children in present_by_parent.items():
                    all_children = set(rt._children_of_type(parent_oid, child_type))
                    if all_children and all_children.issubset(present_children):
                        return [(input_place, tok_by_oid[child_oid]) for child_oid in sorted(all_children)]

        return None

    def _select_when_parent_is_present(self, pre_places: List[str]) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        type_to_places: Dict[str, Set[str]] = {}
        for place_id in pre_places:
            object_type = rt.net.places[place_id].object_type
            if object_type:
                type_to_places.setdefault(object_type, set()).add(place_id)

        for parent_type, parent_places in type_to_places.items():
            for child_type, child_places in type_to_places.items():
                if parent_type == child_type:
                    continue
                if not rt.graph.has_relation(parent_type, child_type):
                    continue

                for parent_place in sorted(parent_places):
                    for parent_tok in rt.marking.tokens(parent_place):
                        selected = self._select_for_parent_token(
                            pre_places=pre_places,
                            parent_place=parent_place,
                            parent_tok=parent_tok,
                            child_type=child_type,
                            child_places=child_places,
                        )
                        if selected is not None:
                            return selected

        return None

    def _select_for_parent_token(
        self,
        *,
        pre_places: List[str],
        parent_place: str,
        parent_tok: Token,
        child_type: str,
        child_places: Set[str],
    ) -> Optional[List[Tuple[str, Token]]]:
        rt = self.runtime
        parent_oid = parent_tok.oid
        root = rt._case_root_of_if_unique(parent_oid)
        if root is None:
            return None

        all_children = set(rt._children_of_type(parent_oid, child_type))
        if not all_children:
            return None

        available_children: Dict[str, Tuple[str, Token]] = {}
        for child_place in sorted(child_places):
            for child_tok in rt.marking.tokens(child_place):
                if child_tok.oid in all_children:
                    available_children[child_tok.oid] = (child_place, child_tok)

        if not all_children.issubset(available_children.keys()):
            return None

        consumed: List[Tuple[str, Token]] = [(parent_place, parent_tok)]
        for child_oid in sorted(all_children):
            consumed.append(available_children[child_oid])

        remaining_places = [
            place_id
            for place_id in pre_places
            if place_id != parent_place and place_id not in child_places
        ]
        for place_id in remaining_places:
            match = next(
                (
                    tok
                    for tok in rt.marking.tokens(place_id)
                    if rt._is_compatible(tok, root)
                ),
                None,
            )
            if match is None:
                return None
            consumed.append((place_id, match))

        return consumed

    def conflict_places(self, enabled_tids: List[str]) -> List[str]:
        rt = self.runtime
        count: Dict[str, int] = {}
        for tid in enabled_tids:
            for place_id in rt.net.pre[tid]:
                count[place_id] = count.get(place_id, 0) + 1
        return sorted([place_id for place_id, seen_count in count.items() if seen_count >= 2])

    def seq_children_still_pending(
        self,
        *,
        parent_oid: str,
        case_root: Optional[str],
        child_type: str,
        transition_id: Optional[str] = None,
    ) -> bool:
        rt = self.runtime
        if transition_id is not None:
            total = set(rt._iteration_targets_for(transition_id, parent_oid, child_type))
        else:
            total = set(rt._children_of_type(parent_oid, child_type))
        if not total:
            return False

        born = rt._born_oids_by_case.get(case_root, set()) if case_root else set()
        if transition_id is not None:
            processed = set(rt._sync_progress.get((parent_oid, transition_id, child_type), set()))
        else:
            processed = set()
            for (progress_parent_oid, _progress_tid, progress_child_type), progress_oids in rt._sync_progress.items():
                if progress_parent_oid == parent_oid and progress_child_type == child_type:
                    processed.update(progress_oids)
        for child_oid in total:
            if child_oid in processed:
                continue
            if child_oid not in born:
                return True
            loc = rt.marking.get_object_location(child_oid)
            terminal_places = getattr(rt, "_terminal_places_set", set())
            if loc in terminal_places:
                continue
            if loc in rt.net.places and rt.net.places[loc].object_type == child_type:
                return True
        return False

    def choose_seq_branch_override(
        self,
        *,
        place_id: str,
        candidates: List[str],
    ) -> Optional[str]:
        rt = self.runtime
        tokens = rt.marking.tokens_by_place.get(place_id, [])
        if not tokens:
            return None

        tok = tokens[0]
        parent_type = rt.net.places[place_id].object_type
        if parent_type is None or tok.case_id is None:
            return None

        case_root = rt._cid_to_root_oid.get(tok.case_id)
        if not case_root:
            return None

        for seq_tid, cfg in rt.aggregation_config.items():
            if cfg.get("parent_type") != parent_type:
                continue

            activity = cfg.get("activity")
            if tok.last_label != activity:
                continue

            mode = cfg.get("mode", "consume")
            child_type = cfg["child_type"]
            if mode == "produce":
                expected = set(rt._iteration_targets_for(seq_tid, tok.oid, child_type))
                created = rt._sync_progress.get((tok.oid, seq_tid, child_type), set())
                child_locs = {
                    child_oid: rt.marking.get_object_location(child_oid)
                    for child_oid in expected
                }
                active_children = {
                    child_oid
                    for child_oid, loc in child_locs.items()
                    if loc in rt.net.places and rt.net.places[loc].object_type == child_type
                }
                if active_children and not created:
                    pending = False
                else:
                    pending = bool(expected - created)
            else:
                pending = self.seq_children_still_pending(
                    parent_oid=tok.oid,
                    case_root=case_root,
                    child_type=child_type,
                    transition_id=seq_tid,
                )

            loop_candidates: List[str] = []
            exit_candidates: List[str] = []
            for tid in candidates:
                firsts = rt._first_visible_labels_for(tid)
                if activity in firsts:
                    loop_candidates.append(tid)
                else:
                    exit_candidates.append(tid)

            if pending and loop_candidates:
                return rt.rng.choice(loop_candidates)
            if not pending and exit_candidates:
                return rt.rng.choice(exit_candidates)
            return None

        return None

    def choose_transition(self, enabled_tids: List[str]) -> str:
        rt = self.runtime
        if len(enabled_tids) == 1:
            return enabled_tids[0]

        conflict_places = self.conflict_places(enabled_tids)
        if not conflict_places:
            return rt.rng.choice(enabled_tids)

        place_id = rt.rng.choice(conflict_places)
        candidates = [tid for tid in enabled_tids if place_id in rt.net.pre[tid]]
        if len(candidates) <= 1:
            return candidates[0] if candidates else rt.rng.choice(enabled_tids)

        seq_override = self.choose_seq_branch_override(place_id=place_id, candidates=candidates)
        if seq_override is not None:
            return seq_override

        tok = rt.marking.tokens_by_place[place_id][0]
        cond = tok.last_label
        if cond is None:
            return rt.rng.choice(candidates)

        dist = rt.branch_prob.get(cond)
        if dist is None:
            return rt.rng.choice(candidates)

        weights: List[float] = []
        for tid in candidates:
            firsts = rt._first_visible_labels_for(tid)
            if not firsts:
                if rt._is_terminal_only_path_from_transition(tid):
                    firsts = {"Complete Case"}
                else:
                    return rt.rng.choice(candidates)

            weight = sum(float(dist.get(label, 0.0)) for label in firsts)
            weights.append(weight)

        if sum(weights) <= 0.0:
            return rt.rng.choice(candidates)

        return rt.rng.choices(candidates, weights=weights, k=1)[0]
