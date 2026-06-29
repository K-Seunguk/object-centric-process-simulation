from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from heapq import heappop
from typing import List, Optional, Set, Tuple

from .object_state_store import Token, _Completion


@dataclass(frozen=True)
class CompletionContext:
    completion: _Completion
    transition_id: str
    consumed: List[Tuple[str, Token]]
    consumed_oids: List[str]
    case_id: Optional[str]
    case_root: Optional[str]
    post_places: List[str]
    produced_label: Optional[str]

    @property
    def completion_time(self) -> datetime:
        return self.completion.time

    @property
    def start_time(self) -> datetime:
        return self.completion.start_time

    @property
    def reserved_resources(self) -> List[Tuple[str, str]]:
        return self.completion.reserved_resources


class TransitionExecutor:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    @staticmethod
    def case_id_from_consumed(consumed: List[Tuple[str, Token]], oid: str) -> Optional[str]:
        return next((tok.case_id for _, tok in consumed if tok.oid == oid), None)

    def add_output_token(
        self,
        *,
        place: str,
        oid: str,
        case_id: Optional[str],
        last_label: Optional[str],
        case_root: Optional[str],
        final_produced_oids: Set[str],
    ) -> None:
        rt = self.runtime
        rt.marking.add(place, Token(oid=oid, case_id=case_id, last_label=last_label))
        rt._register_born_oid(oid, case_id=case_id, case_root=case_root)
        final_produced_oids.add(oid)

    def add_output_pair_once(
        self,
        *,
        place: str,
        oid: str,
        case_id: Optional[str],
        last_label: Optional[str],
        case_root: Optional[str],
        produced_pairs: Set[Tuple[str, str]],
        final_produced_oids: Set[str],
    ) -> bool:
        key = (oid, place)
        if key in produced_pairs:
            return False
        self.add_output_token(
            place=place,
            oid=oid,
            case_id=case_id,
            last_label=last_label,
            case_root=case_root,
            final_produced_oids=final_produced_oids,
        )
        produced_pairs.add(key)
        return True

    def mark_unproduced_consumed_tokens(
        self,
        *,
        consumed: List[Tuple[str, Token]],
        final_produced_oids: Set[str],
        produced_label: Optional[str],
        case_root: Optional[str],
    ) -> None:
        rt = self.runtime
        for place_id, tok in consumed:
            oid = tok.oid
            if oid in final_produced_oids:
                continue
            if rt.net.places[place_id].object_type in rt.resource_object_types:
                rt.marking.add(place_id, Token(oid=oid, last_label=produced_label))
                continue
            rt._mark_done_oid(oid, case_root=case_root)

    def emit_visible_event(
        self,
        *,
        produced_label: Optional[str],
        event_time: datetime,
        consumed_oids: List[str],
        produced_oids: List[str],
        reserved_resources: List[Tuple[str, str]],
        case_root: Optional[str],
    ) -> None:
        if not produced_label:
            return
        rt = self.runtime
        rels = rt._build_event_relationships(
            event_label=produced_label,
            consumed_oids=consumed_oids,
            produced_oids=produced_oids,
            reserved_resources=reserved_resources,
            case_root=case_root,
        )
        rt.log.emit_event(etype=produced_label, time=event_time, attributes={}, relationships=rels)
        rt._update_object_relationships_from_event(rels)

    def event_creates_type(self, event_label: Optional[str], object_type: Optional[str]) -> bool:
        if not event_label or not object_type:
            return False
        qualifiers = self.runtime.event_relationships.get(event_label, {}).get(object_type) or []
        return any(str(qualifier).upper() == "CREATE" for qualifier in qualifiers)

    def created_related_objects_for_event(
        self,
        *,
        event_label: Optional[str],
        out_type: Optional[str],
        consumed_oids: List[str],
        case_root: Optional[str],
    ) -> Set[str]:
        if not case_root or not self.event_creates_type(event_label, out_type):
            return set()
        related: Set[str] = set()
        for in_oid in consumed_oids:
            related.update(self.runtime._get_related_of_type(in_oid, out_type, case_root))
        return related


    def iteration_deferred_create_types(
        self,
        *,
        event_label: Optional[str],
        parent_type: Optional[str],
        child_type: Optional[str],
    ) -> Set[str]:
        if not event_label:
            return set()
        out: Set[str] = set()
        for object_type, qualifiers in self.runtime.event_relationships.get(event_label, {}).items():
            if object_type in (parent_type, child_type):
                continue
            if object_type in self.runtime.resource_object_types:
                continue
            if any(str(qualifier).upper() == "CREATE" for qualifier in qualifiers):
                out.add(object_type)
        return out

    def related_objects_for_event_type(
        self,
        *,
        event_label: Optional[str],
        out_type: Optional[str],
        consumed_oids: List[str],
        case_root: Optional[str],
    ) -> Set[str]:
        if not out_type:
            return set()
        rt = self.runtime
        related = self.created_related_objects_for_event(
            event_label=event_label,
            out_type=out_type,
            consumed_oids=consumed_oids,
            case_root=case_root,
        )
        if not related:
            for in_oid in consumed_oids:
                related.update(rt._directional_related_oids(in_oid, out_type))
        if not related and case_root:
            for in_oid in consumed_oids:
                related.update(rt._get_related_of_type(in_oid, out_type, case_root))
        return related


    def limit_reusable_source_event_targets(
        self,
        *,
        event_label: Optional[str],
        out_type: Optional[str],
        consumed_oids: List[str],
        candidate_oids: Set[str],
        case_root: Optional[str],
    ) -> Set[str]:
        if not event_label or not out_type or not candidate_oids:
            return candidate_oids
        rt = self.runtime
        matching_cfgs = []
        for source_oid in consumed_oids:
            try:
                source_type = rt.graph.type_of(source_oid)
            except Exception:
                continue
            if source_type not in getattr(rt.graph, "reusable_types", set()):
                continue
            cfg = rt._reusable_source_event_target_config(event_label, source_type, out_type)
            if cfg:
                matching_cfgs.append(cfg)
        if not matching_cfgs:
            return candidate_oids

        cfg = matching_cfgs[0]
        done_oids = rt._done_oids_by_case.get(case_root, set()) if case_root else set()
        scoped_candidates = set(candidate_oids)
        if case_root is not None:
            in_current_root = {
                oid for oid in candidate_oids
                if case_root in rt.graph.oid_case_roots.get(oid, set())
            }
            if in_current_root:
                scoped_candidates = in_current_root

        available = [
            oid for oid in sorted(scoped_candidates)
            if rt.marking.get_object_location(oid) is None and oid not in done_oids
        ]
        if not available:
            return set()
        count = rt._sample_reusable_source_event_target_count(cfg)
        return set(available[: min(count, len(available))])

    def begin_completion(self) -> CompletionContext:
        rt = self.runtime
        completion = heappop(rt._q)
        transition_id = completion.tid
        transition = rt.net.transitions[transition_id]
        consumed = completion.consumed
        consumed_oids = [tok.oid for _, tok in consumed]
        case_id = next((tok.case_id for _, tok in consumed if tok.case_id is not None), None)
        rt.case_state.request_check(case_id)
        case_root = rt._cid_to_root_oid.get(case_id) if case_id else None
        post_places = sorted(rt.net.post.get(transition_id, set()))

        return CompletionContext(
            completion=completion,
            transition_id=transition_id,
            consumed=consumed,
            consumed_oids=consumed_oids,
            case_id=case_id,
            case_root=case_root,
            post_places=post_places,
            produced_label=transition.label,
        )

    def complete_one(self) -> datetime:
        rt = self.runtime
        ctx = self.begin_completion()
        comp = ctx.completion
        transition_id = ctx.transition_id
        consumed = ctx.consumed
        consumed_oids = ctx.consumed_oids
        case_id = ctx.case_id
        case_root = ctx.case_root
        post_places = ctx.post_places
        produced_label = ctx.produced_label

        seq_cfg = rt.aggregation_config.get(transition_id)
        loopback_oid, exit_oids = None, []
        produced_child_oids: List[str] = []
        suppress_event_emit = False

        if seq_cfg:
            mode = seq_cfg.get("mode", "consume")
            parent_type, child_type = seq_cfg["parent_type"], seq_cfg["child_type"]
            parent_tok = next(
                (tok for place, tok in consumed if rt.net.places[place].object_type == parent_type),
                None,
            )
            child_toks = [
                tok for place, tok in consumed
                if rt.net.places[place].object_type == child_type
            ]

            if parent_tok:
                expected = set(rt._iteration_targets_for(transition_id, parent_tok.oid, child_type))
                progress = rt._sync_progress.setdefault((parent_tok.oid, transition_id, child_type), set())
                born = rt._born_oids_by_case.get(case_root, set()) if case_root else set()
                if mode == "produce":
                    unprocessed = sorted(oid for oid in expected if oid not in progress or oid not in born)
                    if unprocessed:
                        batch_size = rt._sample_iteration_batch_size(seq_cfg)
                        produced_child_oids = unprocessed[:batch_size]
                        for child_oid in produced_child_oids:
                            progress.add(child_oid)
                            rt._register_born_oid(child_oid, case_id=case_id, case_root=case_root)
                        if len(unprocessed) > len(produced_child_oids):
                            loopback_oid = parent_tok.oid
                        else:
                            exit_oids.append(parent_tok.oid)
                    else:
                        exit_oids.append(parent_tok.oid)
                elif child_toks:
                    for child_tok in child_toks:
                        progress.add(child_tok.oid)
                    unprocessed = {oid for oid in expected if oid not in progress}
                    if unprocessed:
                        loopback_oid = parent_tok.oid
                    else:
                        exit_oids.append(parent_tok.oid)
                elif expected and expected.issubset(progress):
                    exit_oids.append(parent_tok.oid)
                    suppress_event_emit = True

        final_produced_oids: Set[str] = set()
        relationship_only_oids: Set[str] = set()

        if seq_cfg:
            mode = seq_cfg.get("mode", "consume")
            parent_type = seq_cfg["parent_type"]
            child_type = seq_cfg["child_type"]
            loop_back_place = seq_cfg["loop_back_place"]
            seq_produced_pairs: Set[Tuple[str, str]] = set()
            deferred_create_types = self.iteration_deferred_create_types(
                event_label=produced_label,
                parent_type=parent_type,
                child_type=child_type,
            )

            if loopback_oid:
                loopback_case_id = self.case_id_from_consumed(consumed, loopback_oid)
                self.add_output_token(
                    place=loop_back_place,
                    oid=loopback_oid,
                    case_id=loopback_case_id,
                    last_label=produced_label,
                    case_root=case_root,
                    final_produced_oids=final_produced_oids,
                )
            elif exit_oids:
                parent_exit_places = [
                    place
                    for place in post_places
                    if place != loop_back_place and rt.net.places[place].object_type in (parent_type, None)
                ]
                if not parent_exit_places:
                    parent_exit_places = [loop_back_place] if loop_back_place else []
                for oid in exit_oids:
                    parent_case_id = self.case_id_from_consumed(consumed, oid)
                    for exit_place in parent_exit_places:
                        if oid not in final_produced_oids:
                            self.add_output_token(
                                place=exit_place,
                                oid=oid,
                                case_id=parent_case_id,
                                last_label=produced_label,
                                case_root=case_root,
                                final_produced_oids=final_produced_oids,
                            )

            if mode == "produce" and produced_child_oids:
                child_case_id = self.case_id_from_consumed(consumed, parent_tok.oid)
                child_out_places = [
                    place for place in post_places
                    if rt.net.places[place].object_type == child_type
                ]
                for produced_child_oid in produced_child_oids:
                    if child_out_places:
                        for child_place in child_out_places:
                            if produced_child_oid not in final_produced_oids:
                                self.add_output_token(
                                    place=child_place,
                                    oid=produced_child_oid,
                                    case_id=child_case_id,
                                    last_label=produced_label,
                                    case_root=case_root,
                                    final_produced_oids=final_produced_oids,
                                )
                    else:
                        rt._mark_done_oid(produced_child_oid, case_root=case_root)
                        final_produced_oids.add(produced_child_oid)
            else:
                child_toks = [
                    tok for place, tok in consumed
                    if rt.net.places[place].object_type == child_type
                ]
                for child_tok in child_toks:
                    child_out_places = [
                        place for place in post_places
                        if rt.net.places[place].object_type == child_type
                    ]
                    if child_out_places:
                        for child_place in child_out_places:
                            if child_tok.oid not in final_produced_oids:
                                self.add_output_token(
                                    place=child_place,
                                    oid=child_tok.oid,
                                    case_id=child_tok.case_id,
                                    last_label=produced_label,
                                    case_root=case_root,
                                    final_produced_oids=final_produced_oids,
                                )
                    else:
                        rt._mark_done_oid(child_tok.oid, case_root=case_root)
                        final_produced_oids.add(child_tok.oid)

            for in_place, in_tok in consumed:
                in_type = rt.net.places[in_place].object_type
                if in_type in (parent_type, child_type):
                    continue
                if in_type in rt.resource_object_types:
                    continue

                same_type_out_places = [
                    place for place in post_places
                    if rt.net.places[place].object_type == in_type
                ]
                for out_place in same_type_out_places:
                    self.add_output_pair_once(
                        place=out_place,
                        oid=in_tok.oid,
                        case_id=in_tok.case_id,
                        last_label=produced_label,
                        case_root=case_root,
                        produced_pairs=seq_produced_pairs,
                        final_produced_oids=final_produced_oids,
                    )

            done_oids = rt._done_oids_by_case.get(case_root, set()) if case_root else set()
            for out_place in post_places:
                out_type = rt.net.places[out_place].object_type
                if out_type is None:
                    continue
                if out_type in (parent_type, child_type):
                    continue
                if out_type in rt.resource_object_types:
                    continue

                related = self.related_objects_for_event_type(
                    event_label=produced_label,
                    out_type=out_type,
                    consumed_oids=consumed_oids,
                    case_root=case_root,
                )
                related = self.limit_reusable_source_event_targets(
                    event_label=produced_label,
                    out_type=out_type,
                    consumed_oids=consumed_oids,
                    candidate_oids=related,
                    case_root=case_root,
                )

                if out_type in deferred_create_types and loopback_oid:
                    relationship_only_oids.update(related)
                    continue

                for oid in sorted(related):
                    if rt.marking.get_object_location(oid) is not None or oid in done_oids:
                        continue
                    self.add_output_pair_once(
                        place=out_place,
                        oid=oid,
                        case_id=case_id,
                        last_label=produced_label,
                        case_root=case_root,
                        produced_pairs=seq_produced_pairs,
                        final_produced_oids=final_produced_oids,
                    )

        else:
            produced_pairs: Set[Tuple[str, str]] = set()
            for out_place in post_places:
                out_type = rt.net.places[out_place].object_type

                if out_type is None:
                    for oid in consumed_oids:
                        oid_case_id = self.case_id_from_consumed(consumed, oid)
                        self.add_output_pair_once(
                            place=out_place,
                            oid=oid,
                            case_id=oid_case_id,
                            last_label=produced_label,
                            case_root=case_root,
                            produced_pairs=produced_pairs,
                            final_produced_oids=final_produced_oids,
                        )
                    continue

                same_type_oids = [
                    oid for oid in consumed_oids
                    if rt.graph.type_of(oid) == out_type
                ]
                if same_type_oids:
                    for oid in same_type_oids:
                        case_id_for = self.case_id_from_consumed(consumed, oid)
                        self.add_output_pair_once(
                            place=out_place,
                            oid=oid,
                            case_id=case_id_for,
                            last_label=produced_label,
                            case_root=case_root,
                            produced_pairs=produced_pairs,
                            final_produced_oids=final_produced_oids,
                        )
                else:
                    related = self.created_related_objects_for_event(
                        event_label=produced_label,
                        out_type=out_type,
                        consumed_oids=consumed_oids,
                        case_root=case_root,
                    )
                    if not related:
                        for in_oid in consumed_oids:
                            related.update(rt._directional_related_oids(in_oid, out_type))
                    if not related and case_root:
                        for in_oid in consumed_oids:
                            related.update(rt._get_related_of_type(in_oid, out_type, case_root))
                    related = self.limit_reusable_source_event_targets(
                        event_label=produced_label,
                        out_type=out_type,
                        consumed_oids=consumed_oids,
                        candidate_oids=related,
                        case_root=case_root,
                    )

                    done_oids = rt._done_oids_by_case.get(case_root, set()) if case_root else set()
                    for oid in sorted(related):
                        if rt.marking.get_object_location(oid) is not None or oid in done_oids:
                            continue
                        self.add_output_pair_once(
                            place=out_place,
                            oid=oid,
                            case_id=case_id,
                            last_label=produced_label,
                            case_root=case_root,
                            produced_pairs=produced_pairs,
                            final_produced_oids=final_produced_oids,
                        )

        self.mark_unproduced_consumed_tokens(
            consumed=consumed,
            final_produced_oids=final_produced_oids,
            produced_label=produced_label,
            case_root=case_root,
        )
        self.emit_visible_event(
            produced_label=None if suppress_event_emit else produced_label,
            event_time=comp.start_time,
            consumed_oids=consumed_oids,
            produced_oids=list(final_produced_oids | relationship_only_oids),
            reserved_resources=comp.reserved_resources,
            case_root=case_root,
        )

        return comp.time
