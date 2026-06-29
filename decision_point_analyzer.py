from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from process_discoverer import extract as extract_process_model
from object_analyzer import extract as extract_object_semantics



def load_ocel(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "objects": data.get("objects") or data.get("ocel:objects") or [],
        "events": data.get("events") or data.get("ocel:events") or [],
        "objectTypes": data.get("objectTypes") or data.get("ocel:objectTypes") or [],
        "eventTypes": data.get("eventTypes") or data.get("ocel:eventTypes") or [],
    }


COMPLETE_CASE_LABEL = "Complete Case"


# -----------------------------
# Graph helpers (Petri Net Analysis)
# -----------------------------
def is_tau_transition(t_key: str, transitions: Dict[str, Optional[str]]) -> bool:
    """Check if transition is hidden/tau (None label)."""
    return transitions.get(t_key) is None


def build_bipartite_adjacency(
    places: Set[str],
    transitions: Dict[str, Optional[str]],
    arcs: List[Tuple[str, str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Map out Petri Net structure (Place/Transition adjacency)."""
    place_out: Dict[str, Set[str]] = defaultdict(set)
    place_in: Dict[str, Set[str]] = defaultdict(set)
    trans_out: Dict[str, Set[str]] = defaultdict(set)
    trans_in: Dict[str, Set[str]] = defaultdict(set)

    for src, tgt in arcs:
        if src in places and tgt in transitions:
            place_out[src].add(tgt)
            trans_in[tgt].add(src)
        elif src in transitions and tgt in places:
            trans_out[src].add(tgt)
            place_in[tgt].add(src)
    return place_out, place_in, trans_out, trans_in


def collect_visible_neighbors(
    p: str,
    *,
    transitions: Dict[str, Optional[str]],
    adj_out: Dict[str, Set[str]],
    trans_out: Dict[str, Set[str]],
    direction: str = "forward"
) -> Set[str]:
    """DFS to find visible transitions skipping tau/hidden transitions."""
    visible: Set[str] = set()
    visited_p, visited_t = set(), set()

    def dfs_p(curr_p: str):
        if curr_p in visited_p: return
        visited_p.add(curr_p)
        for t in adj_out.get(curr_p, set()): dfs_t(t)

    def dfs_t(curr_t: str):
        if curr_t in visited_t: return
        visited_t.add(curr_t)
        if is_tau_transition(curr_t, transitions):
            for nxt_p in trans_out.get(curr_t, set()): dfs_p(nxt_p)
        else:
            visible.add(transitions[curr_t])

    dfs_p(p)
    return visible


def find_decision_points_artifact(
    places: Set[str],
    transitions: Dict[str, Optional[str]],
    arcs: List[Tuple[str, str]],
    object_types: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Identify branch places including virtual 'Complete Case' exits."""
    place_out, place_in, trans_out, trans_in = build_bipartite_adjacency(places, transitions, arcs)

    known_object_types: Set[str] = set(object_types or set())
    for p in places:
        if p.endswith("_source"):
            known_object_types.add(p[: -len("_source")])
        elif p.endswith("_sink"):
            known_object_types.add(p[: -len("_sink")])

    sorted_types = sorted(known_object_types, key=len, reverse=True)

    def _resolve_object_type(place: str) -> str:
        # Prefer longest known object-type prefix match.
        for ot in sorted_types:
            if place == ot or place.startswith(f"{ot}_"):
                return ot
        # Fallback for unknown/auxiliary places.
        return place.split("_")[0] if "_" in place else "unknown"

    def _reaches_terminal_without_visible(start_place: str) -> bool:
        """
        Return True when a path exists from `start_place` to a terminal place
        through tau transitions only (no visible transition encountered).
        """
        if start_place not in places:
            return False

        terminal_places = {p for p in places if not place_out.get(p)}
        if not terminal_places:
            return False

        seen_places: Set[str] = set()
        seen_trans: Set[str] = set()
        queue: List[str] = [start_place]

        while queue:
            p = queue.pop(0)
            if p in seen_places:
                continue
            seen_places.add(p)

            if p in terminal_places:
                return True

            for tid in place_out.get(p, set()):
                if tid in seen_trans:
                    continue
                seen_trans.add(tid)
                if not is_tau_transition(tid, transitions):
                    continue
                for nxt_p in trans_out.get(tid, set()):
                    if nxt_p not in seen_places:
                        queue.append(nxt_p)

        return False

    decision_points = []
    for p in places:
        obj_type = _resolve_object_type(p)
        
        prev = collect_visible_neighbors(p, transitions=transitions, adj_out=place_in, trans_out=trans_in)
        nxt = set(collect_visible_neighbors(p, transitions=transitions, adj_out=place_out, trans_out=trans_out))
        if _reaches_terminal_without_visible(p):
            nxt.add(COMPLETE_CASE_LABEL)
        if len(nxt) >= 2 and prev:
            decision_points.append({
                "place": p, 
                "object_type": obj_type,
                "prev": sorted(prev), 
                "next": sorted(nxt)
            })
    return decision_points


# -----------------------------
# Object-Centric Probability Calculation
# -----------------------------
def _get_activity_metadata(ocel: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[int, str, List[str]]]], Dict[str, str]]:
    """Pre-scan OCEL to map activities to their object types and generate chronologically sorted event streams."""
    events = ocel.get("events", []) or []
    objects = ocel.get("objects", []) or []
    id_to_type = {str(o.get("id") or o.get("ocel:oid")): str(o.get("type") or o.get("ocel:type")) for o in objects}

    # IMPORTANT: Sort events by timestamp to ensure correct chronological flow tracking
    def _get_ts(ev):
        for k in ["ocel:timestamp", "timestamp", "time"]:
            if k in ev: return str(ev[k])
        return ""
    
    sorted_events = sorted(events, key=_get_ts)

    act_to_types = defaultdict(set)
    full_event_list = [] # List of (index, activity, [oids])

    for i, ev in enumerate(sorted_events):
        act = str(ev.get("type") or ev.get("ocel:activity") or "unknown")
        oids = []
        for rel in (ev.get("relationships") or []):
            oid = str(rel.get("objectId") or rel.get("ocel:oid"))
            oids.append(oid)
            if oid in id_to_type:
                act_to_types[act].add(id_to_type[oid])
        full_event_list.append((i, act, oids))
    
    return act_to_types, full_event_list, id_to_type


def _calculate_branch_probs(
    event_list: List[Tuple[int, str, List[str]]],
    prev_acts: List[str],
    next_acts: List[str],
    target_obj_type: str,
    act_to_types: Dict[str, Set[str]],
    oid_to_type: Dict[str, str],
    rounding: int = 3
) -> Dict[str, Dict[str, float]]:
    """Calculate transition probabilities based on common Object ID flows."""
    results = {}
    
    # 1. Map each OID to its sorted list of activities
    oid_to_stream = defaultdict(list)
    for idx, act, oids in event_list:
        for oid in oids:
            oid_to_stream[oid].append(act)

    all_next_set = set(next_acts)
    non_terminal_next_set = {n for n in all_next_set if n != COMPLETE_CASE_LABEL}
    
    for prev in prev_acts:
        counts = Counter()
        total = 0
        
        # 3. Track transitions ONLY for IDs belonging to the specific target object type
        target_oids = [oid for oid in oid_to_stream.keys() if oid_to_type.get(oid) == target_obj_type]
        
        for oid in target_oids:
            stream = oid_to_stream[oid]
            for i, act in enumerate(stream):
                if act == prev:
                    matched_next = False
                    for j in range(i + 1, len(stream)):
                        nxt_candidate = stream[j]
                        if nxt_candidate in non_terminal_next_set:
                            counts[nxt_candidate] += 1
                            total += 1
                            matched_next = True
                            break
                        elif nxt_candidate == prev:
                            break
                    if not matched_next and COMPLETE_CASE_LABEL in all_next_set:
                        counts[COMPLETE_CASE_LABEL] += 1
                        total += 1
        
        if total > 0:
            # Only include next activities that are actually related to this object type
            valid_next = [
                nxt
                for nxt in next_acts
                if nxt == COMPLETE_CASE_LABEL or target_obj_type in act_to_types.get(nxt, set())
            ]
            results[prev] = {nxt: round(counts[nxt] / total, rounding) for nxt in valid_next}
            
    return results


def extract_decisions_base(
    ocel: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
    rounding: int = 3,
) -> Dict[str, Any]:
    """Main extractor for Branch Probabilities."""
    context = context or {}
    pm = context.get("process_model") or {}
    places = set(pm.get("PLACES") or [])
    transitions = pm.get("TRANSITIONS") or {}
    arcs = [tuple(a) for a in (pm.get("ARCS") or [])]
    object_types = set((context.get("objects") or {}).get("OBJECT_TYPES", []))

    if not places or not transitions:
        return {"BRANCH_PROB": {}}

    decision_points = find_decision_points_artifact(places, transitions, arcs, object_types=object_types)
    if not decision_points:
        return {"BRANCH_PROB": {}}

    act_to_types, event_list, id_to_type = _get_activity_metadata(ocel)
    
    # Structure: BRANCH_PROB[prev_act][next_act] = prob
    merged_probs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for dp in decision_points:
        ot = dp["object_type"]
        probs = _calculate_branch_probs(event_list, dp["prev"], dp["next"], ot, act_to_types, id_to_type, rounding)
        
        for prev, next_dict in probs.items():
            for nxt, prob in next_dict.items():
                merged_probs[prev][nxt].append(float(prob))

    all_final_probs: Dict[str, Dict[str, float]] = {}
    for prev, next_dict in merged_probs.items():
        averaged = {nxt: round(sum(vals) / len(vals), rounding) for nxt, vals in next_dict.items() if vals}
        norm = sum(averaged.values())
        if norm > 0:
            all_final_probs[prev] = {
                nxt: round(val / norm, rounding)
                for nxt, val in averaged.items()
            }

    return {"BRANCH_PROB": all_final_probs}



def _extract_iterative_create_loops(
    branch_prob: Dict[str, Dict[str, float]],
    iterative_targets: Dict[str, Any],
) -> Dict[str, Any]:
    loops = {}
    for target_type, details in sorted(iterative_targets.items()):
        target_loops = []
        for cfg in details:
            create_activity = cfg.get("create_activity")
            if not create_activity:
                continue
            choices = branch_prob.get(create_activity) or {}
            if choices.get(create_activity) is None:
                continue
            loop_cfg = dict(cfg)
            loop_cfg["target_object_type"] = target_type
            target_loops.append(loop_cfg)
        if target_loops:
            loops[target_type] = target_loops
    return loops


def extract(ocel: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = dict(context or {})
    if not context.get("process_model"):
        cf_out = extract_process_model(ocel, context=context)
        context.update(cf_out)
        context["process_model"] = cf_out.get("process_model") or {}

    decision = extract_decisions_base(ocel, context=context)
    semantics = context.get("object_semantics") or extract_object_semantics(ocel)
    iterative_targets = (semantics.get("relations") or {}).get("ITERATIVE_CREATE_TARGETS") or {}
    iterative_loops = _extract_iterative_create_loops(decision.get("BRANCH_PROB") or {}, iterative_targets)
    if iterative_loops:
        decision["ITERATIVE_CREATE_LOOP"] = iterative_loops
        branch_prob = decision.get("BRANCH_PROB") or {}
        for details in iterative_loops.values():
            for cfg in details:
                create_activity = cfg.get("create_activity")
                if create_activity:
                    branch_prob.pop(create_activity, None)
    return decision



if __name__ == "__main__":
    print("This module is used by simulation_input_builder.py. Run run_pipeline.py to build inputs and simulate OCEL logs.")
