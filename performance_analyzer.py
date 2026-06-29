from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone


def load_ocel(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "objects": data.get("objects") or data.get("ocel:objects") or [],
        "events": data.get("events") or data.get("ocel:events") or [],
        "objectTypes": data.get("objectTypes") or data.get("ocel:objectTypes") or [],
        "eventTypes": data.get("eventTypes") or data.get("ocel:eventTypes") or [],
    }


def parse_ocel_time(ts: Optional[str]):
    if not ts or not isinstance(ts, str):
        return None
    s = ts.strip()
    try:
        s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        s = s.rstrip("Z")
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def build_object_id_to_type(objects: List[Dict[str, Any]]) -> Dict[str, str]:
    id_to_type = {}
    for obj in objects:
        oid = obj.get("id") or obj.get("ocel:oid")
        otype = obj.get("type") or obj.get("ocel:type")
        if oid is not None and otype is not None:
            id_to_type[str(oid)] = str(otype)
    return id_to_type


from process_discoverer import extract as extract_process_model


RESOURCE_QUALIFIER = "RESOURCE"
INITIATE_QUALIFIER = "INITIATE"


def _event_activity(event: Dict[str, Any]) -> str:
    return str(event.get("type") or event.get("ocel:activity") or "unknown").strip()


def _event_timestamp(event: Dict[str, Any]) -> float:
    ts = event.get("timestamp") or event.get("ocel:timestamp") or event.get("time")
    parsed = parse_ocel_time(ts)
    return parsed.timestamp() if parsed else 0.0


def _relationship_oid(rel: Dict[str, Any]) -> str:
    return str(rel.get("objectId") or rel.get("ocel:oid") or rel.get("id") or "").strip()


def _relationship_qualifiers(rel: Dict[str, Any]) -> Set[str]:
    qualifier = rel.get("qualifier") or rel.get("ocel:qualifier") or "standard"
    if isinstance(qualifier, list):
        return {str(q).strip().upper() for q in qualifier if str(q).strip()}
    q = str(qualifier).strip()
    return {q.upper()} if q else set()


def _event_has_resource_relationship(event: Dict[str, Any]) -> bool:
    return any(
        RESOURCE_QUALIFIER in _relationship_qualifiers(rel)
        for rel in event.get("relationships", []) or []
    )


def _mean_std_count(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    n = len(values)
    mean_v = sum(values) / n
    std_v = math.sqrt(sum((x - mean_v) ** 2 for x in values) / n) if n > 1 else 0.0
    return {"mean": round(mean_v, 3), "std": round(std_v, 3), "count": n}


def _is_tau_transition(t_key: str, transitions: Dict[str, Optional[str]]) -> bool:
    return transitions.get(t_key) is None


def _build_bipartite_adjacency(
    places: Set[str],
    transitions: Dict[str, Optional[str]],
    arcs: List[Tuple[str, str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    place_out: Dict[str, Set[str]] = defaultdict(set)
    trans_out: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in arcs:
        if src in places and tgt in transitions:
            place_out[src].add(tgt)
        elif src in transitions and tgt in places:
            trans_out[src].add(tgt)
    return place_out, trans_out


def _collect_visible_successors(
    transition_key: str,
    *,
    transitions: Dict[str, Optional[str]],
    trans_out: Dict[str, Set[str]],
    place_out: Dict[str, Set[str]],
) -> Set[str]:
    visible: Set[str] = set()
    visited_places: Set[str] = set()
    visited_transitions: Set[str] = set()

    def visit_place(place: str) -> None:
        if place in visited_places:
            return
        visited_places.add(place)
        for nxt_transition in place_out.get(place, set()):
            visit_transition(nxt_transition)

    def visit_transition(curr_transition: str) -> None:
        if curr_transition in visited_transitions:
            return
        visited_transitions.add(curr_transition)
        if _is_tau_transition(curr_transition, transitions):
            for nxt_place in trans_out.get(curr_transition, set()):
                visit_place(nxt_place)
        else:
            label = transitions.get(curr_transition)
            if label is not None:
                visible.add(label)

    for place in trans_out.get(transition_key, set()):
        visit_place(place)
    return visible


def _build_successor_map(process_model: Dict[str, Any]) -> Dict[str, Set[str]]:
    places = set(process_model.get("PLACES") or [])
    transitions = process_model.get("TRANSITIONS") or {}
    arcs = [tuple(arc) for arc in process_model.get("ARCS") or []]
    if not places or not transitions:
        return {}

    place_out, trans_out = _build_bipartite_adjacency(places, transitions, arcs)
    successors: Dict[str, Set[str]] = {}
    for transition_key, label in transitions.items():
        if label is not None:
            successors[label] = _collect_visible_successors(
                transition_key,
                transitions=transitions,
                trans_out=trans_out,
                place_out=place_out,
            )
    return successors


def _activity_metadata_without_resource_events(
    ocel: Dict[str, Any],
) -> Tuple[Dict[str, Set[str]], List[Tuple[float, str, List[str]]], Dict[str, str]]:
    id_to_type = build_object_id_to_type(ocel.get("objects", []) or [])
    events = sorted(ocel.get("events", []) or [], key=_event_timestamp)
    act_to_types: Dict[str, Set[str]] = defaultdict(set)
    event_list: List[Tuple[float, str, List[str]]] = []

    for event in events:
        if _event_has_resource_relationship(event):
            continue
        act = _event_activity(event)
        ts = _event_timestamp(event)
        oids: List[str] = []
        for rel in event.get("relationships", []) or []:
            oid = _relationship_oid(rel)
            if not oid:
                continue
            oids.append(oid)
            otype = id_to_type.get(oid)
            if otype:
                act_to_types[act].add(otype)
        event_list.append((ts, act, oids))

    return dict(act_to_types), event_list, id_to_type


def calculate_event_duration(
    ocel: Dict[str, Any],
    process_model: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    successors_map = _build_successor_map(process_model)
    if not successors_map:
        return {}

    act_to_types, event_list, oid_to_type = _activity_metadata_without_resource_events(ocel)
    oid_to_stream: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for ts, act, oids in event_list:
        for oid in oids:
            oid_to_stream[oid].append((ts, act))

    deltas_by_activity: Dict[str, List[float]] = defaultdict(list)

    for prev_act, next_acts in successors_map.items():
        for next_act in next_acts:
            common_types = act_to_types.get(prev_act, set()) & act_to_types.get(next_act, set())
            if not common_types:
                continue
            for oid, stream in oid_to_stream.items():
                if oid_to_type.get(oid) not in common_types:
                    continue
                for i, (prev_ts, act) in enumerate(stream):
                    if act != prev_act:
                        continue
                    for next_ts, candidate in stream[i + 1 :]:
                        if candidate == next_act:
                            delta = next_ts - prev_ts
                            if delta >= 0:
                                deltas_by_activity[prev_act].append(delta)
                            break
                        if candidate == prev_act:
                            break

    return {
        act: _mean_std_count(deltas)
        for act, deltas in sorted(deltas_by_activity.items())
        if deltas
    }



def calculate_resource_event_duration(
    ocel: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    id_to_type = build_object_id_to_type(ocel.get("objects", []) or [])
    streams: Dict[Tuple[str, str], List[Tuple[float, str]]] = defaultdict(list)

    for event in ocel.get("events", []) or []:
        ts = _event_timestamp(event)
        act = _event_activity(event)
        for rel in event.get("relationships", []) or []:
            if RESOURCE_QUALIFIER not in _relationship_qualifiers(rel):
                continue
            oid = _relationship_oid(rel)
            rtype = id_to_type.get(oid)
            if oid and rtype:
                streams[(rtype, oid)].append((ts, act))

    deltas: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (rtype, oid), events in streams.items():
        ordered = sorted(events, key=lambda item: item[0])
        for idx in range(len(ordered) - 1):
            ts, act = ordered[idx]
            next_ts = ordered[idx + 1][0]
            diff = next_ts - ts
            if diff >= 0:
                deltas[rtype][oid][act].append(diff)

    return {
        rtype: {
            oid: {
                act: {k: v for k, v in _mean_std_count(values).items() if k != "count"}
                for act, values in sorted(by_act.items())
                if values
            }
            for oid, by_act in sorted(by_id.items())
        }
        for rtype, by_id in sorted(deltas.items())
    }

def _top_level_object_types(ocel: Dict[str, Any]) -> Set[str]:
    id_to_type = build_object_id_to_type(ocel.get("objects", []) or [])
    object_types = set(id_to_type.values())
    child_types: Set[str] = set()

    for obj in ocel.get("objects", []) or []:
        rels = obj.get("relationships") or obj.get("ocel:relationships") or []
        for rel in rels:
            tgt_id = str(rel.get("objectId") or rel.get("ocel:oid") or rel.get("id") or "")
            tgt_type = id_to_type.get(tgt_id)
            if tgt_type:
                child_types.add(tgt_type)

    return object_types - child_types


def calculate_initiate_arrivals(ocel: Dict[str, Any]) -> Dict[str, Any]:
    id_to_type = build_object_id_to_type(ocel.get("objects", []) or [])
    top_level_types = _top_level_object_types(ocel)
    events = sorted(ocel.get("events", []) or [], key=_event_timestamp)

    initiate_types: Set[str] = set()
    arrival_points: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    seen_oids: Set[str] = set()

    for event in events:
        ts = _event_timestamp(event)
        act = _event_activity(event)
        for rel in event.get("relationships", []) or []:
            if INITIATE_QUALIFIER not in _relationship_qualifiers(rel):
                continue
            oid = _relationship_oid(rel)
            otype = id_to_type.get(oid)
            if not oid or not otype:
                continue

            initiate_types.add(otype)
            if oid in seen_oids:
                continue
            seen_oids.add(oid)
            arrival_points[otype].append((ts, act))

    stats: Dict[str, Dict[str, Any]] = {}
    for otype, points in sorted(arrival_points.items()):
        timestamps = sorted(ts for ts, _ in points)
        intervals = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
            if timestamps[i] - timestamps[i - 1] >= 0
        ]
        interval_stats = _mean_std_count(intervals)
        mean_interval = interval_stats["mean"] if interval_stats["count"] else 0.0
        stats[otype] = {
            "lambda": round(1.0 / mean_interval, 8) if mean_interval > 0 else 0.0,
            "count": len(timestamps),
            "mean_interarrival_time": interval_stats["mean"],
            "std_interarrival_time": interval_stats["std"],
            "start_place": f"{otype}_source",
            "start_events": sorted({act for _, act in points}),
            "is_top_level": otype in top_level_types,
        }

    return {
        "ARRIVAL_STATS": stats,
        "INITIATE_OBJECT_TYPES": sorted(initiate_types),
        "TOP_LEVEL_INITIATE_OBJECT_TYPES": sorted(
            otype for otype in stats if otype in top_level_types
        ),
        "NON_TOP_LEVEL_INITIATE_OBJECT_TYPES": sorted(
            otype for otype in stats if otype not in top_level_types
        ),
    }


def extract(ocel: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = dict(context or {})
    if not context.get("process_model"):
        cf_out = extract_process_model(ocel, context=context)
        context.update(cf_out)
        context["process_model"] = cf_out.get("process_model") or {}

    arrivals = calculate_initiate_arrivals(ocel)
    event_duration = calculate_event_duration(ocel, context.get("process_model") or {})
    resource_event_duration = calculate_resource_event_duration(ocel)
    arrival_stats = arrivals.get("ARRIVAL_STATS") or {}

    for cfg in event_duration.values():
        cfg.pop("count", None)
    for cfg in arrival_stats.values():
        cfg.pop("count", None)

    return {
        "EVENT_DURATION": event_duration,
        "RESOURCE_EVENT_DURATION": resource_event_duration,
        "ARRIVAL_STATS": arrival_stats,
    }



if __name__ == "__main__":
    print("This module is used by simulation_input_builder.py. Run run_pipeline.py to build inputs and simulate OCEL logs.")
