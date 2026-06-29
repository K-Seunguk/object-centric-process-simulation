from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import json
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




SERIAL_ID_RE = re.compile(r"^(.+?)[_\-\s]?\d+$")


def _qualifiers(rel: Dict[str, Any]) -> List[str]:
    qual = rel.get("qualifier") or rel.get("ocel:qualifier") or "standard"
    if isinstance(qual, list):
        return [str(q).strip() for q in qual if str(q).strip()]
    return [str(qual).strip()] if str(qual).strip() else []


def _event_type(ev: Dict[str, Any]) -> str:
    return str(ev.get("type") or ev.get("ocel:activity") or "unknown")


def _event_time_value(ev: Dict[str, Any]) -> Optional[float]:
    ts = ev.get("ocel:timestamp") or ev.get("timestamp") or ev.get("time")
    parsed = parse_ocel_time(ts)
    return parsed.timestamp() if parsed else None


def _is_serial_like_id(oid: str) -> bool:
    s = str(oid).strip()
    if not s:
        return False
    match = SERIAL_ID_RE.match(s)
    if not match:
        return False
    prefix = match.group(1).strip("_- ")
    return bool(prefix) and any(ch.isalpha() for ch in prefix)


def _is_name_like_id(oid: str) -> bool:
    s = str(oid).strip()
    if not s:
        return False
    n = len(s)
    alpha = sum(ch.isalpha() for ch in s)
    digit = sum(ch.isdigit() for ch in s)
    return (alpha / n) >= 0.55 and (digit / n) <= 0.35


def _mean_variance(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "variance": 0.0, "count": 0}
    mean_v = sum(values) / len(values)
    variance_v = sum((x - mean_v) ** 2 for x in values) / len(values) if len(values) > 1 else 0.0
    return {"mean": round(mean_v, 3), "variance": round(variance_v, 3), "count": len(values)}


def _build_type_to_ids(objects: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    type_to_ids: Dict[str, List[str]] = defaultdict(list)
    for obj in objects:
        oid = obj.get("id") or obj.get("ocel:oid")
        otype = obj.get("type") or obj.get("ocel:type")
        if oid is not None and otype is not None:
            type_to_ids[str(otype)].append(str(oid))
    return {otype: sorted(set(oids)) for otype, oids in type_to_ids.items()}


def _extract_resource_types(ocel: Dict[str, Any], id_to_type: Dict[str, str]) -> List[str]:
    detected = set()
    for ev in ocel.get("events", []) or []:
        for rel in ev.get("relationships") or []:
            if any(q.upper() == "RESOURCE" for q in _qualifiers(rel)):
                oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
                otype = id_to_type.get(oid)
                if otype:
                    detected.add(otype)
    return sorted(detected)


def _extract_object_lifecycle_events(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
) -> Dict[str, Dict[str, List[str]]]:
    lifecycle = defaultdict(lambda: {"CREATE": set(), "INITIATE": set(), "INCORPORATE": set()})
    for ev in ocel.get("events", []) or []:
        act = _event_type(ev)
        for rel in ev.get("relationships") or []:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            otype = id_to_type.get(oid)
            if not otype:
                continue
            upper_quals = {q.upper() for q in _qualifiers(rel)}
            if "CREATE" in upper_quals:
                lifecycle[otype]["CREATE"].add(act)
            if "INITIATE" in upper_quals:
                lifecycle[otype]["INITIATE"].add(act)
            if "INCORPORATE" in upper_quals:
                lifecycle[otype]["INCORPORATE"].add(act)

    out = {}
    for otype, roles in sorted(lifecycle.items()):
        non_empty = {role: sorted(values) for role, values in roles.items() if values}
        if non_empty:
            out[otype] = non_empty
    return out


def _build_object_graph(
    objects: List[Dict[str, Any]],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
) -> Dict[str, Set[str]]:
    graph = defaultdict(set)
    for obj in objects:
        oid = str(obj.get("id") or obj.get("ocel:oid") or "")
        otype = id_to_type.get(oid)
        if not otype or otype in resource_types:
            continue
        for rel in obj.get("relationships") or obj.get("ocel:relationships") or []:
            tid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            target_type = id_to_type.get(tid)
            if not target_type or target_type in resource_types:
                continue
            graph[oid].add(tid)
            graph[tid].add(oid)
    return graph


def _build_rel_targets(
    objects: List[Dict[str, Any]],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
) -> Dict[str, List[str]]:
    targets = defaultdict(set)
    for obj in objects:
        oid = str(obj.get("id") or obj.get("ocel:oid") or "")
        src_type = id_to_type.get(oid)
        if not src_type or src_type in resource_types:
            continue
        for rel in obj.get("relationships") or obj.get("ocel:relationships") or []:
            tid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            target_type = id_to_type.get(tid)
            if target_type and target_type != src_type and target_type not in resource_types:
                targets[src_type].add(target_type)
    return {src: sorted(values) for src, values in sorted(targets.items())}


def _object_type_stats(
    objects: List[Dict[str, Any]],
    id_to_type: Dict[str, str],
    object_graph: Dict[str, Set[str]],
) -> Dict[str, Dict[str, Any]]:
    type_to_ids = _build_type_to_ids(objects)
    max_count = max((len(oids) for oids in type_to_ids.values()), default=1)
    stats = {}
    for otype, oids in type_to_ids.items():
        related_counts = [
            len({tid for tid in object_graph.get(oid, set()) if id_to_type.get(tid) != otype})
            for oid in oids
        ]
        stats[otype] = {
            "total_id_count": len(oids),
            "relative_size_ratio": round(len(oids) / max_count, 4),
            "serial_like_id_ratio": round(sum(_is_serial_like_id(oid) for oid in oids) / len(oids), 3)
            if oids
            else 0.0,
            "name_like_id_ratio": round(sum(_is_name_like_id(oid) for oid in oids) / len(oids), 3)
            if oids
            else 0.0,
            "avg_distinct_related_ids_per_id": round(sum(related_counts) / len(related_counts), 3)
            if related_counts
            else 0.0,
        }
    return stats


def _classify_reusable_types(
    stats: Dict[str, Dict[str, Any]],
    resource_types: Set[str],
) -> List[str]:
    reusable = []
    for otype, stat in stats.items():
        if otype in resource_types:
            continue
        serial_ratio = float(stat.get("serial_like_id_ratio", 0.0))
        name_ratio = float(stat.get("name_like_id_ratio", 0.0))
        relative_size = float(stat.get("relative_size_ratio", 1.0))
        avg_related = float(stat.get("avg_distinct_related_ids_per_id", 0.0))

        reference_like_ids = serial_ratio <= 0.35 and name_ratio >= 0.5
        compact_reference_pool = relative_size <= 0.35
        reused_across_objects = avg_related >= 1.5

        if reference_like_ids and compact_reference_pool and reused_across_objects:
            reusable.append(otype)
    return sorted(reusable)


def _build_cardinality_probs(
    objects: List[Dict[str, Any]],
    id_to_type: Dict[str, str],
    rel_targets: Dict[str, List[str]],
    object_graph: Dict[str, Set[str]],
    resource_types: Set[str],
    reusable_types: Set[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = defaultdict(lambda: defaultdict(Counter))
    for obj in objects:
        src_oid = str(obj.get("id") or obj.get("ocel:oid") or "")
        src_type = id_to_type.get(src_oid)
        if not src_type or src_type in resource_types or src_type in reusable_types:
            continue
        for target_type in rel_targets.get(src_type, []):
            if target_type in reusable_types:
                continue
            linked_ids = object_graph.get(src_oid, set())
            count = len({tid for tid in linked_ids if id_to_type.get(tid) == target_type}) or 1
            results[src_type][target_type][f"1:{count}"] += 1

    return {
        src: {
            target: {card: round(cnt / sum(counts.values()), 3) for card, cnt in sorted(counts.items())}
            for target, counts in sorted(targets.items())
        }
        for src, targets in sorted(results.items())
    }


def _build_reusable_selection_prob(
    object_graph: Dict[str, Set[str]],
    id_to_type: Dict[str, str],
    reusable_types: Set[str],
    resource_types: Set[str],
) -> Dict[str, Dict[str, float]]:
    counts = defaultdict(Counter)
    for oid, linked_ids in object_graph.items():
        src_type = id_to_type.get(oid)
        if not src_type or src_type in resource_types or src_type in reusable_types:
            continue
        for tid in linked_ids:
            reusable_type = id_to_type.get(tid)
            if reusable_type in reusable_types:
                counts[reusable_type][tid] += 1

    result = {}
    for reusable_type, id_counts in sorted(counts.items()):
        total = sum(id_counts.values())
        if total <= 0:
            continue
        result[reusable_type] = {
            oid: round(count / total, 6)
            for oid, count in sorted(id_counts.items())
        }
    return result




def _object_relationship_pairs(objects: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    pairs = set()
    for obj in objects:
        src_oid = str(obj.get("id") or obj.get("ocel:oid") or "")
        if not src_oid:
            continue
        for rel in obj.get("relationships") or obj.get("ocel:relationships") or []:
            tgt_oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            if tgt_oid:
                pairs.add((src_oid, tgt_oid))
    return pairs


def _prob_distribution(counts: List[int]) -> Dict[str, float]:
    total = len(counts)
    if total <= 0:
        return {}
    counter = Counter(counts)
    return {
        str(value): round(count / total, 6)
        for value, count in sorted(counter.items())
    }


def _mean_count(counts: List[int]) -> float:
    return round(sum(counts) / len(counts), 6) if counts else 0.0


def _build_iterative_create_targets(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
    reusable_types: Set[str],
) -> Dict[str, Dict[str, Any]]:
    objects = ocel.get("objects", []) or []
    rel_pairs = _object_relationship_pairs(objects)
    undirected_pairs = rel_pairs | {(target, source) for source, target in rel_pairs}

    source_to_created = defaultdict(lambda: defaultdict(set))
    event_created_counts = defaultdict(Counter)

    for event_index, ev in enumerate(ocel.get("events", []) or []):
        act = _event_type(ev)
        rels = ev.get("relationships") or []
        created = []
        candidates = []

        for rel in rels:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            otype = id_to_type.get(oid)
            if not oid or not otype or otype in resource_types or otype in reusable_types:
                continue
            upper_quals = {q.upper() for q in _qualifiers(rel)}
            if "CREATE" in upper_quals:
                created.append((oid, otype))
            elif "RESOURCE" not in upper_quals:
                candidates.append((oid, otype))

        if not created or not candidates:
            continue

        event_key = str(ev.get("id") or ev.get("ocel:eid") or event_index)
        for created_oid, created_type in created:
            for source_oid, source_type in candidates:
                if source_oid == created_oid or source_type == created_type:
                    continue
                if source_type in reusable_types or created_type in reusable_types:
                    continue
                if (source_oid, created_oid) not in undirected_pairs:
                    continue
                key = (source_type, created_type, act)
                source_to_created[key][source_oid].add(created_oid)
                event_created_counts[key][(event_key, source_oid)] += 1

    iterative_by_target = defaultdict(list)
    for (source_type, target_type, act), by_source in sorted(source_to_created.items()):
        source_counts = [len(created_ids) for created_ids in by_source.values()]
        event_counts = list(event_created_counts.get((source_type, target_type, act), {}).values())
        o2o_mean = _mean_count(source_counts)
        e2o_mean = _mean_count(event_counts)

        if not source_counts or not event_counts:
            continue
        if o2o_mean <= e2o_mean:
            continue

        iterative_by_target[target_type].append(
            {
                "source_object_type": source_type,
                "create_activity": act,
                "created_per_iteration_distribution": _prob_distribution(event_counts),
            }
        )

    return {
        target_type: details
        for target_type, details in sorted(iterative_by_target.items())
    }

def _build_iterative_incorporate_targets(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
    reusable_types: Set[str],
) -> Dict[str, Dict[str, Any]]:
    objects = ocel.get("objects", []) or []
    rel_pairs = _object_relationship_pairs(objects)
    undirected_pairs = rel_pairs | {(target, source) for source, target in rel_pairs}

    source_to_targets = defaultdict(lambda: defaultdict(set))
    event_target_counts = defaultdict(Counter)

    for event_index, ev in enumerate(ocel.get("events", []) or []):
        act = _event_type(ev)
        rels = ev.get("relationships") or []
        incorporated = []
        candidates = []

        for rel in rels:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            otype = id_to_type.get(oid)
            if not oid or not otype or otype in resource_types or otype in reusable_types:
                continue
            upper_quals = {q.upper() for q in _qualifiers(rel)}
            if "INCORPORATE" in upper_quals:
                incorporated.append((oid, otype))
            elif "RESOURCE" not in upper_quals:
                candidates.append((oid, otype))

        if not incorporated or not candidates:
            continue

        event_key = str(ev.get("id") or ev.get("ocel:eid") or event_index)
        for target_oid, target_type in incorporated:
            for source_oid, source_type in candidates:
                if source_oid == target_oid or source_type == target_type:
                    continue
                if source_type in reusable_types or target_type in reusable_types:
                    continue
                if (source_oid, target_oid) not in undirected_pairs:
                    continue
                key = (source_type, target_type, act)
                source_to_targets[key][source_oid].add(target_oid)
                event_target_counts[key][(event_key, source_oid)] += 1

    iterative_by_target = defaultdict(list)
    for (source_type, target_type, act), by_source in sorted(source_to_targets.items()):
        source_counts = [len(target_ids) for target_ids in by_source.values()]
        event_counts = list(event_target_counts.get((source_type, target_type, act), {}).values())
        if not source_counts or not event_counts:
            continue
        if _mean_count(source_counts) <= _mean_count(event_counts):
            continue

        iterative_by_target[target_type].append(
            {
                "source_object_type": source_type,
                "incorporate_activity": act,
                "incorporated_per_iteration_distribution": _prob_distribution(event_counts),
            }
        )

    return {
        target_type: details
        for target_type, details in sorted(iterative_by_target.items())
    }



def _build_iterative_event_targets(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
    reusable_types: Set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    objects = ocel.get("objects", []) or []
    rel_pairs = _object_relationship_pairs(objects)
    object_graph = _build_object_graph(objects, id_to_type, resource_types)

    source_to_targets = defaultdict(lambda: defaultdict(set))
    event_target_counts = defaultdict(Counter)

    for event_index, ev in enumerate(ocel.get("events", []) or []):
        act = _event_type(ev)
        rels = ev.get("relationships") or []
        participants = []
        created_or_incorporated: Set[str] = set()
        resource_oids: Set[str] = set()

        for rel in rels:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            otype = id_to_type.get(oid)
            if not oid or not otype or otype in resource_types or otype in reusable_types:
                continue
            upper_quals = {q.upper() for q in _qualifiers(rel)}
            if "RESOURCE" in upper_quals:
                resource_oids.add(oid)
                continue
            if "CREATE" in upper_quals or "INCORPORATE" in upper_quals:
                created_or_incorporated.add(oid)
            participants.append((oid, otype, upper_quals))

        if len(participants) < 2:
            continue

        event_key = str(ev.get("id") or ev.get("ocel:eid") or event_index)
        for source_oid, source_type, source_quals in participants:
            if source_oid in created_or_incorporated or source_oid in resource_oids:
                continue
            for target_oid, target_type, target_quals in participants:
                if source_oid == target_oid or source_type == target_type:
                    continue
                if source_type in reusable_types or target_type in reusable_types:
                    continue
                if target_oid in created_or_incorporated or target_oid in resource_oids:
                    continue
                if (source_oid, target_oid) not in rel_pairs:
                    continue
                key = (source_type, target_type, act)
                source_to_targets[key][source_oid].add(target_oid)
                event_target_counts[key][(event_key, source_oid)] += 1

    iterative_by_target = defaultdict(list)
    for (source_type, target_type, act), by_source in sorted(source_to_targets.items()):
        source_counts = [
            sum(1 for target_oid in object_graph.get(source_oid, set()) if id_to_type.get(target_oid) == target_type)
            for source_oid in by_source.keys()
        ]
        source_counts = [count for count in source_counts if count > 0]
        event_counts = list(event_target_counts.get((source_type, target_type, act), {}).values())
        if not source_counts or not event_counts:
            continue
        if _mean_count(source_counts) <= _mean_count(event_counts):
            continue

        iterative_by_target[target_type].append(
            {
                "source_object_type": source_type,
                "activity": act,
                "target_count_distribution": _prob_distribution(event_counts),
            }
        )

    return {
        target_type: details
        for target_type, details in sorted(iterative_by_target.items())
    }



def _build_reusable_source_event_targets(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
    rel_targets: Dict[str, List[str]],
    resource_types: Set[str],
    reusable_types: Set[str],
) -> List[Dict[str, Any]]:
    counts = defaultdict(Counter)
    rel_target_sets = {src: set(targets) for src, targets in rel_targets.items()}

    for event_index, ev in enumerate(ocel.get("events", []) or []):
        act = _event_type(ev)
        rels = ev.get("relationships") or []
        by_type: Dict[str, Set[str]] = defaultdict(set)
        qualifiers_by_type: Dict[str, Set[str]] = defaultdict(set)

        for rel in rels:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            otype = id_to_type.get(oid)
            if not oid or not otype or otype in resource_types:
                continue
            by_type[otype].add(oid)
            qualifiers_by_type[otype].update(q.upper() for q in _qualifiers(rel))

        if not by_type:
            continue
        event_key = str(ev.get("id") or ev.get("ocel:eid") or event_index)
        for source_type in sorted(set(by_type) & reusable_types):
            source_oids = by_type.get(source_type, set())
            if not source_oids:
                continue
            for target_type in sorted(rel_target_sets.get(source_type, set())):
                if target_type in reusable_types or target_type in resource_types:
                    continue
                target_quals = qualifiers_by_type.get(target_type, set())
                role = "CREATE" if "CREATE" in target_quals else "INCORPORATE" if "INCORPORATE" in target_quals else None
                if role is None:
                    continue
                target_count = len(by_type.get(target_type, set()))
                if target_count <= 0:
                    continue
                for source_oid in source_oids:
                    counts[(act, source_type, target_type, role)][(event_key, source_oid)] += target_count

    result = []
    for (act, source_type, target_type, role), counter in sorted(counts.items()):
        event_counts = list(counter.values())
        distribution = _prob_distribution(event_counts)
        if not distribution:
            continue
        result.append({
            "event_type": act,
            "source_object_type": source_type,
            "target_object_type": target_type,
            "target_role": role,
            "target_count_distribution": distribution,
        })
    return result

def _build_event_role_to_resources(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    tmp = defaultdict(lambda: defaultdict(set))
    for ev in ocel.get("events", []) or []:
        act = _event_type(ev)
        for rel in ev.get("relationships") or []:
            if not any(q.upper() == "RESOURCE" for q in _qualifiers(rel)):
                continue
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            rtype = id_to_type.get(oid)
            if rtype in resource_types:
                tmp[act][rtype].add(oid)

    return {
        act: {rtype: sorted(ids) for rtype, ids in sorted(by_type.items())}
        for act, by_type in sorted(tmp.items())
    }


def _build_resource_event_participation(
    ocel: Dict[str, Any],
    id_to_type: Dict[str, str],
    resource_types: Set[str],
) -> Dict[str, Dict[str, List[str]]]:
    participation = defaultdict(lambda: defaultdict(set))
    for ev in ocel.get("events", []) or []:
        act = _event_type(ev)
        for rel in ev.get("relationships") or []:
            if not any(q.upper() == "RESOURCE" for q in _qualifiers(rel)):
                continue
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            rtype = id_to_type.get(oid)
            if rtype in resource_types:
                participation[rtype][oid].add(act)

    return {
        rtype: {
            oid: sorted(events)
            for oid, events in sorted(by_id.items())
        }
        for rtype, by_id in sorted(participation.items())
    }


def extract(
    ocel: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    objects = ocel.get("objects", []) or []
    id_to_type = build_object_id_to_type(objects)
    type_to_ids = _build_type_to_ids(objects)
    resource_types = set(_extract_resource_types(ocel, id_to_type))
    object_graph = _build_object_graph(objects, id_to_type, resource_types)
    stats = _object_type_stats(objects, id_to_type, object_graph)
    reusable_types = set(_classify_reusable_types(stats, resource_types))
    rel_targets = _build_rel_targets(objects, id_to_type, resource_types)
    iterative_create_targets = _build_iterative_create_targets(ocel, id_to_type, resource_types, reusable_types)
    iterative_incorporate_targets = _build_iterative_incorporate_targets(
        ocel, id_to_type, resource_types, reusable_types
    )
    iterative_event_targets = _build_iterative_event_targets(
        ocel, id_to_type, resource_types, reusable_types
    )
    reusable_source_event_targets = _build_reusable_source_event_targets(
        ocel, id_to_type, rel_targets, resource_types, reusable_types
    )

    objects_section = {
        "OBJECT_TYPES": sorted(type_to_ids.keys()),
        "RESOURCE_OBJECT_TYPES": sorted(resource_types),
        "REUSABLE_OBJECT_TYPES": sorted(reusable_types),
        "ID_POOLS": {otype: type_to_ids[otype] for otype in sorted(reusable_types)},
        "OBJECT_LIFECYCLE_EVENTS": _extract_object_lifecycle_events(ocel, id_to_type),
        "ITERATIVE_CREATE_TARGET_TYPES": sorted(iterative_create_targets.keys()),
        "ITERATIVE_INCORPORATE_TARGET_TYPES": sorted(iterative_incorporate_targets.keys()),
        "ITERATIVE_EVENT_TARGET_TYPES": sorted(iterative_event_targets.keys()),
    }

    relations_section = {
        "OBJECT_REL_TARGETS": rel_targets,
        "CARDINALITY_DISTRIBUTION": _build_cardinality_probs(
            objects, id_to_type, rel_targets, object_graph, resource_types, reusable_types
        ),
        "REUSABLE_SELECTION_PROB": _build_reusable_selection_prob(
            object_graph, id_to_type, reusable_types, resource_types
        ),
        "ITERATIVE_CREATE_TARGETS": iterative_create_targets,
        "ITERATIVE_INCORPORATE_TARGETS": iterative_incorporate_targets,
        "ITERATIVE_EVENT_TARGETS": iterative_event_targets,
        "REUSABLE_SOURCE_EVENT_TARGETS": reusable_source_event_targets,
    }

    resources_section = {
        "EVENT_ROLE_TO_RESOURCES": _build_event_role_to_resources(ocel, id_to_type, resource_types),
        "RESOURCE_EVENT_PARTICIPATION": _build_resource_event_participation(ocel, id_to_type, resource_types),
    }

    return {
        "objects": objects_section,
        "relations": relations_section,
        "resources": resources_section,
    }


if __name__ == "__main__":
    print("This module is used by simulation_input_builder.py. Run run_pipeline.py to build inputs and simulate OCEL logs.")
