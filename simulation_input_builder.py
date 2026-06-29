from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from . import decision_point_analyzer, object_analyzer, performance_analyzer, process_discoverer
except ImportError:
    import decision_point_analyzer
    import object_analyzer
    import performance_analyzer
    import process_discoverer


def load_ocel(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "objects": data.get("objects") or data.get("ocel:objects") or [],
        "events": data.get("events") or data.get("ocel:events") or [],
        "objectTypes": data.get("objectTypes") or data.get("ocel:objectTypes") or [],
        "eventTypes": data.get("eventTypes") or data.get("ocel:eventTypes") or [],
    }


def build_object_id_to_type(objects: List[Dict[str, Any]]) -> Dict[str, str]:
    id_to_type = {}
    for obj in objects:
        oid = obj.get("id") or obj.get("ocel:oid")
        otype = obj.get("type") or obj.get("ocel:type")
        if oid is not None and otype is not None:
            id_to_type[str(oid)] = str(otype)
    return id_to_type


DATASETS = ["om", "p2p", "logistics"]


def _write_json(path: Path, data: Dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if not path.exists():
        raise FileNotFoundError(f"Failed to save output file: {path}")
    return path.stat().st_size


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "simulation" / "input"


def _build_event_object_iteration(
    iterative_create_targets: Dict[str, List[Dict[str, Any]]],
    iterative_incorporate_targets: Dict[str, List[Dict[str, Any]]],
    iterative_event_targets: Dict[str, List[Dict[str, Any]]],
    reusable_source_event_targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    for target_type, entries in sorted(iterative_create_targets.items()):
        for entry in entries:
            event_type = entry.get("create_activity")
            source_type = entry.get("source_object_type")
            distribution = entry.get("created_per_iteration_distribution") or {"1": 1.0}
            if not event_type or not source_type:
                continue
            result.append({
                "iteration_type": "CREATE",
                "event_type": event_type,
                "source_object_type": source_type,
                "target_object_type": target_type,
                "target_count_distribution": distribution,
            })

    for target_type, entries in sorted(iterative_incorporate_targets.items()):
        for entry in entries:
            event_type = entry.get("incorporate_activity")
            source_type = entry.get("source_object_type")
            distribution = entry.get("incorporated_per_iteration_distribution") or {"1": 1.0}
            if not event_type or not source_type:
                continue
            result.append({
                "iteration_type": "INCORPORATE",
                "event_type": event_type,
                "source_object_type": source_type,
                "target_object_type": target_type,
                "target_count_distribution": distribution,
            })

    seen = {
        (entry["event_type"], entry["source_object_type"], entry["target_object_type"])
        for entry in result
        if entry.get("iteration_type") in {"CREATE", "INCORPORATE"}
    }
    for target_type, entries in sorted(iterative_event_targets.items()):
        for entry in entries:
            event_type = entry.get("activity")
            source_type = entry.get("source_object_type")
            distribution = entry.get("target_count_distribution") or {"1": 1.0}
            if not event_type or not source_type:
                continue
            key = (event_type, source_type, target_type)
            if key in seen:
                continue
            result.append({
                "iteration_type": "LOOP",
                "event_type": event_type,
                "source_object_type": source_type,
                "target_object_type": target_type,
                "target_count_distribution": distribution,
            })

    for entry in sorted(
        reusable_source_event_targets,
        key=lambda e: (str(e.get("event_type") or ""), str(e.get("source_object_type") or ""), str(e.get("target_object_type") or "")),
    ):
        event_type = entry.get("event_type")
        source_type = entry.get("source_object_type")
        target_type = entry.get("target_object_type")
        distribution = entry.get("target_count_distribution") or {"1": 1.0}
        if not event_type or not source_type or not target_type:
            continue
        result.append({
            "iteration_type": "REUSABLE_SOURCE_EVENT_TARGET",
            "event_type": event_type,
            "source_object_type": source_type,
            "target_object_type": target_type,
            "target_role": entry.get("target_role") or "CREATE",
            "target_count_distribution": distribution,
        })

    return result


def _relationship_qualifiers(rel: Dict[str, Any]) -> List[str]:
    qualifier = rel.get("qualifier") or rel.get("ocel:qualifier") or "standard"
    if isinstance(qualifier, list):
        return [str(q).strip() for q in qualifier if str(q).strip()]
    q = str(qualifier).strip()
    return [q] if q else []


def _extract_event_object_roles(ocel: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    id_to_type = build_object_id_to_type(ocel.get("objects", []) or [])
    roles: Dict[str, Dict[str, Set[str]]] = {}

    for event in ocel.get("events", []) or []:
        activity = str(event.get("type") or event.get("ocel:activity") or "unknown")
        by_type = roles.setdefault(activity, {})
        for rel in event.get("relationships") or []:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or "")
            otype = id_to_type.get(oid)
            if not otype:
                continue
            by_type.setdefault(otype, set()).update(_relationship_qualifiers(rel))

    return {
        activity: {
            otype: sorted(qualifiers)
            for otype, qualifiers in sorted(by_type.items())
        }
        for activity, by_type in sorted(roles.items())
    }


def _extract_event_participants(event_object_roles: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    return {
        activity: sorted(by_type.keys())
        for activity, by_type in sorted(event_object_roles.items())
    }


def _resource_context_from_semantics(object_semantics: Dict[str, Any]) -> Dict[str, Any]:
    resource_types = (
        object_semantics.get("objects", {}).get("RESOURCE_OBJECT_TYPES")
        or object_semantics.get("objects", {}).get("resource_object_types")
        or []
    )
    return {"RESOURCE_OBJECT_TYPES": sorted(resource_types)} if resource_types else {}


def _run_analysis_pipeline(
    dataset: str,
    ocel: Dict[str, Any],
    *,
    noise: float,
) -> Dict[str, Dict[str, Any]]:
    object_semantics = object_analyzer.extract(ocel, context={})
    context = _resource_context_from_semantics(object_semantics)
    process_result = process_discoverer.extract(ocel, context=context, noise=noise)
    return {
        "process_model": process_discoverer.petri_to_process_model(process_result["NET"]),
        "object_semantics": object_semantics,
        "performance": performance_analyzer.extract(ocel, context={"process_model": process_result}),
        "decision": decision_point_analyzer.extract(ocel, context={"process_model": process_result}),
    }


def build_simulation_input(
    dataset: str,
    *,
    data_dir: Optional[Path] = None,
    noise: float = 0.3,
) -> Dict[str, Any]:
    data_dir = data_dir or _data_dir()
    ocel_path = data_dir / f"{dataset}.json"
    if not ocel_path.exists():
        raise FileNotFoundError(f"Missing OCEL file: {ocel_path}")

    ocel = load_ocel(str(ocel_path))
    analysis = _run_analysis_pipeline(dataset, ocel, noise=noise)
    process_model = analysis["process_model"]
    object_semantics = analysis["object_semantics"]
    performance = analysis["performance"]
    decision = analysis["decision"]

    objects = object_semantics.get("objects") or {}
    relations = object_semantics.get("relations") or {}
    resources = object_semantics.get("resources") or {}
    event_object_roles = _extract_event_object_roles(ocel)

    iterative_create_targets = relations.get("ITERATIVE_CREATE_TARGETS") or {}
    iterative_incorporate_targets = relations.get("ITERATIVE_INCORPORATE_TARGETS") or {}
    iterative_event_targets = relations.get("ITERATIVE_EVENT_TARGETS") or {}
    reusable_source_event_targets = relations.get("REUSABLE_SOURCE_EVENT_TARGETS") or []
    event_object_iteration = _build_event_object_iteration(
        iterative_create_targets,
        iterative_incorporate_targets,
        iterative_event_targets,
        reusable_source_event_targets,
    )

    return {
        "dataset": dataset,
        "process_model": process_model,
        "objects": {
            "object_types": objects.get("OBJECT_TYPES") or [],
            "resource_object_types": objects.get("RESOURCE_OBJECT_TYPES") or [],
            "reusable_object_types": objects.get("REUSABLE_OBJECT_TYPES") or [],
            "object_lifecycle_events": objects.get("OBJECT_LIFECYCLE_EVENTS") or {},
            "id_pools": objects.get("ID_POOLS") or {},
        },
        "relations": {
            "object_relation_targets": relations.get("OBJECT_REL_TARGETS") or {},
            "cardinality_distribution": relations.get("CARDINALITY_DISTRIBUTION") or {},
            "reusable_selection_distribution": relations.get("REUSABLE_SELECTION_PROB") or {},
        },
        "resources": {
            "event_resource_pools": resources.get("EVENT_ROLE_TO_RESOURCES") or {},
            "resource_event_participation": resources.get("RESOURCE_EVENT_PARTICIPATION") or {},
            "resource_event_duration_distribution": performance.get("RESOURCE_EVENT_DURATION") or {},
        },
        "events": {
            "event_object_roles": event_object_roles,
            "event_participants": _extract_event_participants(event_object_roles),
            "event_object_iteration": event_object_iteration,
        },
        "performance": {
            "event_duration_distribution": performance.get("EVENT_DURATION") or {},
            "arrival_distribution": performance.get("ARRIVAL_STATS") or {},
        },
        "decision_points": {
            "branch_probabilities": decision.get("BRANCH_PROB") or {},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic simulation input JSON files from new qualifier-based analysis outputs."
    )
    parser.add_argument("dataset", nargs="?", default="all", help="om, p2p, logistics, all")
    parser.add_argument("--data-dir", type=Path, default=_data_dir())
    parser.add_argument("--output-dir", "-o", type=Path, default=_default_output_dir())
    parser.add_argument("--noise", type=float, default=0.3, help="Inductive miner noise threshold for process discovery.")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        print(f"[RUN] {dataset}")
        sim_input = build_simulation_input(dataset, data_dir=args.data_dir, noise=args.noise)
        out_path = args.output_dir / f"simulation_input_{dataset}.json"
        size = _write_json(out_path, sim_input)
        print(f"[OK] Saved: {out_path.resolve()} ({size} bytes)")


if __name__ == "__main__":
    main()
