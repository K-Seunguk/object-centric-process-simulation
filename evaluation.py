from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

try:
    from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_algorithm
except ModuleNotFoundError:
    emd_algorithm = None


BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("om", "p2p", "logistics")

Variant = Tuple[str, ...]
CardinalityLabel = str
Relation = Tuple[str, str]
T = TypeVar("T")


def _load_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _excluded_object_types(sim_input: Dict) -> Set[str]:
    objects = sim_input.get("objects", {})
    return set(objects.get("resource_object_types") or objects.get("RESOURCE_OBJECT_TYPES", []))


def _target_object_types(original_path: Path, simulated_path: Path, excluded: Set[str]) -> List[str]:
    original = _load_json(original_path)
    simulated = _load_json(simulated_path)

    original_types = {obj.get("type") for obj in original.get("objects", [])}
    simulated_types = {obj.get("type") for obj in simulated.get("objects", [])}
    target_types = (original_types & simulated_types) - excluded - {None}
    return sorted(target_types)


def _filter_selected(items: Sequence[str], selected: Optional[Iterable[str]]) -> List[str]:
    if not selected:
        return list(items)
    selected_set = set(selected)
    return [item for item in items if item in selected_set]


def filter_top_variants(
    language: Dict[Variant, int],
    threshold: float = 0.95,
) -> Dict[Variant, int]:
    if not language:
        return {}

    total = sum(language.values())
    sorted_lang = sorted(language.items(), key=lambda x: x[1], reverse=True)

    cum = 0
    filtered = {}
    for variant, freq in sorted_lang:
        filtered[variant] = freq
        cum += freq
        if cum / total >= threshold:
            break
    return filtered


def normalize_language(language: Dict[T, int | float]) -> Dict[T, float]:
    total = sum(language.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in language.items()}


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def extract_object_traces_from_ocel_json(
    ocel_json: Dict,
    excluded_object_types: Optional[Iterable[str]] = None,
) -> Dict[str, List[Variant]]:
    excluded = set(excluded_object_types or [])

    object_type = {
        o["id"]: o["type"]
        for o in ocel_json.get("objects", [])
        if o.get("type") not in excluded
    }

    object_events = defaultdict(list)
    for event in ocel_json.get("events", []):
        etype = event["type"]
        etime = _parse_time(event["time"])
        for rel in event.get("relationships", []):
            oid = rel.get("objectId")
            if oid in object_type:
                object_events[oid].append((etime, etype))

    traces = defaultdict(list)
    for oid, evs in object_events.items():
        if not evs:
            continue
        trace = tuple(e[1] for e in sorted(evs, key=lambda x: x[0]))
        traces[object_type[oid]].append(trace)
    return dict(traces)


def traces_to_variant_language(traces: List[Variant]) -> Dict[Variant, int]:
    lang = defaultdict(int)
    for trace in traces:
        lang[trace] += 1
    return dict(lang)


def levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i]
        for j, right_value in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalized_sequence_distance(left: Variant, right: Variant) -> float:
    max_len = max(len(left), len(right), 1)
    return levenshtein_distance(left, right) / max_len


def transport_emd(
    lang1: Dict[T, float],
    lang2: Dict[T, float],
    distance: Callable[[T, T], float],
) -> float:
    lang1 = normalize_language(lang1)
    lang2 = normalize_language(lang2)
    if not lang1 and not lang2:
        return 0.0
    if not lang1 or not lang2:
        return 1.0

    sources = list(lang1.keys())
    targets = list(lang2.keys())
    source_count = len(sources)
    target_count = len(targets)
    source_node = source_count + target_count
    sink_node = source_node + 1
    node_count = sink_node + 1
    graph: List[List[Dict[str, float | int]]] = [[] for _ in range(node_count)]

    def add_edge(u: int, v: int, capacity: float, cost: float) -> None:
        forward = {"to": v, "rev": len(graph[v]), "cap": capacity, "cost": cost}
        backward = {"to": u, "rev": len(graph[u]), "cap": 0.0, "cost": -cost}
        graph[u].append(forward)
        graph[v].append(backward)

    for i, variant in enumerate(sources):
        add_edge(source_node, i, lang1[variant], 0.0)
    for j, variant in enumerate(targets):
        add_edge(source_count + j, sink_node, lang2[variant], 0.0)
    for i, source_variant in enumerate(sources):
        for j, target_variant in enumerate(targets):
            add_edge(i, source_count + j, 1.0, distance(source_variant, target_variant))

    total_flow = min(sum(lang1.values()), sum(lang2.values()))
    flow = 0.0
    cost = 0.0
    potentials = [0.0] * node_count
    eps = 1e-12

    while flow + eps < total_flow:
        dist = [float("inf")] * node_count
        prev_node = [-1] * node_count
        prev_edge = [-1] * node_count
        dist[source_node] = 0.0
        queue = [(0.0, source_node)]

        while queue:
            current_dist, u = heapq.heappop(queue)
            if current_dist > dist[u] + eps:
                continue
            for edge_index, edge in enumerate(graph[u]):
                if float(edge["cap"]) <= eps:
                    continue
                v = int(edge["to"])
                next_dist = current_dist + float(edge["cost"]) + potentials[u] - potentials[v]
                if next_dist + eps < dist[v]:
                    dist[v] = next_dist
                    prev_node[v] = u
                    prev_edge[v] = edge_index
                    heapq.heappush(queue, (next_dist, v))

        if prev_node[sink_node] == -1:
            break

        for node, node_dist in enumerate(dist):
            if node_dist < float("inf"):
                potentials[node] += node_dist

        add_flow = total_flow - flow
        v = sink_node
        while v != source_node:
            u = prev_node[v]
            edge = graph[u][prev_edge[v]]
            add_flow = min(add_flow, float(edge["cap"]))
            v = u

        v = sink_node
        while v != source_node:
            u = prev_node[v]
            edge = graph[u][prev_edge[v]]
            reverse = graph[v][int(edge["rev"])]
            edge["cap"] = float(edge["cap"]) - add_flow
            reverse["cap"] = float(reverse["cap"]) + add_flow
            cost += add_flow * float(edge["cost"])
            v = u

        flow += add_flow

    return cost


def calculate_variant_emd_between_ocels_json(
    json_path_1: str,
    json_path_2: str,
    output_path: Optional[str] = None,
    target_objects: Optional[List[str]] = None,
    excluded_object_types: Optional[Iterable[str]] = None,
    variant_filter: float = 0.95,
) -> Dict[str, Dict[str, float | int | None]]:
    with open(json_path_1, encoding="utf-8") as f:
        ocel1 = json.load(f)
    with open(json_path_2, encoding="utf-8") as f:
        ocel2 = json.load(f)

    traces_1 = extract_object_traces_from_ocel_json(ocel1, excluded_object_types)
    traces_2 = extract_object_traces_from_ocel_json(ocel2, excluded_object_types)
    shared = set(traces_1.keys()) & set(traces_2.keys())

    target_list = [o for o in target_objects if o in shared] if target_objects is not None else sorted(shared)
    results: Dict[str, Dict[str, float | int | None]] = {}

    for obj in target_list:
        lang1_counts = traces_to_variant_language(traces_1.get(obj, []))
        lang2_counts = traces_to_variant_language(traces_2.get(obj, []))
        lang1 = normalize_language(filter_top_variants(lang1_counts, variant_filter))
        lang2 = normalize_language(filter_top_variants(lang2_counts, variant_filter))

        if emd_algorithm is not None:
            emd_value = float(emd_algorithm.apply(lang1, lang2))
        else:
            emd_value = float(transport_emd(lang1, lang2, normalized_sequence_distance))

        results[obj] = {
            "emd": emd_value,
            "original_traces": len(traces_1.get(obj, [])),
            "simulated_traces": len(traces_2.get(obj, [])),
            "original_variants": len(lang1_counts),
            "simulated_variants": len(lang2_counts),
        }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    return results


def _object_type_by_id(ocel_json: Dict, excluded: Set[str]) -> Dict[str, str]:
    return {
        str(obj.get("id")): str(obj.get("type"))
        for obj in ocel_json.get("objects", [])
        if obj.get("id") and obj.get("type") and obj.get("type") not in excluded
    }


def _relations_from_ocel(
    ocel_json: Dict,
    excluded_object_types: Optional[Iterable[str]] = None,
    selected_object_types: Optional[Iterable[str]] = None,
) -> List[Relation]:
    excluded = set(excluded_object_types or [])
    selected = set(selected_object_types or []) if selected_object_types is not None else None
    id_to_type = _object_type_by_id(ocel_json, excluded)

    relations: Set[Relation] = set()
    for obj in ocel_json.get("objects", []):
        source_id = str(obj.get("id")) if obj.get("id") else ""
        source_type = id_to_type.get(source_id)
        if not source_type:
            continue
        for rel in obj.get("relationships", []):
            target_id = str(rel.get("objectId")) if rel.get("objectId") else ""
            target_type = id_to_type.get(target_id)
            if not target_type:
                continue
            if selected is not None and (source_type not in selected or target_type not in selected):
                continue
            relations.add((source_type, target_type))
    return sorted(relations)


def extract_cardinality_distributions_from_ocel_json(
    ocel_json: Dict,
    relations: Iterable[Relation],
    excluded_object_types: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Counter]]:
    excluded = set(excluded_object_types or [])

    objects = ocel_json.get("objects", [])
    id_to_type = _object_type_by_id(ocel_json, excluded)
    object_index = {
        str(obj.get("id")): obj
        for obj in objects
        if obj.get("id") and str(obj.get("id")) in id_to_type
    }

    results: Dict[str, Dict[str, Counter]] = defaultdict(dict)
    for source_type, target_type in sorted(set(relations)):
        if source_type in excluded or target_type in excluded:
            continue

        distribution: Counter = Counter()
        source_ids = sorted(oid for oid, otype in id_to_type.items() if otype == source_type)
        for source_id in source_ids:
            obj = object_index.get(source_id, {})
            related_targets = {
                str(rel.get("objectId"))
                for rel in obj.get("relationships", [])
                if rel.get("objectId") and id_to_type.get(str(rel.get("objectId"))) == target_type
            }
            distribution[f"1:{len(related_targets)}"] += 1
        results[source_type][target_type] = distribution

    return results

def _cardinality_size(label: CardinalityLabel) -> int:
    try:
        return int(label.split(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def normalized_cardinality_distance(left: CardinalityLabel, right: CardinalityLabel) -> float:
    left_value = _cardinality_size(left)
    right_value = _cardinality_size(right)
    scale = max(left_value, right_value, 1)
    return abs(left_value - right_value) / scale


def calculate_cardinality_emd_between_ocels_json(
    json_path_1: str,
    json_path_2: str,
    sim_input_path: str,
    output_path: Optional[str] = None,
    excluded_object_types: Optional[Iterable[str]] = None,
    selected_object_types: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float | int | Dict[str, float]]]]:
    with open(json_path_1, encoding="utf-8") as f:
        ocel1 = json.load(f)
    with open(json_path_2, encoding="utf-8") as f:
        ocel2 = json.load(f)

    excluded = set(excluded_object_types or [])
    relations = _relations_from_ocel(
        ocel1,
        excluded_object_types=excluded,
        selected_object_types=selected_object_types,
    )
    dist1 = extract_cardinality_distributions_from_ocel_json(
        ocel1, relations, excluded_object_types=excluded
    )
    dist2 = extract_cardinality_distributions_from_ocel_json(
        ocel2, relations, excluded_object_types=excluded
    )

    results: Dict[str, Dict[str, Dict[str, float | int | Dict[str, float]]]] = defaultdict(dict)

    for source_type, target_type in relations:
        if source_type in excluded or target_type in excluded:
            continue

        left_counts = dist1.get(source_type, {}).get(target_type, Counter())
        right_counts = dist2.get(source_type, {}).get(target_type, Counter())
        left_lang = normalize_language(left_counts)
        right_lang = normalize_language(right_counts)
        emd_value = transport_emd(left_lang, right_lang, normalized_cardinality_distance)

        results[source_type][target_type] = {
            "emd": float(emd_value),
            "original_sources": int(sum(left_counts.values())),
            "simulated_sources": int(sum(right_counts.values())),
            "original_distribution": dict(left_lang),
            "simulated_distribution": dict(right_lang),
        }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def evaluate_dataset(
    dataset: str,
    variant_filter: float = 0.95,
    process_objects: Optional[Iterable[str]] = None,
    relationship_objects: Optional[Iterable[str]] = None,
) -> Dict:
    original_path = BASE_DIR / "data" / f"{dataset}.json"
    simulated_path = BASE_DIR / "simulation" / "output" / f"simulated_{dataset}.json"
    sim_input_path = BASE_DIR / "simulation" / "input" / f"simulation_input_{dataset}.json"

    sim_input = _load_json(sim_input_path)
    excluded = _excluded_object_types(sim_input)
    available_target_types = _target_object_types(original_path, simulated_path, excluded)
    process_target_types = _filter_selected(available_target_types, process_objects)
    relationship_target_types = _filter_selected(available_target_types, relationship_objects)

    process_trace = calculate_variant_emd_between_ocels_json(
        json_path_1=str(original_path),
        json_path_2=str(simulated_path),
        target_objects=process_target_types,
        excluded_object_types=excluded,
        variant_filter=variant_filter,
    )
    cardinality = calculate_cardinality_emd_between_ocels_json(
        json_path_1=str(original_path),
        json_path_2=str(simulated_path),
        sim_input_path=str(sim_input_path),
        excluded_object_types=excluded,
        selected_object_types=relationship_target_types,
    )

    log_emd = {
        obj: float(metrics["emd"])
        for obj, metrics in sorted(process_trace.items())
    }
    cardinality_emd = {}
    for source_type in sorted(cardinality.keys()):
        for target_type in sorted(cardinality[source_type].keys()):
            key = f"{source_type} -> {target_type}"
            cardinality_emd[key] = float(cardinality[source_type][target_type]["emd"])

    return {
        "process_objects": process_target_types,
        "relationship_objects": relationship_target_types,
        "log_emd": log_emd,
        "cardinality_emd": cardinality_emd,
    }


def evaluate_all(
    dataset_config: Dict[str, Dict[str, Iterable[str]]],
    variant_filter: float = 0.95,
) -> Dict[str, Dict]:
    results = {}
    for dataset, config in dataset_config.items():
        results[dataset] = evaluate_dataset(
            dataset,
            variant_filter=variant_filter,
            process_objects=config.get("process_objects"),
            relationship_objects=config.get("relationship_objects"),
        )
    return results


def main() -> None:
    # Edit this block when you want to choose evaluation objects per OCEL.
    # process_objects: object types used for process/log behavior comparison.
    # relationship_objects: object types used for relationship/cardinality comparison.
    dataset_config = {
        "om": {
            "process_objects": [
                "customers",
                "orders",
                "items",
                "packages",
            ],
            "relationship_objects": [
                "customers",
                "orders",
                "items",
                "packages",
                "products",
            ],
        },
        "p2p": {
            "process_objects": [
                "purchase_requisition",
                "quotation",
                "purchase_order",
                "goods_receipt",
                "invoice_receipt",
                "payment",
            ],
            "relationship_objects": [
                "purchase_requisition",
                "quotation",
                "purchase_order",
                "goods_receipt",
                "invoice_receipt",
                "payment",
            ],
        },
        "logistics": {
            "process_objects": [
                "Customer_Order",
                "Transport_Document",
                "Container",
                "Handling_Unit",
                "Vehicle",
            ],
            "relationship_objects": [
                "Customer_Order",
                "Transport_Document",
                "Container",
                "Handling_Unit",
                "Vehicle",
            ],
        },
    }
    variant_filter = 0.95
    output_path = BASE_DIR / "eval" / "evaluation_results.json"

    results = evaluate_all(
        dataset_config=dataset_config,
        variant_filter=variant_filter,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for dataset, result in results.items():
        print(f"\n[{dataset}]")
        print(f"process objects: {result['process_objects'] or '-'}")
        print("process/log EMD")
        if result["log_emd"]:
            for obj, emd in result["log_emd"].items():
                print(f"  {obj}: {emd:.6f}")
        else:
            print("  -")

        print(f"relationship objects: {result['relationship_objects'] or '-'}")
        print("relationship/cardinality EMD")
        if result["cardinality_emd"]:
            for relation, emd in result["cardinality_emd"].items():
                print(f"  {relation}: {emd:.6f}")
        else:
            print("  -")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
