from __future__ import annotations

import pm4py
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils, reduction
from typing import Dict, Tuple, List, Any, Set, Optional
from collections import defaultdict
from pathlib import Path
import json
from datetime import datetime, timezone

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


# ============================================================
# 1) Preprocessing & Boundary Classification
# ============================================================

def _qualifiers(rel: Dict[str, Any]) -> Set[str]:
    qualifier = rel.get("qualifier") or rel.get("ocel:qualifier") or "standard"
    if isinstance(qualifier, list):
        return {str(q).strip().upper() for q in qualifier if str(q).strip()}
    return {str(qualifier).strip().upper()} if str(qualifier).strip() else set()


def _extract_initiate_types(quals: Dict[Tuple[str, str], Set[str]]) -> Set[str]:
    return {otype for (_, otype), qset in quals.items() if "INITIATE" in qset}


def analyze_qualifiers(ocel: Dict[str, Any], excluded_types: Set[str]) -> Tuple[List[str], Dict[Tuple[str, str], Set[str]]]:
    objects = ocel.get("objects", [])
    id_to_type = build_object_id_to_type(objects)
    quals = defaultdict(set)
    type_quals = defaultdict(set)
    all_known_types = set(id_to_type.values())

    for event in ocel.get("events", []):
        act = str(event.get("type") or event.get("ocel:activity") or "").strip()
        for rel in event.get("relationships", []) or []:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or rel.get("id") or "")
            otype = id_to_type.get(oid)
            if not otype:
                continue
            rel_quals = _qualifiers(rel)
            if not rel_quals:
                continue
            quals[(act, otype)].update(rel_quals)
            type_quals[otype].update(rel_quals)

    selected = []
    for otype in sorted(all_known_types):
        if otype in excluded_types:
            continue
        qset = type_quals.get(otype, set())
        if "RESOURCE" in qset:
            continue
        non_create = qset - {"CREATE"}
        if not non_create:
            continue
        selected.append(otype)

    return selected, dict(quals)

# ============================================================
# 2) Log Extraction (Exclude Boundaries)
# ============================================================

def extract_lifecycle_logs(ocel: Dict[str, Any], selected_types: List[str], quals: Dict[Tuple[str, str], Set[str]]):
    objects = ocel.get("objects", [])
    id_to_type = build_object_id_to_type(objects)
    selected = set(selected_types)
    type_to_rows = defaultdict(list)

    for event in ocel.get("events", []):
        act = str(event.get("type") or event.get("ocel:activity") or "").strip()
        ts_raw = event.get("timestamp") or event.get("ocel:timestamp") or event.get("time")
        dt = parse_ocel_time(ts_raw)
        if not dt:
            continue
        seen_oids = set()
        for rel in event.get("relationships", []) or []:
            oid = str(rel.get("objectId") or rel.get("ocel:oid") or rel.get("id") or "")
            otype = id_to_type.get(oid)
            if otype not in selected or oid in seen_oids:
                continue
            q_set = quals.get((act, otype), set())
            if q_set & {"CREATE", "INCORPORATE", "RESOURCE"}:
                seen_oids.add(oid)
                continue
            type_to_rows[otype].append((oid, act, dt))
            seen_oids.add(oid)

    logs = {}
    for otype, rows in type_to_rows.items():
        rows.sort(key=lambda x: (x[0], x[2]))
        logs[otype] = rows
    return logs

# ============================================================
# 3) Discovery & Dummy Logic
# ============================================================

def _rows_to_event_log(rows: List[Tuple[str, str, Any]]) -> EventLog:
    by_case = defaultdict(list)
    for oid, activity, timestamp in rows:
        by_case[oid].append((activity, timestamp))

    event_log = EventLog()
    for oid, events in sorted(by_case.items()):
        trace = Trace()
        trace.attributes["concept:name"] = oid
        for activity, timestamp in sorted(events, key=lambda item: item[1]):
            trace.append(Event({"concept:name": activity, "time:timestamp": timestamp}))
        event_log.append(trace)
    return event_log


def discover_net(rows: List[Tuple[str, str, Any]], otype: str, noise: float) -> Tuple[PetriNet, Marking, Marking]:
    """Discover a Petri net for a specific object type."""
    if not rows:
        net = PetriNet(f"net_{otype}")
        p = PetriNet.Place(f"{otype}_source")
        net.places.add(p)
        return net, Marking({p: 1}), Marking({p: 1})

    event_log = _rows_to_event_log(rows)

    # Use Inductive Miner
    net, im, fm = pm4py.discover_petri_net_inductive(event_log, noise_threshold=noise)
    
    # Apply simple reduction to clean up redundant structures
    try:
        reduction.apply_simple_reduction(net)
    except Exception:
        pass
    
    return net, im, fm

# ============================================================
# 4) Merging
# ============================================================

def merge_into_ocpn(
    infos: Dict[str, Dict[str, Any]],
    quals: Dict[Tuple[str, str], Set[str]],
    selected_types: List[str],
    initiate_types: Optional[Set[str]] = None,
) -> PetriNet:
    """Merge individual object-type nets into a unified Object-Centric Petri Net."""
    merged = PetriNet("FINAL_OCPN")
    vis_map = {}

    def get_transition(label):
        if label not in vis_map:
            t = PetriNet.Transition(f"act__{label}", label)
            merged.transitions.add(t)
            vis_map[label] = t
        return vis_map[label]

    start_ps_by_type = defaultdict(set)
    end_ps_by_type = defaultdict(set)

    # 1) Add individual net elements with consistent naming
    for ot, info in infos.items():
        net = info["net"]
        p_start_orig = list(info["im"].keys())[0] if info["im"] else None
        p_end_orig = list(info["fm"].keys())[0] if info["fm"] else None
        
        p_map = {}
        for p in net.places:
            if p == p_start_orig:
                new_name = f"{ot}_source"
            elif p == p_end_orig:
                new_name = f"{ot}_sink"
            else:
                new_name = f"{ot}_{p.name}"
            
            p_new = PetriNet.Place(new_name)
            merged.places.add(p_new)
            p_map[p] = p_new
            
            if p == p_start_orig: start_ps_by_type[ot].add(p_new)
            if p == p_end_orig: end_ps_by_type[ot].add(p_new)
        
        hid_map = {} # for silent transitions
        for t in net.transitions:
            if t.label:
                t_new = get_transition(t.label)
            else:
                # Use object type prefix for silent transitions to avoid accidental merges
                t_new = PetriNet.Transition(f"{ot}__{t.name}", None)
                merged.transitions.add(t_new)
                hid_map[t] = t_new
        
        for a in net.arcs:
            source_node = p_map[a.source] if isinstance(a.source, PetriNet.Place) else (get_transition(a.source.label) if a.source.label else hid_map[a.source])
            target_node = p_map[a.target] if isinstance(a.target, PetriNet.Place) else (get_transition(a.target.label) if a.target.label else hid_map[a.target])
            if not any(aa.source == source_node and aa.target == target_node for aa in merged.arcs):
                petri_utils.add_arc_from_to(source_node, target_node, merged)

    initiate_types = set(initiate_types or set())
    for otype in sorted(initiate_types):
        if not start_ps_by_type.get(otype):
            p_new = PetriNet.Place(f"{otype}_source")
            merged.places.add(p_new)
            start_ps_by_type[otype].add(p_new)

    # 2) Link creation/consumption boundaries based on qualifiers
    selected_type_set = set(selected_types)
    for (act_lbl, otype), qs in quals.items():
        if otype not in selected_type_set and "INITIATE" not in qs:
            continue
        t_act = get_transition(act_lbl)
        if "INCORPORATE" in qs:
            for p_end in end_ps_by_type[otype]:
                if not any(aa.source == p_end and aa.target == t_act for aa in merged.arcs):
                    petri_utils.add_arc_from_to(p_end, t_act, merged)
        if "CREATE" in qs:
            for p_start in start_ps_by_type[otype]:
                if not any(aa.source == t_act and aa.target == p_start for aa in merged.arcs):
                    petri_utils.add_arc_from_to(t_act, p_start, merged)
        if "INITIATE" in qs:
            for p_start in start_ps_by_type[otype]:
                if not any(aa.source == p_start and aa.target == t_act for aa in merged.arcs):
                    petri_utils.add_arc_from_to(p_start, t_act, merged)

    # 3) Parallelization (v32.2): Force AND-split for object-type entry points if they have multiple visible successors
    for otype in sorted(set(selected_types) | set(initiate_types or set())):
        for p_start in list(start_ps_by_type[otype]):
            if p_start not in merged.places: continue
            
            # Find visible transitions (with labels) directly following p_start
            successors = sorted([a.target for a in p_start.out_arcs if a.target.label is not None], key=lambda x: x.name)
            
            if len(successors) > 1:
                # Transform XOR into AND-split
                # p_start -> {t1, t2}  =>  p_start -> t_split -> {p_sub1, p_sub2} -> {t1, t2}
                t_split = PetriNet.Transition(f"split__{otype}_{p_start.name}", None)
                merged.transitions.add(t_split)
                petri_utils.add_arc_from_to(p_start, t_split, merged)
                
                for i, t_succ in enumerate(successors):
                    p_sub = PetriNet.Place(f"{p_start.name}_p{i}")
                    merged.places.add(p_sub)
                    petri_utils.add_arc_from_to(t_split, p_sub, merged)
                    
                    # Reroute arc outputting from p_start to t_succ
                    for a in list(p_start.out_arcs):
                        if a.target == t_succ:
                            petri_utils.add_arc_from_to(p_sub, t_succ, merged)
                            petri_utils.remove_arc(merged, a)
                
                # Handle any remaining hidden transitions of p_start
                for a in list(p_start.out_arcs):
                    if a.target == t_split: continue
                    p_rem = PetriNet.Place(f"{p_start.name}_rem_{a.target.name}")
                    merged.places.add(p_rem)
                    petri_utils.add_arc_from_to(t_split, p_rem, merged)
                    petri_utils.add_arc_from_to(p_rem, a.target, merged)
                    petri_utils.remove_arc(merged, a)

    return merged

    return merged

def export_net_to_json(net: PetriNet, output_path: str):
    """Export the Petri net structure to an integrated JSON format."""
    data = {
        "places": [{"name": p.name} for p in sorted(net.places, key=lambda x: x.name)],
        "transitions": [],
        "arcs": []
    }
    
    for t in sorted(net.transitions, key=lambda x: (x.label or "", x.name)):
        t_info = {
            "name": t.name,
            "label": t.label,
            "in_arcs": [a.source.name for a in t.in_arcs],
            "out_arcs": [a.target.name for a in t.out_arcs]
        }
        data["transitions"].append(t_info)
        
    for a in sorted(net.arcs, key=lambda x: (x.source.name, x.target.name)):
        data["arcs"].append({
            "source": a.source.name,
            "target": a.target.name
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ============================================================
# 5) Main
# ============================================================


def petri_to_process_model(net: PetriNet) -> Dict[str, Any]:
    """Convert PetriNet object to internal process model format (JSON serializable)."""
    return {
        "PLACES": sorted([str(p.name) for p in net.places]),
        "TRANSITIONS": {str(t.name): t.label for t in net.transitions},
        "ARCS": [[str(a.source.name), str(a.target.name)] for a in net.arcs]
    }

def extract(ocel: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None, noise: float = 0.3) -> Dict[str, Any]:
    """Main extraction point for Control Flow (OCPN)."""
    context = context or {}
    objs_section = context.get("objects") or context.get("OBJECTS_SECTION") or context
    res_types = set(objs_section.get("RESOURCE_OBJECT_TYPES", []))
    reusable_types = set(objs_section.get("REUSABLE_OBJECT_TYPES", []))

    if not reusable_types and extract_object_semantics is not None:
        try:
            semantics = extract_object_semantics(ocel)
            objects_out = semantics.get("objects", {})
            reusable_types = set(objects_out.get("REUSABLE_OBJECT_TYPES", []))
            res_types.update(objects_out.get("RESOURCE_OBJECT_TYPES", []))
        except Exception as exc:
            print(f"[WARN] Object semantics unavailable for reusable/resource exclusion: {exc}")

    selected_types, quals = analyze_qualifiers(ocel, res_types | reusable_types)
    initiate_types = _extract_initiate_types(quals) & reusable_types
    logs = extract_lifecycle_logs(ocel, selected_types, quals)

    infos = {}
    for ot in selected_types:
        rows = logs.get(ot, [])
        net, im, fm = discover_net(rows, ot, noise)
        infos[ot] = {"net": net, "im": im, "fm": fm}

    merged_net = merge_into_ocpn(infos, quals, selected_types, initiate_types=initiate_types)

    return {
        "NET": merged_net,
        "process_model": petri_to_process_model(merged_net),
        "selected_types": sorted(selected_types),
        "initiate_types": sorted(initiate_types),
        "reusable_types": sorted(reusable_types),
        "resource_types": sorted(res_types),
    }

def print_model_structure(net: PetriNet):
    print("\n" + "="*80)
    print("OBJECT-CENTRIC PETRI NET (OCPN) STRUCTURE REPORT")
    print("="*80)
    
    print(f"\n[PLACES] Total: {len(net.places)}")
    for p in sorted(net.places, key=lambda x: str(x.name)):
        print(f"  - {p.name}")
        
    print(f"\n[TRANSITIONS & ARCS] Total: {len(net.transitions)}")
    for t in sorted(net.transitions, key=lambda x: (str(x.label or "TAU"), str(x.name))):
        lbl = f" '{t.label}'" if t.label else " [tau]"
        print(f"\n* {t.name}{lbl}")
        for a in sorted(t.in_arcs, key=lambda x: str(x.source.name)):
            print(f"    <- [IN]  {a.source.name}")
        for a in sorted(t.out_arcs, key=lambda x: str(x.target.name)):
            print(f"    -> [OUT] {a.target.name}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("This module is used by simulation_input_builder.py. Run run_pipeline.py to build inputs and simulate OCEL logs.")
