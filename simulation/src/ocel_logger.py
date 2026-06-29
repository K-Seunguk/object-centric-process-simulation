# ocel2_log.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class OCEL2Log:
    """
    OCEL 2.0 JSON (eventTypes/objectTypes/events/objects).
    Attributes are kept empty by default (per your requirement).
    """
    object_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    objects_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    events_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    _event_seq: Dict[str, int] = field(default_factory=dict)
    _emit_seq: int = 0

    @staticmethod
    def _etype_key(etype: str) -> str:
        return etype.strip().replace(" ", "_")

    def ensure_object_type(self, name: str) -> None:
        if name not in self.object_types:
            self.object_types[name] = {"name": name, "attributes": []}

    def ensure_event_type(self, name: str) -> None:
        if name not in self.event_types:
            self.event_types[name] = {"name": name, "attributes": []}

    def create_object(self, otype: str, oid: Optional[str] = None) -> str:
        if oid is None:
            raise ValueError("create_object requires explicit oid (no implicit defaults).")
        self.ensure_object_type(otype)
        if oid not in self.objects_by_id:
            self.objects_by_id[oid] = {
                "id": oid,
                "type": otype,
                "attributes": [],     # keep empty
                "relationships": [],  # will be filled later
            }
        return oid

    # ✅ 추가: object->object relationship 기록 (중복 방지)
    def add_object_relationship(self, src_oid: str, tgt_oid: str, qualifier: str) -> None:
        if src_oid not in self.objects_by_id:
            raise ValueError(f"add_object_relationship: unknown src oid: {src_oid}")
        if tgt_oid not in self.objects_by_id:
            raise ValueError(f"add_object_relationship: unknown tgt oid: {tgt_oid}")

        rel = {"objectId": tgt_oid, "qualifier": qualifier}
        bucket = self.objects_by_id[src_oid].setdefault("relationships", [])

        # dedup
        for r in bucket:
            if r.get("objectId") == tgt_oid and r.get("qualifier") == qualifier:
                return
        bucket.append(rel)

    def _next_eid(self, etype: str) -> str:
        key = self._etype_key(etype)
        n = self._event_seq.get(key, 0) + 1
        self._event_seq[key] = n
        return f"{key}_{n:05d}"

    def emit_event(
        self,
        etype: str,
        time: datetime,
        attributes: Dict[str, Any],
        relationships: List[Dict[str, str]],
        eid: Optional[str] = None,
    ) -> str:
        self.ensure_event_type(etype)
        if eid is None:
            eid = self._next_eid(etype)

        attr_arr = [{"name": k, "value": v} for k, v in attributes.items()]

        self._emit_seq += 1
        self.events_by_id[eid] = {
            "id": eid,
            "type": etype,
            "time": time.isoformat(),
            "attributes": attr_arr,
            "relationships": list(relationships),
            "_emit_seq": self._emit_seq,
        }
        return eid

    def to_dict(self) -> Dict[str, Any]:
        object_types = [self.object_types[k] for k in sorted(self.object_types)]
        event_types = [self.event_types[k] for k in sorted(self.event_types)]
        objects = [self.objects_by_id[k] for k in sorted(self.objects_by_id)]

        evs = [dict(e) for e in self.events_by_id.values()]
        evs.sort(key=lambda e: (e["time"], e.get("_emit_seq", 0)))
        for e in evs:
            e.pop("_emit_seq", None)

        return {
            "eventTypes": event_types,
            "objectTypes": object_types,
            "events": evs,
            "objects": objects,
        }

    def dump(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class OCELRelationshipBuilder:
    """
    Builds event-object and object-object OCEL relationships using the active
    runtime context. The runtime supplies graph access, event schemas, resource
    object types, and case-root resolution.
    """

    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    def qualifier_for_event_object(self, event_label: str, oid: str, default: str) -> str:
        rt = self.runtime
        try:
            otype = rt.graph.type_of(oid)
        except Exception:
            return default

        quals = rt.event_relationships.get(event_label, {}).get(otype, [])
        if not quals:
            return default

        lower_to_original = {str(q).lower(): str(q) for q in quals}
        default_l = default.lower()
        if default_l in lower_to_original:
            return lower_to_original[default_l]

        for preferred in ("resource", "output", "input", "standard"):
            if preferred in lower_to_original:
                return lower_to_original[preferred]
        return str(quals[0])

    def build_event_relationships(
        self,
        event_label: str,
        consumed_oids: List[str],
        produced_oids: List[str],
        reserved_resources: List[Tuple[str, str]],
        case_root: str,
    ) -> List[Dict[str, str]]:
        rt = self.runtime
        if event_label not in rt.event_participants and event_label not in rt.event_relationships:
            raise ValueError(f"Missing required key: ocel.EVENT_RELATIONSHIPS['{event_label}']")

        req_types_all = list(rt.event_relationships.get(event_label, {}).keys())
        if not req_types_all:
            req_types_all = list(rt.event_participants[event_label])

        if case_root is None:
            case_root = rt._resolve_case_root_for_event(consumed_oids + produced_oids)

        if case_root is None:
            raise ValueError(f"Cannot resolve case_root for event '{event_label}'. OIDs: {consumed_oids + produced_oids}")

        reserved_map: Dict[str, str] = {}
        for _role, rid in reserved_resources:
            reserved_map[rid] = "resource"

        out: List[Dict[str, str]] = []
        seen: Set[str] = set()
        req_types = set(req_types_all)

        def is_requested_type(oid: str) -> bool:
            try:
                return rt.graph.type_of(oid) in req_types
            except Exception:
                return False

        def add_relationship(oid: str, default_qualifier: str) -> None:
            if oid in seen or not is_requested_type(oid):
                return
            seen.add(oid)
            out.append({
                "objectId": oid,
                "qualifier": self.qualifier_for_event_object(event_label, oid, default_qualifier),
            })

        for oid in consumed_oids:
            add_relationship(oid, "input")

        for oid in produced_oids:
            add_relationship(oid, "output")

        for rid, role in reserved_map.items():
            add_relationship(rid, role)

        context_oids = list(dict.fromkeys(consumed_oids + produced_oids))
        for otype in req_types_all:
            current_types_in_out = {rt.graph.type_of(r["objectId"]) for r in out}
            if otype in current_types_in_out or otype in rt.resource_object_types:
                continue

            found: Set[str] = set()
            for anchor_oid in context_oids:
                try:
                    direct = list(rt.graph.parents(anchor_oid, otype)) + list(rt._children_of_type(anchor_oid, otype))
                except Exception:
                    direct = []
                for oid in direct:
                    found.add(oid)

            for oid in sorted(found):
                add_relationship(oid, "input")

        return out

    def update_object_relationships_from_event(self, rels: List[Dict[str, str]]) -> None:
        # Object-to-object relationships are synchronized from the final graph
        # before dumping the OCEL, so event-local updates cannot miss cross-case
        # or reverse-direction graph links.
        return


def sync_object_relationships_from_graph(
    log: OCEL2Log,
    graph: object,
    object_rel_targets: Dict[str, List[str]],
) -> None:
    for obj in log.objects_by_id.values():
        obj["relationships"] = []

    dummy_oids = set(getattr(graph, "dummy_oids", set()))
    for src_oid, src_type in sorted(getattr(graph, "oid_type", {}).items()):
        if src_oid in dummy_oids or src_oid not in log.objects_by_id:
            continue

        for tgt_type in object_rel_targets.get(src_type, []):
            linked: Set[str] = set()
            try:
                linked.update(graph.children(src_oid, tgt_type))
            except Exception:
                pass
            try:
                linked.update(graph.parents(src_oid, tgt_type))
            except Exception:
                pass

            for tgt_oid in sorted(linked):
                if tgt_oid == src_oid or tgt_oid in dummy_oids or tgt_oid not in log.objects_by_id:
                    continue
                log.add_object_relationship(src_oid=src_oid, tgt_oid=tgt_oid, qualifier=tgt_type)
