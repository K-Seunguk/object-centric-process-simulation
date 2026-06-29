# graph.py
from __future__ import annotations
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Set, Tuple
from .id_factory import IdFactory
from typing import Deque
from collections import deque


def _require(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if k not in cur:
            raise ValueError(f"Missing required key: {path}")
        cur = cur[k]
    return cur


def _parse_relation_key(k: str) -> Tuple[str, str]:
    if "→" not in k:
        raise ValueError(f"Invalid relation key: {k} (expected 'A→B')")
    a, b = k.split("→", 1)
    return a.strip(), b.strip()


def _normalize_dist(d: dict) -> dict:
    """
    Convert different distribution formats to internal {"values": [], "weights": []}.
    Supported:
      - {"values": [1,2], "weights": [0.5, 0.5]}
      - {"1:1": 0.5, "1:2": 0.5} (OCPS v2 schema)
    """
    if "values" in d and "weights" in d:
        return d

    values = []
    weights = []
    for k, w in d.items():
        if isinstance(k, str) and ":" in k:
            try:
                # e.g. "1:2" -> 2
                v = int(k.split(":")[-1])
            except ValueError:
                v = int(k) # fallback
        else:
            v = int(k)
        values.append(v)
        weights.append(float(w))

    return {"values": values, "weights": weights}


@dataclass
class Graph:
    """
    Case-scoped object graph driven ONLY by sim_input.json:

      - relations.cardinality_distribution
      - objects.reusable_object_types + objects.id_pools

    Stores:
      - oid_type: exact object typing
      - out_adj / in_adj: adjacency (directed: src -> tgt)
      - case_objects_by_type[case_root][otype] = {oids}
      - oid_case_roots[oid] = {case_root,...}  (reusable/resources can be shared)

    Dummy bridge (NEW, minimal-change):
      - Keep original Phase1/Phase2 logic.
      - When Phase1 samples k==0 for edge A→B, and B has outgoing edges (B→* exists in cardinality_distribution),
        create a dummy B node and connect A→dummy(B).
      - Then allow Phase1 to expand outgoing edges from dummy(B) as well (so B→C can still be created).
      - Dummy nodes are NOT written to OCEL (no log.create_object).
      - Dummy nodes are excluded from:
          * build_case returned case_index
          * related_oids_for_type collection (but traversal can pass through)
    """

    sim_input: dict
    log: object              # OCEL2Log
    rng: Random
    idf: IdFactory

    reusable_types: Set[str] = field(default_factory=set)
    id_pools: Dict[str, List[str]] = field(default_factory=dict)
    reusable_selection: Dict[str, dict] = field(default_factory=dict)
    reusable_targets_by_src: Dict[str, Set[str]] = field(default_factory=dict, init=False)

    # (src_type, tgt_type) -> dist
    cardinality: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    create_types: Set[str] = field(default_factory=set, init=False)

    # NEW: src_type -> [tgt_type...]
    outgoing_by_src: Dict[str, List[str]] = field(default_factory=dict, init=False)
    global_assignment_relations: Set[Tuple[str, str]] = field(default_factory=set, init=False)
    primary_source_by_shared_target: Dict[str, str] = field(default_factory=dict, init=False)

    oid_type: Dict[str, str] = field(default_factory=dict)

    # adjacency in terms of object ids
    out_adj: Dict[str, List[str]] = field(default_factory=dict)  # parent -> [child...]
    in_adj: Dict[str, List[str]] = field(default_factory=dict)   # child  -> [parent...]

    oid_case_roots: Dict[str, Set[str]] = field(default_factory=dict)
    case_objects_by_type: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)

    # NEW: dummy tracking
    dummy_oids: Set[str] = field(default_factory=set, init=False)
    _dummy_seq: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        objs = _require(self.sim_input, "objects")
        rels = _require(self.sim_input, "relations")

        self.reusable_types = set(objs.get("reusable_object_types", []))
        self.id_pools = dict(objs.get("id_pools", {}))
        lifecycle_events = objs.get("object_lifecycle_events", {})
        self.create_types = {otype for otype, roles in lifecycle_events.items() if roles.get("CREATE")}

        for t in self.reusable_types:
            if t not in self.id_pools or not self.id_pools[t]:
                raise ValueError(f"Reusable object type '{t}' is missing from objects.id_pools.")

        selection = rels.get("reusable_selection_distribution", {})
        for reusable_type, dist in selection.items():
            values = list(dist.keys())
            weights = [float(w) for w in dist.values()]
            if values:
                self.reusable_selection[reusable_type] = {"values": values, "weights": weights}

        rel_targets = rels.get("object_relation_targets", {})
        self.reusable_targets_by_src = {
            src_type: {target_type for target_type in targets if target_type in self.reusable_types}
            for src_type, targets in rel_targets.items()
        }

        card = rels.get("cardinality_distribution", {})
        for k, dist in card.items():
            if isinstance(dist, dict) and "values" not in dist:
                # Nested structure: src -> { tgt -> dist_dict }
                src = k
                for tgt, d in dist.items():
                    self.cardinality[(src, tgt)] = _normalize_dist(d)
            else:
                # Flat structure: "src→tgt" -> dist_dict
                src, tgt = _parse_relation_key(k)
                self.cardinality[(src, tgt)] = _normalize_dist(dist)

        # NEW: outgoing index
        self.outgoing_by_src = {}
        for (src, tgt) in self.cardinality.keys():
            self.outgoing_by_src.setdefault(src, []).append(tgt)
        for src in list(self.outgoing_by_src.keys()):
            self.outgoing_by_src[src] = sorted(set(self.outgoing_by_src[src]))

        self._compute_global_assignment_relations()

    def _compute_global_assignment_relations(self) -> None:
        target_sources: Dict[str, List[str]] = {}
        for src, tgt in self.cardinality.keys():
            target_sources.setdefault(tgt, []).append(src)

        events = self.sim_input.get("events", {})
        participants_by_event = events.get("event_participants", {})
        roles_by_event = events.get("event_object_roles", {})

        for tgt, sources in target_sources.items():
            unique_sources = sorted(set(sources))
            if len(unique_sources) <= 1:
                continue

            create_event_participants: List[Set[str]] = []
            for event_type, roles_by_type in roles_by_event.items():
                target_roles = roles_by_type.get(tgt, []) if isinstance(roles_by_type, dict) else []
                if any(str(role).upper() == "CREATE" for role in target_roles):
                    participants = set(participants_by_event.get(event_type, []))
                    if not participants and isinstance(roles_by_type, dict):
                        participants = set(roles_by_type.keys())
                    create_event_participants.append(participants)

            primary_candidates = [
                src
                for src in unique_sources
                if any(src in participants for participants in create_event_participants)
            ]
            primary = sorted(primary_candidates)[0] if primary_candidates else unique_sources[0]
            self.primary_source_by_shared_target[tgt] = primary

            for src in unique_sources:
                if src != primary:
                    self.global_assignment_relations.add((src, tgt))

    # ---------------- cardinality ----------------
    def _sample_cardinality(self, src_type: str, tgt_type: str) -> int:
        """
        - k == 0 allowed (optional relation)
        - k < 0 error
        """
        key = (src_type, tgt_type)
        if key not in self.cardinality:
            raise ValueError(f"Missing required key: relations.cardinality_distribution['{src_type}→{tgt_type}']")
        dist = self.cardinality[key]
        values = dist["values"]
        weights = dist["weights"]
        if len(values) != len(weights):
            raise ValueError(f"Cardinality '{src_type}→{tgt_type}' values/weights length mismatch.")
        k = int(self.rng.choices(values, weights=weights, k=1)[0])
        if k < 0:
            raise ValueError(f"Sampled negative cardinality for '{src_type}→{tgt_type}': {k}")
        return k  # 0 allowed

    def has_relation(self, src_type: str, tgt_type: str) -> bool:
        return (src_type, tgt_type) in self.cardinality

    def _has_outgoing(self, src_type: str) -> bool:
        """Returns True if src_type has any outgoing edges in the cardinality config."""
        return bool(self.outgoing_by_src.get(src_type))

    def is_dummy(self, oid: str) -> bool:
        return oid in self.dummy_oids

    def parents(self, oid: str) -> List[str]:
        return self.in_adj.get(oid, [])

    def children(self, oid: str) -> List[str]:
        return self.out_adj.get(oid, [])

    def get_all_ancestors(self, oid: str) -> Set[str]:
        """Returns all recursive ancestors (parents, grandparents, etc.) of an oid."""
        visited = set()
        stack = self.parents(oid)
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(self.parents(curr))
        return visited

    def get_all_descendants(self, oid: str) -> Set[str]:
        """Returns all recursive descendants (children, grandchildren, etc.) of an oid."""
        visited = set()
        stack = self.children(oid)
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(self.children(curr))
        return visited

    def get_lowest_common_ancestors(self, oid1: str, oid2: str) -> Set[str]:
        """Returns the set of lowest common ancestors (LCAs) for two objects."""
        if oid1 == oid2: return {oid1}
        anc1 = self.get_all_ancestors(oid1) | {oid1}
        anc2 = self.get_all_ancestors(oid2) | {oid2}
        common = anc1 & anc2
        if not common: return set()
        
        # LCA means an ancestor in the common set that has NO child also in the common set
        lcas = set()
        for cand in common:
            # Check if any child of cand is also in common
            has_child_in_common = any(c in common for c in self.children(cand))
            if not has_child_in_common:
                lcas.add(cand)
        return lcas

    def get_all_related_oids(self, anchor_oid: str, target_type: str, case_root: str) -> Set[str]:
        """
        Returns ALL objects of target_type in the case that are RELATED to anchor_oid
        according to the semantic hierarchy (Ancestry/Siblings/LCA Isolation).
        Used by the runtime to verify Join completeness.
        """
        candidates = self.case_objects_by_type.get(case_root, {}).get(target_type, set())
        results = set()
        # We need check each candidate using the SAME logic as _is_related.
        # This is slightly expensive but crucial for correctness at Join points.
        for cand in candidates:
            if self._is_related_semantic(anchor_oid, cand, case_root):
                results.add(cand)
        return results

    def _is_related_semantic(self, oid1: str, oid2: str, root_oid: str) -> bool:
        """Internal semantic relationship check used by runtime compatibility checks."""
        if oid1 == oid2: return True
        # 1. Vertical
        anc1 = self.get_all_ancestors(oid1)
        if oid2 in anc1: return True
        anc2 = self.get_all_ancestors(oid2)
        if oid1 in anc2: return True
        # 2. Neighbors
        if set(self.parents(oid1)) & set(self.parents(oid2)): return True
        if set(self.children(oid1)) & set(self.children(oid2)): return True
        # 3. LCA
        lcas = self.get_lowest_common_ancestors(oid1, oid2)
        if not lcas: return False
        if any(anc != root_oid for anc in lcas): return True
        if not self.parents(oid1) or not self.parents(oid2): return True
        return False

    # ---------------- registry/membership ----------------
    def _ensure_membership(self, case_root: str, oid: str, otype: str) -> None:
        self.case_objects_by_type.setdefault(case_root, {}).setdefault(otype, set()).add(oid)
        self.oid_case_roots.setdefault(oid, set()).add(case_root)

    def register_existing(self, case_root: str, otype: str, oid: str) -> None:
        self.oid_type[oid] = otype
        self._ensure_membership(case_root, oid, otype)

    def register_shared(self, otype: str, oid: str) -> None:
        # reusable/resource etc.
        self.oid_type[oid] = otype

    def type_of(self, oid: str) -> str:
        if oid not in self.oid_type:
            raise ValueError(f"Unknown oid in Graph registry: {oid}")
        return self.oid_type[oid]

    def case_roots_of(self, oid: str) -> Set[str]:
        return set(self.oid_case_roots.get(oid, set()))

    def case_root_of(self, oid: str) -> str:
        roots = self.oid_case_roots.get(oid)
        if not roots:
            raise ValueError(f"Cannot resolve case_root: oid not registered in any case: {oid}")
        if len(roots) != 1:
            raise ValueError(f"Ambiguous case_root for oid='{oid}': roots={sorted(roots)}")
        return next(iter(roots))

    # ---------------- edges ----------------
    def add_edge(self, case_root: str, parent_oid: str, child_oid: str) -> None:
        self.out_adj.setdefault(parent_oid, []).append(child_oid)
        self.in_adj.setdefault(child_oid, []).append(parent_oid)

        # propagate membership
        ptype = self.type_of(parent_oid)
        ctype = self.type_of(child_oid)
        self._ensure_membership(case_root, parent_oid, ptype)
        self._ensure_membership(case_root, child_oid, ctype)

    def children(self, oid: str, child_type: Optional[str] = None) -> List[str]:
        xs = list(self.out_adj.get(oid, []))
        if child_type is None:
            return xs
        return [x for x in xs if self.type_of(x) == child_type]

    def parents(self, oid: str, parent_type: Optional[str] = None) -> List[str]:
        xs = list(self.in_adj.get(oid, []))
        if parent_type is None:
            return xs
        return [x for x in xs if self.type_of(x) == parent_type]

    # ---------------- reusable selection ----------------
    def _sample_reusable_oid(self, reusable_type: str) -> str:
        dist = self.reusable_selection.get(reusable_type)
        if dist:
            return str(self.rng.choices(dist["values"], weights=dist["weights"], k=1)[0])
        return str(self.rng.choice(self.id_pools[reusable_type]))

    def _attach_reusable_relations(self, case_root: str, src_oid: str, src_type: str) -> None:
        for reusable_type in sorted(self.reusable_targets_by_src.get(src_type, set())):
            if self.children(src_oid, reusable_type):
                continue
            reusable_oid = self._sample_reusable_oid(reusable_type)
            self.log.create_object(reusable_type, oid=reusable_oid)
            self.register_existing(case_root, reusable_type, reusable_oid)
            self.add_edge(case_root, src_oid, reusable_oid)

    # ---------------- object creation ----------------
    def _create_object(self, case_root: str, otype: str) -> str:
        # reusable -> use pool value as oid
        if otype in self.reusable_types:
            oid = self.rng.choice(self.id_pools[otype])
            self.log.create_object(otype, oid=oid)
            self.register_existing(case_root, otype, oid)
            return oid

        oid = self.idf.next_object_id(otype)
        self.log.create_object(otype, oid=oid)
        self.register_existing(case_root, otype, oid)
        self._attach_reusable_relations(case_root, oid, otype)
        return oid

    def _create_dummy(self, case_root: str, otype: str) -> str:
        """
        Dummy object:
          - Graph registry only (for traversal / downstream expansion)
          - NOT created in OCEL log
        """
        n = self._dummy_seq.get(otype, 0) + 1
        self._dummy_seq[otype] = n
        oid = f"__dummy__{otype}__{n:05d}"

        self.register_existing(case_root, otype, oid)
        self.dummy_oids.add(oid)
        return oid

    # ---------------- case build ----------------
    def build_case(self, root_type: str, root_oid: str, case_root_oid: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Keep original Phase1/Phase2 logic, add dummy bridge support.

        Phase 1:
          - same as before, but when k==0 for src→tgt:
              if tgt has outgoing edges and tgt is not reusable -> create dummy tgt and connect
          - additionally, after normal forward generation, expand forward_edges AGAIN on any newly created
            objects of reachable types (including dummy), until saturation.

        Phase 2:
          - unchanged (reverse assignment edges).
        """
        case_root = case_root_oid or root_oid
        self.register_existing(case_root, root_type, root_oid)
        if root_type not in self.reusable_types:
            self._attach_reusable_relations(case_root, root_oid, root_type)

        relations = sorted(rel for rel in self.cardinality.keys() if rel not in self.global_assignment_relations)

        # ---- Phase 1: reachable forward edges (type-level) ----
        reachable_types: Set[str] = {root_type}
        forward_edges: List[Tuple[str, str]] = []

        changed = True
        while changed:
            changed = False
            for src, tgt in relations:
                if src in reachable_types and tgt not in reachable_types:
                    reachable_types.add(tgt)
                    forward_edges.append((src, tgt))
                    changed = True

        # NEW: saturating expansion over forward_edges (handles dummy-created nodes)
        # processed_pairs prevents reprocessing same (src_oid, tgt_type) multiple times
        processed_pairs: Set[Tuple[str, str]] = set()

        # Phase 1: Forward expansion (Parents to Children)
        # cardinality keys are (src_type, tgt_type) tuples, values are dist dicts.
        # Iterate until no more objects can be generated downstream.
        made_progress = True
        while made_progress:
            made_progress = False
            for (src, tgt), dist in self.cardinality.items():
                # If we don't have any source objects of type 'src' yet, we can't create children.
                src_oids = sorted(self.case_objects_by_type[case_root].get(src, set()))
                if not src_oids:
                    continue

                for soid in src_oids:
                    keyp = (soid, tgt)
                    if keyp in processed_pairs:
                        continue

                    k = self._sample_cardinality(src, tgt)

                    existing_targets = [
                        oid
                        for oid in sorted(self.case_objects_by_type[case_root].get(tgt, set()))
                        if oid not in self.dummy_oids
                        and oid not in self.children(soid, tgt)
                        and not any(self.type_of(parent_oid) == src for parent_oid in self.parents(oid))
                    ]
                    if existing_targets and tgt in self.create_types and tgt not in self.reusable_types:
                        take = min(k, len(existing_targets)) if k > 0 else 0
                        if take > 0:
                            selected = self.rng.sample(existing_targets, take)
                            for toid in selected:
                                self.add_edge(case_root, soid, toid)
                            made_progress = True
                    elif k == 0:
                        # Dummy bridge for structural continuity if tgt has further descendants
                        if tgt not in self.reusable_types and self._has_outgoing(tgt):
                            doid = self._create_dummy(case_root, tgt)
                            self.add_edge(case_root, soid, doid)
                            made_progress = True
                    else:
                        for _ in range(k):
                            toid = self._create_object(case_root, tgt)
                            self.add_edge(case_root, soid, toid)
                            made_progress = True
                    
                    processed_pairs.add(keyp)

        # ---- Phase 2: reverse assignment edges (Recursive Ancestor Creation) ----
        # Loop until no more ancestors can be added (saturate upstream)
        phase2_created_sources: Dict[str, Set[str]] = {}
        changed = True
        while changed:
            changed = False
            for src, tgt in relations:
                if (src, tgt) in forward_edges:
                    continue

                tgt_oids = sorted(self.case_objects_by_type[case_root].get(tgt, set()))
                src_oids = sorted(self.case_objects_by_type[case_root].get(src, set()))

                if not tgt_oids:
                    continue

                # NEW: Ensure every target token has at least one parent of the 'src' type.
                # If src_oids already exists, we link tgt_oids to them instead of creating new ones.
                if src_oids:
                    # Logic: Distribute existing targets among existing sources (N:1 or N:M)
                    # For simplicity, we can round-robin or just assign all to the first available source
                    # to satisfy the graph-consistent synchronization.
                    for toid in tgt_oids:
                        # Check if toid already has a parent of style src
                        already_linked = any(p in src_oids for p in self.in_adj.get(toid, []))
                        if not already_linked:
                            # Link to the first available source oid
                            self.add_edge(case_root, src_oids[0], toid)
                    continue

                # Traditional ancestor creation if src doesn't exist yet
                remaining = list(tgt_oids)
                while remaining:
                    k = self._sample_cardinality(src, tgt)
                    if k <= 0:
                        # 0이 뽑혔더라도 남은 자식이 있다면 최소 1개는 할정해야 고아(orphan) 방지 가능
                        k = 1
                    
                    soid = self._create_object(case_root, src)
                    phase2_created_sources.setdefault(src, set()).add(soid)
                    take = min(k, len(remaining))
                    chunk = remaining[:take]
                    remaining = remaining[take:]
                    for toid in chunk:
                        self.add_edge(case_root, soid, toid)
                    changed = True

        # ---- Phase 3: selective forward completion for Phase-2-created sources ----
        # If reverse assignment created a new source type (e.g., invoice_receipt), ensure
        # its own outgoing relations (e.g., invoice_receipt -> payment) are populated,
        # but only when currently missing for that source instance.
        for src_type, src_oids in phase2_created_sources.items():
            outgoing = [(s, t) for (s, t) in relations if s == src_type]
            if not outgoing:
                continue

            for soid in sorted(src_oids):
                for _, tgt in outgoing:
                    if self.children(soid, tgt):
                        continue

                    k = self._sample_cardinality(src_type, tgt)
                    if k == 0:
                        if tgt not in self.reusable_types and self._has_outgoing(tgt):
                            doid = self._create_dummy(case_root, tgt)
                            self.add_edge(case_root, soid, doid)
                        continue

                    for _ in range(k):
                        toid = self._create_object(case_root, tgt)
                        self.add_edge(case_root, soid, toid)

        # return per-type index WITHOUT dummy (중요)
        case_index: Dict[str, List[str]] = {}
        by_type = self.case_objects_by_type.get(case_root, {})
        for t, s in by_type.items():
            real = [oid for oid in s if oid not in self.dummy_oids]
            case_index[t] = sorted(real)
        return case_index

    def _complete_outgoing_relations_for_sources(self, source_oids: Set[str]) -> None:
        processed_pairs: Set[Tuple[str, str]] = set()
        queue: List[str] = sorted(source_oids)

        while queue:
            soid = queue.pop(0)
            if soid in self.dummy_oids or soid not in self.oid_type:
                continue
            src_type = self.type_of(soid)
            roots = sorted(self.oid_case_roots.get(soid, set()))
            if not roots:
                continue

            for tgt in self.outgoing_by_src.get(src_type, []):
                if (src_type, tgt) in self.global_assignment_relations:
                    continue
                keyp = (soid, tgt)
                if keyp in processed_pairs:
                    continue
                processed_pairs.add(keyp)

                if self.children(soid, tgt):
                    continue

                k = self._sample_cardinality(src_type, tgt)
                primary_root = roots[0]
                if k == 0:
                    if tgt not in self.reusable_types and self._has_outgoing(tgt):
                        doid = self._create_dummy(primary_root, tgt)
                        for case_root in roots:
                            self.add_edge(case_root, soid, doid)
                        queue.append(doid)
                    continue

                existing_targets = []
                seen_targets: Set[str] = set()
                for case_root in roots:
                    for oid in sorted(self.case_objects_by_type.get(case_root, {}).get(tgt, set())):
                        if oid in seen_targets:
                            continue
                        if oid in self.dummy_oids or oid in self.children(soid, tgt):
                            continue
                        if any(self.type_of(parent_oid) == src_type for parent_oid in self.parents(oid)):
                            continue
                        existing_targets.append(oid)
                        seen_targets.add(oid)

                if existing_targets and tgt in self.create_types and tgt not in self.reusable_types:
                    selected = existing_targets[: min(k, len(existing_targets))]
                    for toid in selected:
                        for case_root in roots:
                            self.add_edge(case_root, soid, toid)
                        queue.append(toid)
                    continue

                for _ in range(k):
                    toid = self._create_object(primary_root, tgt)
                    for case_root in roots:
                        self.add_edge(case_root, soid, toid)
                    queue.append(toid)

    def apply_global_assignment_relations(self) -> None:
        created_sources: Set[str] = set()
        for src, tgt in sorted(self.global_assignment_relations):
            remaining = [
                oid
                for oid, otype in sorted(self.oid_type.items())
                if otype == tgt
                and oid not in self.dummy_oids
                and not any(self.type_of(parent_oid) == src for parent_oid in self.parents(oid))
            ]

            while remaining:
                k = self._sample_cardinality(src, tgt)
                if k <= 0:
                    k = 1
                chunk = remaining[:k]
                remaining = remaining[k:]
                first_roots = sorted(self.oid_case_roots.get(chunk[0], set()))
                if not first_roots:
                    continue

                soid = self._create_object(first_roots[0], src)
                created_sources.add(soid)
                for toid in chunk:
                    roots = sorted(self.oid_case_roots.get(toid, set()))
                    for case_root in roots:
                        self.add_edge(case_root, soid, toid)

        self._complete_outgoing_relations_for_sources(created_sources)

    def get_real_case_objects_by_type(self, case_root: str) -> Dict[str, List[str]]:
        """
        Returns all real (non-dummy) objects in the case, grouped by type.
        """
        out: Dict[str, List[str]] = {}
        for otype, oids in self.case_objects_by_type.get(case_root, {}).items():
            real = sorted([oid for oid in oids if not self.is_dummy(oid)])
            if real:
                out[otype] = real
        return out

    # ---------------- relationship helper ----------------
    def related_oids_for_type(
        self,
        *,
        anchor_oid: str,
        target_type: str,
        case_root: str,
        include_parent_one_hop: bool,
        include_descendants: bool,
        stop_expand_at_reusable: bool,
    ) -> List[str]:
        """
        Policy:
          - descendants: multi-hop outgoing expansion
          - parents: one-hop incoming only
          - stop expanding at reusable nodes
        NEW:
          - dummy nodes are never collected, but traversal can pass through them
        """
        if case_root not in self.case_objects_by_type:
            raise RuntimeError(f"Internal error: missing case index for case_root={case_root}")

        visited: Set[str] = set()
        collected: Set[str] = set()

        def maybe_collect(x: str) -> None:
            if self.is_dummy(x):
                return
            if self.type_of(x) == target_type:
                collected.add(x)

        visited.add(anchor_oid)
        maybe_collect(anchor_oid)

        if include_parent_one_hop:
            for p in self.parents(anchor_oid):
                if p not in visited:
                    visited.add(p)
                    maybe_collect(p)

        if include_descendants:
            q: List[str] = [anchor_oid]
            while q:
                cur = q.pop(0)
                if stop_expand_at_reusable and self.type_of(cur) in self.reusable_types:
                    continue
                for ch in self.children(cur):
                    if ch in visited:
                        continue
                    visited.add(ch)
                    maybe_collect(ch)
                    q.append(ch)

        return sorted(collected)

    def related_oids_for_type_undirected(
        self,
        *,
        anchor_oid: str,
        target_type: str,
        case_root: str,
        stop_expand_at_reusable: bool,
        max_hops: int = 6,
    ) -> List[str]:
        """
        Undirected BFS over the object graph (children + parents), case-scoped.

        Purpose:
        - cardinality_distribution의 방향이 Petri-net의 cross-type 흐름과 불일치할 때도
            "연결성" 기준으로 target_type oid들을 찾아낼 수 있도록 함.

        Policy:
        - adjacency = children(cur) + parents(cur)  (방향 무시)
        - stop_expand_at_reusable: reusable 타입 노드에서는 더 확장하지 않음
        - max_hops: 과도한 확장/토큰 폭증 방지용 hop 제한
        - dummy 노드는 수집하지 않음(있다면), 단 탐색은 통과 가능

        Notes:
        - case_root는 현재 Graph.case_objects_by_type에 존재해야 함.
        - 반환은 target_type에 해당하는 oid들의 sorted list.
        """
        if case_root not in self.case_objects_by_type:
            raise RuntimeError(f"Internal error: missing case index for case_root={case_root}")

        # dummy 필터(있는 경우에만)
        is_dummy = getattr(self, "is_dummy", None)

        visited: Set[str] = set()
        collected: Set[str] = set()

        def can_collect(x: str) -> bool:
            if is_dummy is not None and is_dummy(x):
                return False
            return self.type_of(x) == target_type

        # BFS queue holds (oid, depth)
        q: Deque[Tuple[str, int]] = deque()
        q.append((anchor_oid, 0))
        visited.add(anchor_oid)

        if can_collect(anchor_oid):
            collected.add(anchor_oid)

        while q:
            cur, d = q.popleft()
            if d >= max_hops:
                continue

            # anchor_oid(d=0)는 reusable이더라도 이웃을 찾아야 하므로 d > 0 조건 추가
            if d > 0 and stop_expand_at_reusable and self.type_of(cur) in getattr(self, "reusable_types", set()):
                continue

            # undirected neighbors = outgoing + incoming
            nbrs = []
            nbrs.extend(self.children(cur))
            nbrs.extend(self.parents(cur))

            for nb in nbrs:
                if nb in visited:
                    continue
                
                # 케이스 간 간섭 방지를 위해 동일 케이스 내 객체만 탐색
                if case_root not in self.oid_case_roots.get(nb, set()):
                    continue

                visited.add(nb)

                if can_collect(nb):
                    collected.add(nb)

                q.append((nb, d + 1))

        return sorted(collected)
