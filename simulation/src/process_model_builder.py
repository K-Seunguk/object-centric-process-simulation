# net.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
import re


def _require(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if k not in cur:
            raise ValueError(f"Missing required key: {path}")
        cur = cur[k]
    return cur


def _norm(s: str) -> str:
    """
    Normalize type/place strings so that:
      - spaces/hyphens -> underscore
      - multiple underscores collapsed
      - lower-cased
    Examples:
      "goods receipt" -> "goods_receipt"
      "purchase_requisition_sink" -> "purchase_requisition_sink"
    """
    s = s.strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def _strip_place_suffix(pid_norm: str) -> str:
    """
    Strip common place suffix patterns to get a base prefix candidate.
    """
    # _source
    x = re.sub(r"_source$", "", pid_norm)
    # _p_123 (your current convention)
    x = re.sub(r"_p_\d+$", "", x)
    # fallback: _p123
    x = re.sub(r"_p\d+$", "", x)
    return x


@dataclass(frozen=True)
class Place:
    pid: str
    object_type: str  # inferred using objects.OBJECT_TYPES


@dataclass(frozen=True)
class Transition:
    tid: str
    label: str | None  # None = tau


@dataclass
class PetriNet:
    places: Dict[str, Place] = field(default_factory=dict)
    transitions: Dict[str, Transition] = field(default_factory=dict)

    # pre/post are in terms of places
    pre: Dict[str, Set[str]] = field(default_factory=dict)   # tid -> {place...}
    post: Dict[str, Set[str]] = field(default_factory=dict)  # tid -> {place...}

    _all_post_places: Set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self._all_post_places = set().union(*self.post.values())

    def is_source_place(self, place: str) -> bool:
        return place in self.places and not any(place in post for post in self.post.values())

    def get_start_places_for_types(self) -> Dict[str, str]:
        """
        Identify the 'entry' place for each object type.
        A place P of type T is a start place if it is an output of a transition 
        that does NOT consume type T as input, OR if it has no incoming arcs.
        """
        type_to_start: Dict[str, str] = {}
        
        # Collect candidates: places with no incoming arcs
        for p, pl in self.places.items():
            if pl.object_type and self.is_source_place(p):
                if pl.object_type not in type_to_start:
                    type_to_start[pl.object_type] = p

        # Better heuristic: first place produced by a cross-type transition
        for tid, post_ps in self.post.items():
            pre_types = {self.places[pre_p].object_type for pre_p in self.pre.get(tid, []) if self.places[pre_p].object_type}
            for post_p in post_ps:
                pt = self.places[post_p].object_type
                if pt and pt not in pre_types:
                    # This transition produces type pt without consuming it. 
                    # Use the 'first' one we find if not already set.
                    if pt not in type_to_start:
                        type_to_start[pt] = post_p
                        
        return type_to_start

    def get_true_sink_places(self) -> List[str]:
        """
        Identify places that have NO outgoing arcs to any transition (Structural Sinks).
        """
        pre_ps = set()
        for pset in self.pre.values():
            pre_ps.update(pset)
            
        sinks = [p for p in self.places if p not in pre_ps]
        return sorted(sinks)

    @staticmethod
    def _place_type(pid: str, object_types: List[str]) -> str:
        """
        Infer object type from place id using sim_input.objects.OBJECT_TYPES.

        Strategy:
          1) Normalize pid and all object_types.
          2) Strip place suffixes (_source/_sink/_p_#) to get a base.
          3) Longest-prefix match among normalized object_types.
          4) Return the ORIGINAL object_type string from OBJECT_TYPES (not normalized),
             so it matches graph/runtime type strings.

        This fixes cases like:
          - "purchase_requisition_sink" being incorrectly inferred as "purchase"
          - types with spaces, e.g., "goods receipt"
        """
        if not object_types:
            raise ValueError("objects.OBJECT_TYPES is empty; cannot infer place object_type.")

        pid_norm = _norm(pid)
        base = _strip_place_suffix(pid_norm)

        norm_to_raw: Dict[str, str] = {_norm(t): t for t in object_types}

        # 1) exact match on base
        if base in norm_to_raw:
            return norm_to_raw[base]

        # 2) longest prefix match on full pid_norm (more robust)
        candidates = sorted(norm_to_raw.keys(), key=len, reverse=True)
        for nt in candidates:
            if pid_norm == nt or pid_norm.startswith(nt + "_"):
                return norm_to_raw[nt]

        raise ValueError(
            f"Cannot infer object_type for place '{pid}'. "
            f"Normalized='{pid_norm}', base='{base}'. "
            f"Known OBJECT_TYPES(norm)={sorted(norm_to_raw.keys())}"
        )

    @classmethod
    def from_sim_input(cls, sim_input: dict) -> "PetriNet":
        pm = _require(sim_input, "process_model")
        objs = _require(sim_input, "objects")

        object_types: List[str] = list(objs.get("object_types") or objs.get("OBJECT_TYPES", []))
        if not object_types:
            raise ValueError("Missing required key or empty: objects.OBJECT_TYPES")

        places_list: List[str] = pm.get("places") or pm["PLACES"]
        trans_map: Dict[str, str | None] = pm.get("transitions") or pm["TRANSITIONS"]
        arcs: List[List[str]] = pm.get("arcs") or pm["ARCS"]

        places: Dict[str, Place] = {}
        for p in places_list:
            places[p] = Place(pid=p, object_type=cls._place_type(p, object_types))

        transitions: Dict[str, Transition] = {}
        for tid, lbl in trans_map.items():
            transitions[tid] = Transition(tid=tid, label=lbl)

        place_ids = set(places.keys())
        trans_ids = set(transitions.keys())

        pre: Dict[str, Set[str]] = {tid: set() for tid in trans_ids}
        post: Dict[str, Set[str]] = {tid: set() for tid in trans_ids}

        for a in arcs:
            if len(a) != 2:
                raise ValueError(f"Arc must be [src,dst]: {a}")
            src, dst = a[0], a[1]

            if src in place_ids and dst in trans_ids:
                pre[dst].add(src)
            elif src in trans_ids and dst in place_ids:
                post[src].add(dst)
            else:
                raise ValueError(
                    f"Arc endpoints must be (place->transition) or (transition->place): {src} -> {dst}"
                )

        return cls(places=places, transitions=transitions, pre=pre, post=post)
