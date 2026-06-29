import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _sanitize(s: str) -> str:
    """
    Create a safe id prefix from type/activity names.
    Rules:
      - lowercase
      - spaces -> underscore
      - keep [a-z0-9_]
      - collapse consecutive underscores
    """
    s = s.strip().lower().replace(" ", "_")
    out = []
    prev_us = False
    for ch in s:
        ok = ("a" <= ch <= "z") or ("0" <= ch <= "9") or (ch == "_")
        if not ok:
            ch = "_"
        if ch == "_":
            if prev_us:
                continue
            prev_us = True
        else:
            prev_us = False
        out.append(ch)
    s2 = "".join(out).strip("_")
    return s2 if s2 else "x"


@dataclass
class IdFactory:
    """
    Deterministic per-type/per-event counters, with support for ID pools.
    """
    width: int = 5
    rng: random.Random = field(default_factory=lambda: random.Random(42))
    id_pools: Dict[str, List[str]] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)

    def next_object_id(self, otype: str) -> str:
        # If a pool exists for this type, pick one at random
        if otype in self.id_pools and self.id_pools[otype]:
            return self.rng.choice(self.id_pools[otype])

        p = _sanitize(otype)
        n = self.counters.get(f"obj::{p}", 0) + 1
        self.counters[f"obj::{p}"] = n
        return f"{p}_{n:0{self.width}d}"

    def next_event_id(self, activity: str) -> str:
        p = _sanitize(activity)
        n = self.counters.get(f"evt::{p}", 0) + 1
        self.counters[f"evt::{p}"] = n
        return f"{p}_{n:0{self.width}d}"