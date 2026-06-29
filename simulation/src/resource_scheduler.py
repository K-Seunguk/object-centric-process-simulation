from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from random import Random
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ResourceScheduler:
    event_relationships: Dict[str, Dict[str, List[str]]]
    event_participants: Dict[str, List[str]]
    resource_object_types: Set[str]
    event_role_to_resources: Dict[str, Dict[str, List[str]]]
    resource_event_duration: Dict[str, Dict[str, Dict[str, float]]]
    rng: Random
    busy_until: Dict[Tuple[str, str], datetime] = field(default_factory=dict)
    _required_cache: Dict[str, Tuple[str, ...]] = field(default_factory=dict, init=False)

    def required_resource_types(self, event_label: str) -> List[str]:
        cached = self._required_cache.get(event_label)
        if cached is not None:
            return list(cached)

        if event_label in self.event_relationships:
            rels = self.event_relationships.get(event_label, {})
            required = tuple(sorted([
                otype for otype, quals in rels.items()
                if otype in self.resource_object_types and "resource" in {str(q).lower() for q in quals}
            ]))
            self._required_cache[event_label] = required
            return list(required)

        if event_label not in self.event_participants:
            raise ValueError(f"Missing required key: ocel.EVENT_PARTICIPANTS['{event_label}']")

        required = tuple(sorted([
            otype for otype in self.event_participants[event_label]
            if otype in self.resource_object_types
        ]))
        self._required_cache[event_label] = required
        return list(required)

    def try_acquire(
        self,
        now: datetime,
        event_label: str,
    ) -> Tuple[Optional[List[Tuple[str, str]]], Optional[datetime]]:
        reserved: List[Tuple[str, str]] = []
        next_avail: Optional[datetime] = None
        event_resources = self.event_role_to_resources.get(event_label, {})

        for rtype in self.required_resource_types(event_label):
            candidates = event_resources.get(rtype, [])
            if not candidates:
                continue

            available = [
                rid for rid in candidates
                if self.busy_until.get((rtype, rid), now) <= now + timedelta(milliseconds=1)
            ]
            if not available:
                busy_times = [self.busy_until.get((rtype, rid), now) for rid in candidates]
                t_min = min(busy_times) if busy_times else now
                if next_avail is None or t_min < next_avail:
                    next_avail = t_min
                return None, next_avail

            reserved.append((rtype, self.rng.choice(available)))

        return reserved, None


    def sample_duration_seconds(
        self,
        event_label: str,
        reserved: List[Tuple[str, str]],
        fallback_seconds: float,
    ) -> float:
        durations: List[float] = []
        for rtype, rid in reserved:
            cfg = (
                self.resource_event_duration
                .get(rtype, {})
                .get(rid, {})
                .get(event_label)
            )
            if not cfg:
                continue
            mean = cfg.get("mean")
            variance = cfg.get("variance", 0.0)
            if mean is None:
                continue
            mean = float(mean)
            std = math.sqrt(max(0.0, float(variance)))
            if std <= 0:
                sample = mean
            else:
                sample = 0.0
                for _ in range(1000):
                    sample = self.rng.gauss(mean, std)
                    if sample > 0:
                        break
                if sample <= 0:
                    sample = mean
            durations.append(max(0.0, float(sample)))

        if durations:
            return max(durations)
        return fallback_seconds

    def set_busy(self, reserved: List[Tuple[str, str]], busy_until: datetime) -> None:
        for rtype, rid in reserved:
            self.busy_until[(rtype, rid)] = busy_until
