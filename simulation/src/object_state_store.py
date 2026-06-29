from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple


@dataclass
class Token:
    """
    Token carries an object id, an optional case id, and the most recent visible
    event label used by branch-probability logic.
    """

    oid: str
    case_id: Optional[str] = None
    last_label: Optional[str] = None


@dataclass
class Marking:
    tokens_by_place: Dict[str, List[Token]] = field(default_factory=dict)
    _location_by_token: Dict[Tuple[str, Optional[str]], str] = field(default_factory=dict)
    _locations_by_oid: Dict[str, Set[str]] = field(default_factory=dict)
    _oid_place_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    _oids_by_place: Dict[str, Set[str]] = field(default_factory=dict)
    _tokens_by_place_case: Dict[Tuple[str, Optional[str]], List[Token]] = field(default_factory=dict)
    _tokens_by_place_oid: Dict[Tuple[str, str], List[Token]] = field(default_factory=dict)

    def _index_add(self, place: str, tok: Token) -> None:
        key = (tok.oid, tok.case_id)
        old_place = self._location_by_token.get(key)
        if old_place == place:
            return
        if old_place is not None:
            self._decrement_oid_place(tok.oid, old_place)
            self._remove_indexed_token(self._tokens_by_place_case, (old_place, tok.case_id), tok)
            self._remove_indexed_token(self._tokens_by_place_oid, (old_place, tok.oid), tok)
        self._location_by_token[key] = place
        count_key = (tok.oid, place)
        self._oid_place_counts[count_key] = self._oid_place_counts.get(count_key, 0) + 1
        self._locations_by_oid.setdefault(tok.oid, set()).add(place)
        self._oids_by_place.setdefault(place, set()).add(tok.oid)
        self._tokens_by_place_case.setdefault((place, tok.case_id), []).append(tok)
        self._tokens_by_place_oid.setdefault((place, tok.oid), []).append(tok)

    def _index_remove(self, place: str, tok: Token) -> None:
        key = (tok.oid, tok.case_id)
        if self._location_by_token.get(key) == place:
            self._location_by_token.pop(key, None)
        self._decrement_oid_place(tok.oid, place)
        self._remove_indexed_token(self._tokens_by_place_case, (place, tok.case_id), tok)
        self._remove_indexed_token(self._tokens_by_place_oid, (place, tok.oid), tok)

    def _remove_indexed_token(self, index: Dict[Tuple, List[Token]], key: Tuple, tok: Token) -> None:
        bucket = index.get(key)
        if not bucket:
            return
        try:
            bucket.remove(tok)
        except ValueError:
            return
        if not bucket:
            index.pop(key, None)

    def _decrement_oid_place(self, oid: str, place: str) -> None:
        count_key = (oid, place)
        count = self._oid_place_counts.get(count_key, 0)
        if count > 1:
            self._oid_place_counts[count_key] = count - 1
            return
        self._oid_place_counts.pop(count_key, None)
        places = self._locations_by_oid.get(oid)
        if places is not None:
            places.discard(place)
            if not places:
                self._locations_by_oid.pop(oid, None)
        place_oids = self._oids_by_place.get(place)
        if place_oids is not None:
            place_oids.discard(oid)
            if not place_oids:
                self._oids_by_place.pop(place, None)

    def add(self, place: str, tok: Token) -> None:
        bucket = self.tokens_by_place.setdefault(place, [])
        for existing in bucket:
            if existing.oid == tok.oid and existing.case_id == tok.case_id:
                if tok.last_label is not None:
                    existing.last_label = tok.last_label
                self._index_add(place, existing)
                return

        bucket.append(tok)
        self._index_add(place, tok)

    def has(self, place: str) -> bool:
        return bool(self.tokens_by_place.get(place))

    def tokens(self, place: str) -> List[Token]:
        return list(self.tokens_by_place.get(place, []))

    def tokens_for_case(self, place: str, case_id: Optional[str]) -> List[Token]:
        return [tok for tok in self.tokens_by_place.get(place, []) if tok.case_id == case_id]

    def first_token_for_case(self, place: str, case_id: Optional[str]) -> Optional[Token]:
        return next((tok for tok in self.tokens_by_place.get(place, []) if tok.case_id == case_id), None)

    def first_token_for_oid(self, place: str, oid: str) -> Optional[Token]:
        return next((tok for tok in self.tokens_by_place.get(place, []) if tok.oid == oid), None)

    def oids(self, place: str) -> Set[str]:
        return {tok.oid for tok in self.tokens_by_place.get(place, [])}

    def pop_matching(self, place: str, pred: Callable[[Token], bool]) -> Token:
        bucket = self.tokens_by_place.get(place, [])
        for idx, token in enumerate(bucket):
            if pred(token):
                tok = bucket.pop(idx)
                self._index_remove(place, tok)
                return tok
        raise RuntimeError(f"No matching token at place: {place}")

    def contains_oid(self, place: str, oid: str) -> bool:
        return any(token.oid == oid for token in self.tokens_by_place.get(place, []))

    def get_object_location(self, oid: str, case_id: Optional[str] = None) -> Optional[str]:
        if case_id is not None:
            place = self._location_by_token.get((oid, case_id))
            if place is not None:
                return place

        places = self._locations_by_oid.get(oid)
        if not places:
            return None
        if len(places) == 1:
            return next(iter(places))

        for place, bucket in self.tokens_by_place.items():
            for token in bucket:
                if token.oid == oid:
                    return place
        return None

    def all_oids(self) -> Set[str]:
        oids: Set[str] = set()
        for bucket in self.tokens_by_place.values():
            for token in bucket:
                oids.add(token.oid)
        return oids


@dataclass(order=True)
class _Completion:
    time: datetime
    seq: int
    tid: str = field(compare=False)
    start_time: datetime = field(compare=False)
    consumed: List[Tuple[str, Token]] = field(compare=False)
    reserved_resources: List[Tuple[str, str]] = field(compare=False)


@dataclass
class CaseLifecycleState:
    done_oids_by_case: Dict[str, Set[str]] = field(default_factory=dict)
    born_oids_by_case: Dict[str, Set[str]] = field(default_factory=dict)
    completed_cases: Set[str] = field(default_factory=set)
    cases_to_check: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.done_oids_by_case.clear()
        self.born_oids_by_case.clear()
        self.completed_cases.clear()
        self.cases_to_check.clear()

    def mark_born(self, case_root: Optional[str], oid: str) -> None:
        if case_root is None:
            return
        self.born_oids_by_case.setdefault(case_root, set()).add(oid)

    def mark_done(self, case_root: Optional[str], oid: str) -> None:
        if case_root is None:
            return
        self.done_oids_by_case.setdefault(case_root, set()).add(oid)

    def request_check(self, case_id: Optional[str]) -> None:
        if case_id is not None:
            self.cases_to_check.add(case_id)

    def cases_due_for_check(self, arrived_roots: List[str]) -> Set[str]:
        candidates = self.cases_to_check & set(arrived_roots)
        self.cases_to_check.difference_update(candidates)
        return candidates

    def mark_completed(self, case_id: str) -> None:
        self.completed_cases.add(case_id)

    def is_completed(self, case_id: str) -> bool:
        return case_id in self.completed_cases


__all__ = ["_Completion", "CaseLifecycleState", "Marking", "Token"]
