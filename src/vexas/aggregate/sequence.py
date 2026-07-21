"""Symptom onset ordering -- the headline pattern.

Published VEXAS reviews are cross-sectional: they report what fraction of
patients ever had a feature. None tabulate per-case ordering, because doing
it by hand across ~200 papers is labour nobody spends. This is the gap.

Denominator discipline: a precedence claim is only supported or refuted by
records that state an order. Records with no ordering are excluded from
BOTH numerator and denominator, otherwise the denominator silently inflates
and every pattern looks weaker than the evidence warrants.
"""

from collections import defaultdict
from itertools import combinations

from vexas.aggregate.denominators import Count, Pattern
from vexas.normalize.ontology import search_hpo
from vexas.schema import Certainty, PatientRecord

#: Only findings the report affirms belong in a sequence. Negated,
#: hypothetical and merely possible findings are excluded.
_ADMISSIBLE = {Certainty.POSITIVE}


def canonical_symptom(value: str) -> str:
    """Canonical key for a symptom surface phrase.

    Uses the HPO top-hit label (via search_hpo, DiskCache-backed, free) so
    surface variants collapse; falls back to a whitespace/case-normalized
    surface form when there is no match or the lookup is unavailable. The
    model never emits the identifier -- retrieval proposes, code validates.
    """
    normalized = " ".join((value or "").strip().lower().split())
    if not normalized:
        return normalized
    try:
        hits = search_hpo(normalized, limit=1)
    except Exception:
        return normalized
    return hits[0].label.strip().lower() if hits else normalized


def _ordered_events(record: PatientRecord):
    events = [
        event
        for event in record.timeline
        if event.sequence_index is not None
        and event.symptom.certainty in _ADMISSIBLE
        and event.symptom.value
    ]
    return sorted(events, key=lambda e: e.sequence_index)


def ordered_pairs(record: PatientRecord,
                  canonicalize=lambda s: s) -> set[tuple[str, str]]:
    """All (earlier, later) symptom pairs stated by this record."""
    events = _ordered_events(record)
    return {
        (canonicalize(a.symptom.value), canonicalize(b.symptom.value))
        for a, b in combinations(events, 2)
        if a.sequence_index < b.sequence_index
    }


def first_manifestation(record: PatientRecord) -> str | None:
    events = _ordered_events(record)
    return events[0].symptom.value if events else None


def rank_precedence(records: list[PatientRecord], min_support: int = 3,
                    canonicalize=canonical_symptom) -> list[Pattern]:
    """Rank 'A precedes B' claims by how many reports independently state
    them.

    The denominator for a pair (A, B) is the number of records that placed
    both A and B in some order (either direction) -- those are the only
    records that can support or refute "A precedes B".
    """
    informative = [r for r in records if _ordered_events(r)]
    if not informative:
        return []

    support: dict[tuple[str, str], list[str]] = defaultdict(list)
    coverage: dict[frozenset, set[str]] = defaultdict(set)
    for record in informative:
        for a, b in ordered_pairs(record, canonicalize):
            support[(a, b)].append(record.pmid)
            coverage[frozenset((a, b))].add(record.pmid)

    patterns = []
    for (earlier, later), pmids in support.items():
        if len(pmids) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"{earlier} precedes {later}",
                count=Count(
                    n=len(pmids),
                    denominator=len(coverage[frozenset((earlier, later))]),
                    label=f"{earlier}->{later}",
                    supporting_pmids=sorted(set(pmids)),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)
