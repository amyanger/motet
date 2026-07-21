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
from vexas.schema import Certainty, PatientRecord

#: Only findings the report affirms belong in a sequence. Negated,
#: hypothetical and merely possible findings are excluded.
_ADMISSIBLE = {Certainty.POSITIVE}


def _ordered_events(record: PatientRecord):
    events = [
        event
        for event in record.timeline
        if event.sequence_index is not None
        and event.symptom.certainty in _ADMISSIBLE
        and event.symptom.value
    ]
    return sorted(events, key=lambda e: e.sequence_index)


def ordered_pairs(record: PatientRecord) -> set[tuple[str, str]]:
    """All (earlier, later) symptom pairs stated by this record."""
    events = _ordered_events(record)
    return {
        (a.symptom.value, b.symptom.value)
        for a, b in combinations(events, 2)
        if a.sequence_index < b.sequence_index
    }


def first_manifestation(record: PatientRecord) -> str | None:
    events = _ordered_events(record)
    return events[0].symptom.value if events else None


def rank_precedence(records: list[PatientRecord],
                    min_support: int = 3) -> list[Pattern]:
    """Rank 'A precedes B' claims by how many reports independently state
    them.

    Only records carrying ordering information count toward the
    denominator.
    """
    informative = [r for r in records if _ordered_events(r)]
    denominator = len(informative)
    if denominator == 0:
        return []

    support: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in informative:
        for pair in ordered_pairs(record):
            support[pair].append(record.pmid)

    patterns = []
    for (earlier, later), pmids in support.items():
        if len(pmids) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"{earlier} precedes {later}",
                count=Count(
                    n=len(pmids),
                    denominator=denominator,
                    label=f"{earlier}->{later}",
                    supporting_pmids=sorted(set(pmids)),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)
