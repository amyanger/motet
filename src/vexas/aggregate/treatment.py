"""Treatment response by line of therapy.

Two things distinguish this from the published meta-analyses:

1. They pool by drug. We condition on sequence position. "Ruxolitinib as
   second line after tocilizumab failure" is a different question from
   "ruxolitinib", and it is the one a treating clinician faces.

2. No consensus response criteria exist in VEXAS -- every paper defines
   "complete response" ad hoc, which is the stated main limitation of the
   published meta-analysis. So we ignore the authors' labels and apply ONE
   definition to the raw components, consistently, across every report.

Our definition, adapted from the momelotinib trial criteria and stated
openly so it can be argued with:
    RESPONDED = symptoms resolved
                AND steroid dose reduced by >= 50%
                AND follow-up duration is stated.
Anything missing a component is INDETERMINATE, never a failure. A report
that does not say is not a report that says no.
"""

from collections import defaultdict
from enum import Enum

from vexas.aggregate.denominators import Count, Pattern
from vexas.normalize.units import parse_numeric
from vexas.schema import PatientRecord, TreatmentCourse

#: Minimum fractional steroid reduction to count as a response.
STEROID_REDUCTION_THRESHOLD = 0.5

RESPONSE_DEFINITION = (
    "Symptoms resolved, steroid dose reduced by at least 50%, and a "
    "follow-up duration stated. Applied uniformly; authors' own response "
    "labels are not used."
)


class ResponseCall(str, Enum):
    RESPONDED = "responded"
    NOT_RESPONDED = "not_responded"
    INDETERMINATE = "indeterminate"


def _resolved(course: TreatmentCourse) -> bool | None:
    field = course.symptom_resolution
    if field is None or not field.value:
        return None
    return field.value.strip().lower() in {"yes", "true", "resolved",
                                           "complete"}


def call_response(course: TreatmentCourse) -> ResponseCall:
    """Apply one consistent response definition to raw components."""
    # Follow-up is required first: a response claim without it is
    # uninterpretable regardless of the other components.
    if course.follow_up_duration is None or not course.follow_up_duration.value:
        return ResponseCall.INDETERMINATE

    resolved = _resolved(course)
    if resolved is None:
        return ResponseCall.INDETERMINATE

    before = course.steroid_dose_before
    after = course.steroid_dose_after
    if before is None or after is None:
        return ResponseCall.INDETERMINATE

    before_value = parse_numeric(before.value or "")
    after_value = parse_numeric(after.value or "")
    if before_value is None or after_value is None or before_value <= 0:
        return ResponseCall.INDETERMINATE

    reduction = (before_value - after_value) / before_value
    if resolved and reduction >= STEROID_REDUCTION_THRESHOLD:
        return ResponseCall.RESPONDED
    return ResponseCall.NOT_RESPONDED


def group_by_line(
    records: list[PatientRecord],
) -> dict[tuple[str, int], list[tuple[str, TreatmentCourse]]]:
    """Group treatment courses by (drug, line of therapy).

    Values carry the source pmid so patterns stay traceable.
    """
    grouped: dict[tuple[str, int], list[tuple[str, TreatmentCourse]]] = (
        defaultdict(list)
    )
    for record in records:
        for course in record.treatments:
            if course.line_of_therapy is None or not course.treatment.value:
                continue
            key = (course.treatment.value.strip().lower(),
                   course.line_of_therapy)
            grouped[key].append((record.pmid, course))
    return dict(grouped)


def rank_by_line(records: list[PatientRecord],
                 min_support: int = 3) -> list[Pattern]:
    """Rank (drug, line) combinations by reports describing a response.

    Indeterminate courses are excluded from numerator AND denominator --
    they can neither support nor refute, and counting them would
    understate every signal.
    """
    patterns = []
    for (drug, line), entries in group_by_line(records).items():
        called = [
            (pmid, call_response(course))
            for pmid, course in entries
            if call_response(course) is not ResponseCall.INDETERMINATE
        ]
        if not called:
            continue
        responders = [pmid for pmid, call in called
                      if call is ResponseCall.RESPONDED]
        if len(responders) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"{drug} at line {line}: response reported",
                count=Count(
                    n=len(responders),
                    denominator=len(called),
                    label=f"{drug}@L{line}",
                    supporting_pmids=sorted(set(responders)),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)
