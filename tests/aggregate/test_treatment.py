from vexas.aggregate.treatment import (
    ResponseCall,
    call_response,
    group_by_line,
    rank_by_line,
)
from vexas.schema import GroundedField, PatientRecord, TreatmentCourse


def _f(value, quote="q"):
    return GroundedField(value=value, quote=quote)


def _course(drug, line, **kwargs):
    return TreatmentCourse(treatment=_f(drug), line_of_therapy=line, **kwargs)


def test_responded_requires_steroid_reduction_and_resolution():
    course = _course(
        "ruxolitinib", 2,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
        symptom_resolution=_f("yes"),
        follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.RESPONDED


def test_no_followup_is_indeterminate_even_if_it_looks_good():
    """Response without follow-up duration is meaningless."""
    course = _course(
        "ruxolitinib", 2,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
        symptom_resolution=_f("yes"),
        follow_up_duration=None,
    )
    assert call_response(course) is ResponseCall.INDETERMINATE


def test_missing_components_are_indeterminate_not_failure():
    course = _course("tocilizumab", 1, follow_up_duration=_f("6"))
    assert call_response(course) is ResponseCall.INDETERMINATE


def test_insufficient_steroid_reduction_is_not_a_response():
    course = _course(
        "anakinra", 1,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("35"),
        symptom_resolution=_f("yes"), follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.NOT_RESPONDED


def test_symptoms_unresolved_is_not_a_response():
    course = _course(
        "anakinra", 1,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
        symptom_resolution=_f("no"), follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.NOT_RESPONDED


def test_author_label_is_ignored_when_calling_response():
    """The whole point: we apply one definition, not the paper's."""
    course = _course(
        "anakinra", 1,
        author_response_label=_f("complete response"),
        follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.INDETERMINATE


def test_group_by_line_keys_on_drug_and_line():
    records = [
        PatientRecord(pmid="1", patient_id="1-p1",
                      treatments=[_course("ruxolitinib", 2)]),
        PatientRecord(pmid="2", patient_id="2-p1",
                      treatments=[_course("ruxolitinib", 1)]),
    ]
    grouped = group_by_line(records)
    assert set(grouped) == {("ruxolitinib", 2), ("ruxolitinib", 1)}


def test_drug_names_are_lowercased_for_grouping():
    records = [
        PatientRecord(pmid="1", patient_id="1-p1",
                      treatments=[_course("Ruxolitinib", 1)]),
        PatientRecord(pmid="2", patient_id="2-p1",
                      treatments=[_course("ruxolitinib", 1)]),
    ]
    assert list(group_by_line(records)) == [("ruxolitinib", 1)]


def test_courses_without_a_line_are_excluded():
    records = [PatientRecord(pmid="1", patient_id="1-p1",
                             treatments=[_course("ruxolitinib", None)])]
    assert group_by_line(records) == {}


def test_rank_by_line_excludes_indeterminate_from_denominator():
    """Indeterminate courses can neither support nor refute; counting them
    in the denominator would understate every signal."""
    def responded(pmid):
        return PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1", treatments=[
            _course("ruxolitinib", 2,
                    steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
                    symptom_resolution=_f("yes"), follow_up_duration=_f("12"))
        ])

    def indeterminate(pmid):
        return PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1",
                             treatments=[_course("ruxolitinib", 2)])

    records = [responded(str(i)) for i in range(3)]
    records += [indeterminate(f"x{i}") for i in range(10)]
    patterns = rank_by_line(records, min_support=3)
    assert len(patterns) == 1
    assert patterns[0].count.n == 3
    assert patterns[0].count.denominator == 3


def test_rank_by_line_respects_min_support():
    records = [
        PatientRecord(pmid=str(i), patient_id=f"{i}-p1", treatments=[
            _course("ruxolitinib", 2,
                    steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
                    symptom_resolution=_f("yes"), follow_up_duration=_f("12"))
        ])
        for i in range(2)
    ]
    assert rank_by_line(records, min_support=3) == []
