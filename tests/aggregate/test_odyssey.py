from vexas.aggregate.odyssey import (
    canonical_diagnosis,
    diagnostic_delays,
    rank_misdiagnoses,
)
from vexas.schema import Certainty, GroundedField, PatientRecord


def _f(value, quote="q", certainty=Certainty.POSITIVE):
    return GroundedField(value=value, quote=quote, certainty=certainty)


def _record(pmid, diagnoses, delay=None, delay_unit="years"):
    record = PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1",
                           prior_diagnoses=[_f(d) for d in diagnoses])
    if delay is not None:
        record.demographics["time_to_diagnosis"] = GroundedField(
            value=str(delay), unit_as_written=delay_unit, quote="q"
        )
    return record


def test_aliases_collapse_to_one_canonical_name():
    for raw in ["relapsing polychondritis", "Relapsing Polychondritis", "RP"]:
        assert canonical_diagnosis(raw) == "relapsing polychondritis"


def test_giant_cell_arteritis_aliases_collapse():
    for raw in ["GCA", "giant cell arteritis", "temporal arteritis"]:
        assert canonical_diagnosis(raw) == "giant cell arteritis"


def test_unknown_diagnosis_is_lowercased_not_dropped():
    assert canonical_diagnosis("Some Rare Thing") == "some rare thing"


def test_rank_misdiagnoses_counts_reports():
    records = [_record(str(i), ["relapsing polychondritis"])
               for i in range(4)]
    patterns = rank_misdiagnoses(records, min_support=3)
    assert patterns[0].count.n == 4
    assert "relapsing polychondritis" in patterns[0].title


def test_same_diagnosis_twice_in_one_report_counts_once():
    records = [_record("1", ["RP", "relapsing polychondritis"])]
    records += [_record(str(i), ["RP"]) for i in range(2, 5)]
    patterns = rank_misdiagnoses(records, min_support=3)
    assert patterns[0].count.n == 4


def test_negated_prior_diagnoses_are_excluded():
    record = PatientRecord(pmid="1", patient_id="1-p1", prior_diagnoses=[
        _f("relapsing polychondritis", certainty=Certainty.NEGATED)
    ])
    assert rank_misdiagnoses([record], min_support=1) == []


def test_denominator_counts_only_reports_mentioning_prior_diagnoses():
    with_diagnoses = [_record(str(i), ["RP"]) for i in range(3)]
    silent = [PatientRecord(pmid="9", patient_id="9-p1")]
    patterns = rank_misdiagnoses(with_diagnoses + silent, min_support=3)
    assert patterns[0].count.denominator == 3


def test_diagnostic_delays_normalizes_to_years():
    records = [_record("1", [], delay=24, delay_unit="months")]
    assert diagnostic_delays(records) == [("1", 2.0)]


def test_diagnostic_delays_skips_unparseable():
    records = [_record("1", [], delay="several", delay_unit="years")]
    assert diagnostic_delays(records) == []


def test_diagnostic_delays_skips_unknown_units():
    records = [_record("1", [], delay=3, delay_unit="fortnights")]
    assert diagnostic_delays(records) == []
