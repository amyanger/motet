import pytest
from pydantic import ValidationError

from vexas.schema import (
    Certainty,
    Experiencer,
    GroundedField,
    PatientRecord,
    Temporality,
    TimelineEvent,
)


def _field(**kwargs):
    defaults = dict(value="fever", quote="recurrent fevers", section="Case")
    defaults.update(kwargs)
    return GroundedField(**defaults)


def test_defaults_are_positive_present_patient():
    f = _field()
    assert f.certainty is Certainty.POSITIVE
    assert f.temporality is Temporality.PRESENT
    assert f.experiencer is Experiencer.PATIENT


def test_quote_is_required():
    with pytest.raises(ValidationError):
        GroundedField(value="fever")


def test_negated_finding_is_a_real_field_carrying_evidence():
    """'The paper said no' is a present field with a supporting quote."""
    f = _field(value="ear chondritis",
               quote="no evidence of ear chondritis",
               certainty=Certainty.NEGATED)
    assert f.certainty is Certainty.NEGATED
    assert f.quote == "no evidence of ear chondritis"


def test_unstated_finding_is_absent_not_negated():
    """'The paper didn't say' is a missing field, never certainty=negated.

    Collapsing these two is the failure the schema exists to prevent:
    authors under-report common features, so silence carries no information
    about true absence.
    """
    record = PatientRecord(pmid="12345678", patient_id="12345678-p1")
    assert record.outcome is None
    assert record.prior_diagnoses == []
    assert record.demographics == {}


def test_hypothetical_certainty_is_representable():
    f = _field(certainty=Certainty.HYPOTHETICAL,
               quote="we considered lymphoma")
    assert f.certainty is Certainty.HYPOTHETICAL


def test_family_member_experiencer_is_representable():
    f = _field(experiencer=Experiencer.FAMILY_MEMBER,
               quote="his father had a similar rash")
    assert f.experiencer is Experiencer.FAMILY_MEMBER


def test_is_grounded_in_delegates_to_gate():
    source = "The patient had recurrent fevers for six months."
    assert _field().is_grounded_in(source) is True
    assert _field(quote="ear chondritis").is_grounded_in(source) is False


def test_unit_is_stored_as_written():
    f = _field(value="12.4", unit_as_written="mg/dL", quote="CRP 12.4 mg/dL")
    assert f.unit_as_written == "mg/dL"


def test_timeline_event_allows_unordered_sequence():
    e = TimelineEvent(
        symptom=_field(value="fever", quote="fevers"),
        sequence_index=None,
        interval_value=None,
        interval_unit=None,
    )
    assert e.sequence_index is None


def test_patient_record_requires_pmid_and_patient_id():
    with pytest.raises(ValidationError):
        PatientRecord(pmid="12345678")
