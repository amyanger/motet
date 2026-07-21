from vexas.schema import (
    Certainty,
    GroundedField,
    LabValue,
    PatientRecord,
    TimelineEvent,
    TreatmentCourse,
)
from vexas.verify import Verdict, VerifyStats, build_verify_prompt, verify_field, verify_record

SOURCE = "The patient had no evidence of ear chondritis at presentation."


def _field():
    return GroundedField(value="ear chondritis",
                         quote="no evidence of ear chondritis",
                         certainty=Certainty.NEGATED)


def _flat(field_name="symptom", field=None, source=SOURCE) -> str:
    blocks = build_verify_prompt(field_name, field or _field(), source)
    return "\n".join(b["text"] for b in blocks)


def test_prompt_contains_source_and_candidate():
    prompt = _flat()
    assert SOURCE in prompt
    assert "ear chondritis" in prompt


def test_prompt_does_not_leak_extraction_reasoning():
    """Independence between passes is the mechanism. Guard it."""
    lowered = _flat().lower()
    assert "step by step" not in lowered
    assert "you previously" not in lowered
    assert "extraction prompt" not in lowered


def test_prompt_asks_about_assertion_axes():
    assert "certainty" in _flat().lower()


def test_source_precedes_the_claim_and_carries_the_breakpoint():
    """Caching: ~30 verify calls per report share this prefix."""
    blocks = build_verify_prompt("symptom", _field(), SOURCE)
    source_index = next(i for i, b in enumerate(blocks) if SOURCE in b["text"])
    claim_index = next(
        i for i, b in enumerate(blocks) if b["text"].startswith("CLAIM:")
    )
    assert source_index < claim_index
    cached = [b for b in blocks if "cache_control" in b]
    assert len(cached) == 1 and SOURCE in cached[0]["text"]


def test_cached_prefix_is_identical_across_different_claims():
    a = build_verify_prompt("symptom", _field(), SOURCE)
    b = build_verify_prompt("treatment", _field(), SOURCE)
    assert a[:2] == b[:2]


def test_supported_verdict_is_parsed():
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"supported": True,
                                                    "reason": "explicit"})
    assert verdict.supported is True
    assert verdict.reason == "explicit"


def test_unsupported_verdict_is_parsed():
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"supported": False,
                                                    "reason": "wrong certainty"})
    assert verdict.supported is False


def test_missing_supported_key_defaults_to_unsupported():
    """Fail closed: an unparseable verdict must not silently pass."""
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"reason": "unclear"})
    assert verdict.supported is False


def test_api_failure_defaults_to_unsupported():
    def boom(prompt, **kwargs):
        raise RuntimeError("API down")

    verdict = verify_field("symptom", _field(), SOURCE, complete=boom)
    assert verdict.supported is False
    assert "error" in verdict.reason.lower()


# --- verify_record: prune a whole record against the source ---

RECORD_SOURCE = (
    "A 68-year-old man had recurrent fevers and was treated with "
    "ruxolitinib. CRP was 12 mg/dL. He was previously misdiagnosed with "
    "polymyalgia rheumatica. He died of pneumonia."
)


def _gf(value, quote, **kwargs):
    return GroundedField(value=value, quote=quote, **kwargs)


def _stub(supported_quotes):
    """A complete() that supports a claim iff its quote is in the set.

    The verify prompt's final block is the CLAIM, which embeds the quote.
    """
    def complete(prompt, **kwargs):
        claim = prompt[-1]["text"]
        supported = any(q in claim for q in supported_quotes)
        return {"supported": supported, "reason": "stub"}
    return complete


def _full_record():
    return PatientRecord(
        pmid="39012345",
        patient_id="39012345-p1",
        demographics={
            "age": _gf("68", "68-year-old"),
            "country": _gf("France", "in France"),
        },
        timeline=[TimelineEvent(symptom=_gf("fever", "recurrent fevers"))],
        labs=[LabValue(analyte=_gf("CRP", "CRP was 12 mg/dL"))],
        treatments=[
            TreatmentCourse(
                treatment=_gf("ruxolitinib", "treated with ruxolitinib"),
                crp_before=_gf("12", "CRP was 12 mg/dL"),
            )
        ],
        prior_diagnoses=[_gf("PMR", "misdiagnosed with polymyalgia rheumatica")],
        outcome=_gf("died", "died of pneumonia"),
    )


def test_verify_record_drops_unsupported_demographic():
    record = _full_record()
    every_real_quote = {
        "68-year-old", "recurrent fevers", "CRP was 12 mg/dL",
        "treated with ruxolitinib", "misdiagnosed with polymyalgia rheumatica",
        "died of pneumonia",
    }
    record, stats = verify_record(record, RECORD_SOURCE,
                                  complete=_stub(every_real_quote))
    assert "age" in record.demographics
    assert "country" not in record.demographics  # "in France" not in source
    assert stats.rejected == 1
    assert "demographics.country" in stats.rejected_fields


def test_verify_record_drops_timeline_event_when_symptom_unsupported():
    record = _full_record()
    record, _ = verify_record(record, RECORD_SOURCE,
                              complete=_stub({"68-year-old"}))
    assert record.timeline == []


def test_verify_record_drops_lab_when_analyte_unsupported():
    record = _full_record()
    record, _ = verify_record(record, RECORD_SOURCE,
                              complete=_stub({"68-year-old"}))
    assert record.labs == []


def test_verify_record_drops_treatment_when_treatment_field_unsupported():
    record = _full_record()
    record, _ = verify_record(record, RECORD_SOURCE,
                              complete=_stub({"68-year-old"}))
    assert record.treatments == []


def test_verify_record_nulls_unsupported_optional_treatment_subfield():
    record = _full_record()
    # Support the treatment itself but not its crp_before component.
    record, _ = verify_record(
        record, RECORD_SOURCE,
        complete=_stub({"treated with ruxolitinib", "68-year-old",
                        "recurrent fevers", "misdiagnosed with polymyalgia "
                        "rheumatica", "died of pneumonia"}),
    )
    assert len(record.treatments) == 1
    assert record.treatments[0].crp_before is None


def test_verify_record_drops_unsupported_prior_diagnosis():
    record = _full_record()
    record, _ = verify_record(record, RECORD_SOURCE,
                              complete=_stub({"68-year-old"}))
    assert record.prior_diagnoses == []


def test_verify_record_nulls_unsupported_outcome():
    record = _full_record()
    record, _ = verify_record(record, RECORD_SOURCE,
                              complete=_stub({"68-year-old"}))
    assert record.outcome is None


def test_verify_record_keeps_everything_when_all_supported():
    record = _full_record()
    del record.demographics["country"]  # remove the one bogus field
    record, stats = verify_record(
        record, RECORD_SOURCE,
        complete=lambda prompt, **k: {"supported": True, "reason": "ok"},
    )
    assert "age" in record.demographics
    assert len(record.timeline) == 1
    assert len(record.labs) == 1
    assert len(record.treatments) == 1
    assert record.treatments[0].crp_before is not None
    assert len(record.prior_diagnoses) == 1
    assert record.outcome is not None
    assert stats.rejected == 0
    assert stats.verified > 0


def test_verify_record_fails_closed_on_api_error():
    record = _full_record()

    def boom(prompt, **kwargs):
        raise RuntimeError("API down")

    record, stats = verify_record(record, RECORD_SOURCE, complete=boom)
    assert record.demographics == {}
    assert record.timeline == []
    assert record.labs == []
    assert record.treatments == []
    assert record.prior_diagnoses == []
    assert record.outcome is None
    assert stats.verified == 0


def test_verify_stats_rejection_rate():
    stats = VerifyStats(verified=3, rejected=1)
    assert stats.rejection_rate == 0.25
    assert VerifyStats().rejection_rate == 0.0
