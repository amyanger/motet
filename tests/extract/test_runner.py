from vexas.aggregate.odyssey import diagnostic_delays
from vexas.extract.runner import GateStats, apply_gate, extract_unit
from vexas.segment import PatientUnit

SOURCE = (
    "A 68-year-old man presented with recurrent fevers for six months, "
    "followed by ear chondritis. He was treated with ruxolitinib."
)


def test_grounded_field_survives_the_gate():
    payload = {"fields": {"age": {"value": "68", "quote": "68-year-old man"}}}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert "age" in cleaned["fields"]
    assert stats.kept == 1
    assert stats.rejected == 0


def test_ungrounded_field_is_removed():
    payload = {"fields": {"age": {"value": "70", "quote": "70-year-old woman"}}}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert cleaned["fields"] == {}
    assert stats.rejected == 1


def test_paraphrased_quote_is_removed():
    """The exact failure the gate exists to stop."""
    payload = {"treatments": [
        {"treatment": {"value": "ruxolitinib",
                       "quote": "treated with a JAK inhibitor"}}
    ]}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert cleaned["treatments"] == []
    assert stats.rejected == 1


def test_rejected_quotes_are_recorded_for_inspection():
    payload = {"fields": {"age": {"value": "70", "quote": "invented text"}}}
    _, stats = apply_gate(payload, SOURCE)
    assert "invented text" in stats.rejected_quotes


def test_gate_recurses_into_nested_lists():
    payload = {"timeline": [
        {"symptom": {"value": "fevers", "quote": "recurrent fevers"},
         "sequence_index": 1},
        {"symptom": {"value": "rash", "quote": "widespread rash"},
         "sequence_index": 2},
    ]}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert len(cleaned["timeline"]) == 1
    assert cleaned["timeline"][0]["symptom"]["value"] == "fevers"


def test_field_missing_a_quote_is_rejected():
    payload = {"fields": {"age": {"value": "68"}}}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert cleaned["fields"] == {}
    assert stats.rejected == 1


def test_non_field_scalars_pass_through_untouched():
    payload = {"timeline": [
        {"symptom": {"value": "fevers", "quote": "recurrent fevers"},
         "sequence_index": 1, "interval_value": 6, "interval_unit": "months"}
    ]}
    cleaned, _ = apply_gate(payload, SOURCE)
    assert cleaned["timeline"][0]["interval_value"] == 6


def test_extract_unit_builds_a_record_from_stubbed_responses():
    unit = PatientUnit(patient_id="39012345-p1", pmid="39012345", text=SOURCE)

    def fake_complete(prompt, max_tokens=4096):
        # prompt is now a list of content blocks, not a string
        flat = "\n".join(b["text"] for b in prompt)
        if "age, sex" in flat:
            return {"fields": {"age": {"value": "68",
                                       "quote": "68-year-old man"}}}
        if "onset sequence" in flat:
            return {"timeline": [{"symptom": {"value": "recurrent fevers",
                                              "quote": "recurrent fevers"},
                                  "sequence_index": 1}]}
        return {}

    record, stats = extract_unit(unit, complete=fake_complete)
    assert record.pmid == "39012345"
    assert record.patient_id == "39012345-p1"
    assert record.demographics["age"].value == "68"
    assert record.timeline[0].sequence_index == 1
    assert stats.rejected == 0


def test_extract_unit_wires_time_to_diagnosis_into_demographics():
    """The timeline prompt emits a top-level time_to_diagnosis field, but
    extract_unit used to drop it, so odyssey.diagnostic_delays always saw
    an empty demographics entry. This exercises the full seam."""
    source = (
        "A 71-year-old woman had recurrent fevers. Diagnosis of VEXAS was "
        "confirmed 2 years after symptom onset."
    )
    unit = PatientUnit(patient_id="40000001-p1", pmid="40000001", text=source)

    def fake_complete(prompt, max_tokens=4096):
        flat = "\n".join(b["text"] for b in prompt)
        if "onset sequence" in flat:
            return {
                "time_to_diagnosis": {
                    "value": "2",
                    "unit_as_written": "years",
                    "quote": "confirmed 2 years after symptom onset",
                }
            }
        return {}

    record, stats = extract_unit(unit, complete=fake_complete)
    assert record.demographics["time_to_diagnosis"].value == "2"

    delays = diagnostic_delays([record])
    assert delays == [("40000001", 2.0)]
