import json

from vexas.eval import EvalField, load_eval_file


def _doc_dict():
    return {
        "pmc_id": "PMC9999999",
        "pmid": "39012345",
        "tier": "silver",
        "verified_by": None,
        "contamination_check": False,
        "units": [
            {
                "patient_id": "39012345-p1",
                "text": "A 68-year-old man with recurrent fevers, on ruxolitinib. CRP 12 mg/dL.",
                "demographics": {
                    "age": {"value": "68", "unit_as_written": "years",
                            "quote": "A 68-year-old man"}
                },
                "timeline": [
                    {"symptom": {"value": "fever", "quote": "recurrent fevers"},
                     "sequence_index": 1}
                ],
                "labs": [
                    {"analyte": {"value": "CRP", "quote": "CRP 12 mg/dL"}}
                ],
                "treatments": [
                    {"treatment": {"value": "ruxolitinib", "quote": "on ruxolitinib"}}
                ],
                "prior_diagnoses": [],
                "outcome": None,
                "reason_for_publication": None,
            }
        ],
    }


def test_loads_top_level_metadata(tmp_path):
    path = tmp_path / "PMC9999999.json"
    path.write_text(json.dumps(_doc_dict()), encoding="utf-8")
    doc = load_eval_file(path)
    assert doc.pmc_id == "PMC9999999"
    assert doc.tier == "silver"
    assert doc.contamination_check is False
    assert len(doc.units) == 1


def test_scalar_field_defaults_axes(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_doc_dict()), encoding="utf-8")
    unit = load_eval_file(path).units[0]
    age = unit.demographics["age"]
    assert isinstance(age, EvalField)
    assert age.value == "68"
    assert age.certainty == "positive"
    assert age.temporality == "present"
    assert age.experiencer == "patient"


def test_list_items_map_to_identifying_field(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_doc_dict()), encoding="utf-8")
    unit = load_eval_file(path).units[0]
    assert unit.timeline[0].value == "fever"
    assert unit.labs[0].value == "CRP"
    assert unit.treatments[0].value == "ruxolitinib"


def test_missing_optional_fields_are_none(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_doc_dict()), encoding="utf-8")
    unit = load_eval_file(path).units[0]
    assert unit.outcome is None
    assert unit.reason_for_publication is None
    assert unit.prior_diagnoses == []
