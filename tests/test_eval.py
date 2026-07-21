import json

from vexas.eval import Counts, EvalField, load_eval_file, score_scalar_dict, values_match


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


def _f(value, **kw):
    return EvalField(value=value, quote=kw.pop("quote", "q"), **kw)


def test_counts_metrics():
    c = Counts(tp=8, fp=2, fn=4)
    assert c.precision == 0.8
    assert c.recall == 8 / 12
    assert round(c.f1, 4) == round(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12), 4)


def test_counts_metrics_are_none_when_undefined():
    assert Counts(tp=0, fp=0, fn=0).precision is None
    assert Counts(tp=0, fp=0, fn=0).recall is None
    assert Counts(tp=0, fp=0, fn=0).f1 is None


def test_values_match_is_case_and_space_insensitive():
    assert values_match(_f("Macrocytic Anemia"), _f("macrocytic anemia"), "x")


def test_values_match_variant_uses_normalizer():
    assert values_match(_f("p.Met41Thr"), _f("M41T"), "uba1_variant")


def test_values_match_numeric_agrees_across_unit_text():
    assert values_match(_f("68"), _f("68 years"), "age")


def test_values_match_rejects_different_values():
    assert not values_match(_f("male"), _f("female"), "sex")


def test_scalar_dict_counts_tp_fp_fn():
    gold = {"age": _f("68"), "sex": _f("male"), "country": _f("Japan")}
    pred = {"age": _f("68"), "sex": _f("female"), "vaf": _f("62")}
    # age: TP; sex: value mismatch -> FP + FN; country: missing -> FN;
    # vaf: not in gold -> FP
    counts = score_scalar_dict(gold, pred)
    assert counts.tp == 1
    assert counts.fp == 2   # sex-wrong, vaf-extra
    assert counts.fn == 2   # sex-wrong, country-missing
