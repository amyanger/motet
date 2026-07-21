import vexas.normalize.ontology as ontology
from vexas.cache import DiskCache
from vexas.normalize.ontology import (
    OntologyTerm,
    is_wellformed_hpo_id,
    lookup_rxnorm,
    parse_ols_response,
    parse_rxnorm_response,
)


def test_parses_ols_search_response():
    payload = {"response": {"docs": [
        {"obo_id": "HP:0000969", "label": "Edema", "ontology_name": "hp"},
        {"obo_id": "HP:0002027", "label": "Abdominal pain",
         "ontology_name": "hp"},
    ]}}
    terms = parse_ols_response(payload)
    assert terms[0] == OntologyTerm(term_id="HP:0000969", label="Edema",
                                    ontology="hp")
    assert len(terms) == 2


def test_ols_docs_without_obo_id_are_skipped():
    payload = {"response": {"docs": [
        {"label": "No id here", "ontology_name": "hp"},
        {"obo_id": "HP:0000969", "label": "Edema", "ontology_name": "hp"},
    ]}}
    assert len(parse_ols_response(payload)) == 1


def test_empty_ols_response_yields_no_terms():
    assert parse_ols_response({"response": {"docs": []}}) == []


def test_malformed_ols_response_yields_no_terms():
    assert parse_ols_response({}) == []


def test_wellformed_hpo_id_accepted():
    assert is_wellformed_hpo_id("HP:0000969") is True


def test_malformed_hpo_ids_rejected():
    """Catches the shape of IDs models invent."""
    for bad in ["HP:969", "HPO:0000969", "HP0000969", "", "HP:abcdefg"]:
        assert is_wellformed_hpo_id(bad) is False


def test_parses_rxnorm_response():
    payload = {"idGroup": {"rxnormId": ["1193326"]}}
    term = parse_rxnorm_response(payload, "ruxolitinib")
    assert term.term_id == "RxCUI:1193326"
    assert term.label == "ruxolitinib"


def test_rxnorm_miss_returns_none():
    assert parse_rxnorm_response({"idGroup": {}}, "unobtainium") is None


def test_lookup_rxnorm_caches_a_miss_and_skips_second_network_call(
    monkeypatch, tmp_path
):
    """A cached RxNorm miss must not fall through to a live call again.

    Regression test: put_json(drug, None) writes JSON `null`, and
    get_json returns None both for "never cached" and "cached null" --
    so the old `if cached is not None` guard could never detect a
    cached miss and would re-hit the network every time.
    """
    monkeypatch.setattr(
        ontology, "_rxnorm_cache",
        DiskCache("ontology_rxnorm_test", root=tmp_path),
    )

    calls = {"count": 0}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"idGroup": {}}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        return FakeResponse()

    monkeypatch.setattr(ontology.httpx, "get", fake_get)

    first = lookup_rxnorm("unobtainium")
    second = lookup_rxnorm("unobtainium")

    assert first is None
    assert second is None
    assert calls["count"] == 1
