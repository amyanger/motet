import pytest

from vexas.normalize.ontology import (
    OntologyTerm,
    is_wellformed_hpo_id,
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
