"""Ontology mapping: retrieve, then validate.

The division of labour is fixed and must not be relaxed:
  1. The extraction model emits only a surface phrase and its quote.
  2. This module retrieves candidate terms from the real ontology.
  3. Code validates the identifier exists.
  4. A model may re-rank among validated candidates, never invent one.

GPT-4 maps signs to HPO at 57.9% accuracy versus 90.6% for a plain
embedding baseline (arXiv:2408.01214), because it generates plausible IDs
with no mechanism to verify them (arXiv:2509.04458).

APIs used (both free, no key, verified live 2026-07-18):
  OLS4   https://www.ebi.ac.uk/ols4/api/search
  RxNorm https://rxnav.nlm.nih.gov/REST/rxcui.json
"""

import re
from dataclasses import dataclass

import httpx

from vexas.cache import DiskCache

OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
RXNORM_LOOKUP = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

#: Pinned for reproducibility -- recorded in output metadata so a clinician
#: can reproduce any mapping. Update deliberately, never incidentally.
HPO_RELEASE = "2025-05-06"

_hpo_cache = DiskCache("ontology_hpo")
_rxnorm_cache = DiskCache("ontology_rxnorm")

_HPO_ID = re.compile(r"^HP:\d{7}$")


@dataclass(frozen=True)
class OntologyTerm:
    term_id: str
    label: str
    ontology: str


def is_wellformed_hpo_id(term_id: str) -> bool:
    """Shape check. Catches invented identifiers before a network call."""
    return bool(_HPO_ID.match(term_id or ""))


def parse_ols_response(payload: dict) -> list[OntologyTerm]:
    docs = payload.get("response", {}).get("docs", [])
    terms = []
    for doc in docs:
        term_id = doc.get("obo_id")
        if not term_id:
            continue
        terms.append(
            OntologyTerm(
                term_id=term_id,
                label=doc.get("label", ""),
                ontology=doc.get("ontology_name", ""),
            )
        )
    return terms


def parse_rxnorm_response(payload: dict, drug: str) -> OntologyTerm | None:
    ids = payload.get("idGroup", {}).get("rxnormId", [])
    if not ids:
        return None
    return OntologyTerm(term_id=f"RxCUI:{ids[0]}", label=drug,
                        ontology="rxnorm")


def search_hpo(phrase: str, limit: int = 5) -> list[OntologyTerm]:
    """Retrieve candidate HPO terms for a surface phrase. Cached."""
    cached = _hpo_cache.get_json(phrase)
    if cached is not None:
        return [OntologyTerm(**item) for item in cached]

    response = httpx.get(
        OLS4_SEARCH,
        params={"q": phrase, "ontology": "hp", "rows": limit},
        timeout=30,
    )
    response.raise_for_status()
    terms = parse_ols_response(response.json())
    # Belt and braces: reject anything malformed even from the real API.
    terms = [t for t in terms if is_wellformed_hpo_id(t.term_id)]
    _hpo_cache.put_json(phrase, [t.__dict__ for t in terms])
    return terms


def lookup_rxnorm(drug: str) -> OntologyTerm | None:
    """Resolve a drug name to an RxCUI. Cached."""
    cached = _rxnorm_cache.get_json(drug)
    if cached is not None:
        return OntologyTerm(**cached) if cached else None

    response = httpx.get(RXNORM_LOOKUP, params={"name": drug}, timeout=30)
    response.raise_for_status()
    term = parse_rxnorm_response(response.json(), drug)
    _rxnorm_cache.put_json(drug, term.__dict__ if term else None)
    return term
