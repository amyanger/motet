"""Validation against published findings.

Everything in KNOWN_FINDINGS is already in the literature (NIH cohort,
Ferrada et al., Blood 2022; GeneReviews NBK614471). Recovering these
demonstrates the pipeline works. Presenting them as discoveries would be
misrepresentation, so the prior art is recorded alongside each one.
"""

from collections import defaultdict

from vexas.aggregate.denominators import Count
from vexas.normalize.hgvs import normalize_variant
from vexas.schema import PatientRecord

KNOWN_FINDINGS: dict[str, str] = {
    "M41V": (
        "Worst prognosis; highest mortality; less ear chondritis. "
        "Already published (NIH cohort, Blood 2022)."
    ),
    "M41T": (
        "Most common variant; more inflammatory eye disease. "
        "Already published (NIH cohort, Blood 2022)."
    ),
    "M41L": (
        "Most indolent course; more skin involvement. "
        "Already published (NIH cohort, Blood 2022)."
    ),
    "chondritis": (
        "Ear chondritis is protective for survival. "
        "Already published (NIH cohort, Blood 2022)."
    ),
}


def check_genotype_gradient(
    records: list[PatientRecord],
) -> dict[str, Count]:
    """Count reports per normalized UBA1 variant.

    Denominator is reports carrying a parseable genotype -- reports that do
    not state one cannot contribute either way.
    """
    by_variant: dict[str, set[str]] = defaultdict(set)
    genotyped: set[str] = set()

    for record in records:
        field = record.demographics.get("uba1_variant")
        if field is None or not field.value:
            continue
        variant = normalize_variant(field.value)
        if variant is None:
            continue
        genotyped.add(record.pmid)
        by_variant[variant].add(record.pmid)

    denominator = len(genotyped)
    if denominator == 0:
        return {}

    return {
        variant: Count(
            n=len(pmids),
            denominator=denominator,
            label=variant,
            supporting_pmids=sorted(pmids),
        )
        for variant, pmids in by_variant.items()
    }
