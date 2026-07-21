"""The diagnostic odyssey: what VEXAS was called before it was called VEXAS.

VEXAS is routinely misdiagnosed as relapsing polychondritis, giant cell
arteritis, Sweet syndrome, polymyalgia rheumatica, or MDS alone. Individual
case reports mention their own patient's wrong turns; nobody has tabulated
them across the literature.
"""

from collections import defaultdict

from vexas.aggregate.denominators import Count, Pattern
from vexas.normalize.units import parse_numeric
from vexas.schema import Certainty, PatientRecord

#: Alias -> canonical name. Lowercased keys.
MISDIAGNOSIS_ALIASES: dict[str, str] = {
    "rp": "relapsing polychondritis",
    "relapsing polychondritis": "relapsing polychondritis",
    "gca": "giant cell arteritis",
    "giant cell arteritis": "giant cell arteritis",
    "temporal arteritis": "giant cell arteritis",
    "pmr": "polymyalgia rheumatica",
    "polymyalgia rheumatica": "polymyalgia rheumatica",
    "sweet syndrome": "sweet syndrome",
    "sweet's syndrome": "sweet syndrome",
    "acute febrile neutrophilic dermatosis": "sweet syndrome",
    "mds": "myelodysplastic syndrome",
    "myelodysplastic syndrome": "myelodysplastic syndrome",
    "aosd": "adult-onset still disease",
    "adult onset still's disease": "adult-onset still disease",
    "adult-onset still's disease": "adult-onset still disease",
    "pan": "polyarteritis nodosa",
    "polyarteritis nodosa": "polyarteritis nodosa",
}

_TO_YEARS: dict[str, float] = {
    "year": 1.0, "years": 1.0, "yr": 1.0, "yrs": 1.0,
    "month": 1 / 12, "months": 1 / 12, "mo": 1 / 12, "mos": 1 / 12,
    "week": 1 / 52, "weeks": 1 / 52, "wk": 1 / 52, "wks": 1 / 52,
    "day": 1 / 365, "days": 1 / 365,
}


def canonical_diagnosis(raw: str) -> str:
    """Collapse aliases. Unknown names are lowercased, never dropped."""
    key = raw.strip().lower()
    return MISDIAGNOSIS_ALIASES.get(key, key)


def rank_misdiagnoses(records: list[PatientRecord],
                      min_support: int = 3) -> list[Pattern]:
    """Rank prior diagnoses by how many reports describe them.

    Denominator counts only reports that mention prior diagnoses at all --
    silence is not evidence the patient had none.
    """
    informative = [
        record for record in records
        if any(d.certainty is Certainty.POSITIVE and d.value
               for d in record.prior_diagnoses)
    ]
    denominator = len(informative)
    if denominator == 0:
        return []

    support: dict[str, set[str]] = defaultdict(set)
    for record in informative:
        for diagnosis in record.prior_diagnoses:
            if diagnosis.certainty is not Certainty.POSITIVE:
                continue
            if not diagnosis.value:
                continue
            # A set per pmid, so one report naming the same condition
            # twice still counts once.
            support[canonical_diagnosis(diagnosis.value)].add(record.pmid)

    patterns = []
    for diagnosis, pmids in support.items():
        if len(pmids) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"diagnosed as {diagnosis} before VEXAS confirmed",
                count=Count(
                    n=len(pmids),
                    denominator=denominator,
                    label=diagnosis,
                    supporting_pmids=sorted(pmids),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)


def diagnostic_delays(records: list[PatientRecord]) -> list[tuple[str, float]]:
    """Time from first symptom to UBA1 testing, in years, per report.

    Unparseable values and unknown units are skipped rather than guessed.
    """
    delays = []
    for record in records:
        field = record.demographics.get("time_to_diagnosis")
        if field is None or not field.value:
            continue
        value = parse_numeric(field.value)
        if value is None:
            continue
        factor = _TO_YEARS.get((field.unit_as_written or "").strip().lower())
        if factor is None:
            continue
        delays.append((record.pmid, round(value * factor, 3)))
    return delays
