"""Offline extraction-accuracy scorer.

Reads hand-authored answer keys (gold) and pipeline predictions (pred) in one
shared JSON shape, aligns predicted fields to gold fields, and reports
precision/recall/F1 per category, assertion-axis accuracy separately, and a
contamination-abstention check. Pure: no network, no model calls, no API key.

The shape mirrors PatientRecord so a serialized real-run prediction drops in
unchanged. It is deliberately plain JSON on disk, not the pydantic model, so a
human can read and edit an answer key and diff-review it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from vexas.gate import normalize_for_match
from vexas.normalize.hgvs import normalize_variant
from vexas.normalize.units import parse_numeric

# Sub-key each list category identifies its item by. The item's other keys
# (e.g. sequence_index, lab value components) are ignored by v1 scoring.
_LIST_IDENTIFIER = {
    "timeline": "symptom",
    "labs": "analyte",
    "treatments": "treatment",
}


@dataclass
class EvalField:
    value: str | None
    quote: str
    certainty: str = "positive"
    temporality: str = "present"
    experiencer: str = "patient"
    unit_as_written: str | None = None


@dataclass
class EvalUnit:
    patient_id: str
    text: str
    demographics: dict[str, EvalField] = field(default_factory=dict)
    timeline: list[EvalField] = field(default_factory=list)
    labs: list[EvalField] = field(default_factory=list)
    treatments: list[EvalField] = field(default_factory=list)
    prior_diagnoses: list[EvalField] = field(default_factory=list)
    outcome: EvalField | None = None
    reason_for_publication: EvalField | None = None


@dataclass
class EvalDoc:
    pmc_id: str
    pmid: str | None
    tier: str
    verified_by: str | None
    contamination_check: bool
    units: list[EvalUnit] = field(default_factory=list)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)


def values_match(gold: EvalField, pred: EvalField, key: str) -> bool:
    """True if two fields state the same value, normalization-aware."""
    if gold.value is None or pred.value is None:
        return False
    if key == "uba1_variant":
        g, p = normalize_variant(gold.value), normalize_variant(pred.value)
        return g is not None and g == p
    g_num, p_num = parse_numeric(gold.value), parse_numeric(pred.value)
    if g_num is not None and p_num is not None:
        return g_num == p_num
    return normalize_for_match(gold.value) == normalize_for_match(pred.value)


def score_scalar_dict(
    gold: dict[str, EvalField], pred: dict[str, EvalField]
) -> Counts:
    """Score keyed scalar fields. Each key is TP / FP / FN independently."""
    counts = Counts()
    for key, gold_field in gold.items():
        pred_field = pred.get(key)
        if pred_field is not None and values_match(gold_field, pred_field, key):
            counts.tp += 1
        else:
            counts.fn += 1
            if pred_field is not None:
                counts.fp += 1  # present but wrong: both a miss and a false hit
    for key in pred:
        if key not in gold:
            counts.fp += 1
    return counts


def _field(raw: dict | None) -> EvalField | None:
    if raw is None:
        return None
    return EvalField(
        value=raw.get("value"),
        quote=raw.get("quote", ""),
        certainty=raw.get("certainty", "positive"),
        temporality=raw.get("temporality", "present"),
        experiencer=raw.get("experiencer", "patient"),
        unit_as_written=raw.get("unit_as_written"),
    )


def _list(raw_items: list[dict], category: str) -> list[EvalField]:
    key = _LIST_IDENTIFIER.get(category)
    fields: list[EvalField] = []
    for item in raw_items:
        # prior_diagnoses items are field objects directly; others wrap the
        # identifying field under a sub-key.
        source = item if key is None else item.get(key)
        parsed = _field(source)
        if parsed is not None:
            fields.append(parsed)
    return fields


def _unit(raw: dict) -> EvalUnit:
    return EvalUnit(
        patient_id=raw["patient_id"],
        text=raw.get("text", ""),
        demographics={
            k: f
            for k, v in raw.get("demographics", {}).items()
            if (f := _field(v)) is not None
        },
        timeline=_list(raw.get("timeline", []), "timeline"),
        labs=_list(raw.get("labs", []), "labs"),
        treatments=_list(raw.get("treatments", []), "treatments"),
        prior_diagnoses=_list(raw.get("prior_diagnoses", []), "prior_diagnoses"),
        outcome=_field(raw.get("outcome")),
        reason_for_publication=_field(raw.get("reason_for_publication")),
    )


def load_eval_file(path: Path) -> EvalDoc:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDoc(
        pmc_id=raw["pmc_id"],
        pmid=raw.get("pmid"),
        tier=raw.get("tier", "silver"),
        verified_by=raw.get("verified_by"),
        contamination_check=raw.get("contamination_check", False),
        units=[_unit(u) for u in raw.get("units", [])],
    )
