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
        if p is None or r is None:
            return None
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class AxisScore:
    correct: int = 0
    total: int = 0

    def __add__(self, other: "AxisScore") -> "AxisScore":
        return AxisScore(self.correct + other.correct, self.total + other.total)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


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


def align_list(
    gold: list[EvalField], pred: list[EvalField], key: str
) -> list[tuple[EvalField, EvalField]]:
    """Greedily pair gold items with matching predictions, one-to-one."""
    used: set[int] = set()
    pairs: list[tuple[EvalField, EvalField]] = []
    for gold_field in gold:
        for i, pred_field in enumerate(pred):
            if i in used:
                continue
            if values_match(gold_field, pred_field, key):
                used.add(i)
                pairs.append((gold_field, pred_field))
                break
    return pairs


def axis_accuracy(pairs: list[tuple[EvalField, EvalField]]) -> AxisScore:
    """Score assertion-axis accuracy over matched (true-positive) field pairs.

    A pair is axis-correct only if certainty, temporality, and experiencer
    all match.
    """
    score = AxisScore()
    for gold, pred in pairs:
        score.total += 1
        if (
            gold.certainty == pred.certainty
            and gold.temporality == pred.temporality
            and gold.experiencer == pred.experiencer
        ):
            score.correct += 1
    return score


def score_list(
    gold: list[EvalField], pred: list[EvalField], key: str
) -> tuple[Counts, list[tuple[EvalField, EvalField]]]:
    pairs = align_list(gold, pred, key)
    counts = Counts(
        tp=len(pairs),
        fp=len(pred) - len(pairs),
        fn=len(gold) - len(pairs),
    )
    return counts, pairs


def _field(raw: dict | None) -> EvalField | None:
    if raw is None:
        return None
    if raw.get("value") is None:
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
            for k, v in (raw.get("demographics") or {}).items()
            if (f := _field(v)) is not None
        },
        timeline=_list(raw.get("timeline") or [], "timeline"),
        labs=_list(raw.get("labs") or [], "labs"),
        treatments=_list(raw.get("treatments") or [], "treatments"),
        prior_diagnoses=_list(raw.get("prior_diagnoses") or [], "prior_diagnoses"),
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


CONTAMINATION_MAX_FIELDS = 2

CATEGORY_COUNTS = (
    "demographics", "timeline", "labs", "treatments",
    "prior_diagnoses", "outcome", "reason_for_publication",
)

_LIST_CATEGORIES = ("timeline", "labs", "treatments", "prior_diagnoses")


@dataclass
class UnitScore:
    per_category: dict[str, Counts]
    axes: AxisScore
    total_pred_fields: int

    @property
    def overall(self) -> Counts:
        total = Counts()
        for counts in self.per_category.values():
            total = total + counts
        return total


def _single_as_dict(name: str, f: EvalField | None) -> dict[str, EvalField]:
    return {name: f} if f is not None else {}


def _count_pred_fields(unit: EvalUnit) -> int:
    n = len(unit.demographics)
    for cat in _LIST_CATEGORIES:
        n += len(getattr(unit, cat))
    n += 1 if unit.outcome is not None else 0
    n += 1 if unit.reason_for_publication is not None else 0
    return n


def score_unit(gold: EvalUnit, pred: EvalUnit) -> UnitScore:
    per_category: dict[str, Counts] = {}
    axes = AxisScore()

    # Scalars: demographics dict, plus the two single optional fields.
    scalar_groups = {
        "demographics": (gold.demographics, pred.demographics),
        "outcome": (
            _single_as_dict("outcome", gold.outcome),
            _single_as_dict("outcome", pred.outcome),
        ),
        "reason_for_publication": (
            _single_as_dict("reason_for_publication", gold.reason_for_publication),
            _single_as_dict("reason_for_publication", pred.reason_for_publication),
        ),
    }
    for cat, (g, p) in scalar_groups.items():
        per_category[cat] = score_scalar_dict(g, p)
        # Axis pairs for matched scalar keys.
        for key, gold_field in g.items():
            pred_field = p.get(key)
            if pred_field is not None and values_match(gold_field, pred_field, key):
                axes = axes + axis_accuracy([(gold_field, pred_field)])

    # Lists.
    for cat in _LIST_CATEGORIES:
        counts, pairs = score_list(getattr(gold, cat), getattr(pred, cat), cat)
        per_category[cat] = counts
        axes = axes + axis_accuracy(pairs)

    return UnitScore(
        per_category=per_category,
        axes=axes,
        total_pred_fields=_count_pred_fields(pred),
    )


@dataclass
class DocScore:
    pmc_id: str
    tier: str
    contamination_check: bool
    units: list[UnitScore]
    abstained: bool | None

    @property
    def overall(self) -> Counts:
        total = Counts()
        for unit in self.units:
            total = total + unit.overall
        return total

    @property
    def axes(self) -> AxisScore:
        total = AxisScore()
        for unit in self.units:
            total = total + unit.axes
        return total


@dataclass
class TierReport:
    tier: str
    per_category: dict[str, Counts]
    overall: Counts
    axes: AxisScore
    contamination_pass: int
    contamination_total: int


def _pair_units(gold: EvalDoc, pred: EvalDoc) -> list[tuple[EvalUnit, EvalUnit]]:
    """Pair gold and pred units by patient_id, falling back to position."""
    by_id = {u.patient_id: u for u in pred.units}
    empty = EvalUnit(patient_id="", text="")
    pairs = []
    for i, gold_unit in enumerate(gold.units):
        pred_unit = by_id.get(gold_unit.patient_id)
        if pred_unit is None:
            pred_unit = pred.units[i] if i < len(pred.units) else empty
        pairs.append((gold_unit, pred_unit))
    return pairs


def score_document(gold: EvalDoc, pred: EvalDoc) -> DocScore:
    unit_scores = [score_unit(g, p) for g, p in _pair_units(gold, pred)]
    abstained: bool | None = None
    if gold.contamination_check:
        predicted = sum(_count_pred_fields(u) for u in pred.units)
        abstained = predicted <= CONTAMINATION_MAX_FIELDS
    return DocScore(
        pmc_id=gold.pmc_id,
        tier=gold.tier,
        contamination_check=gold.contamination_check,
        units=unit_scores,
        abstained=abstained,
    )


def score_corpus(docs: list[DocScore]) -> dict[str, TierReport]:
    reports: dict[str, TierReport] = {}
    tiers: dict[str, list[DocScore]] = {}
    for doc in docs:
        tiers.setdefault(doc.tier, []).append(doc)

    for tier, tier_docs in tiers.items():
        per_category: dict[str, Counts] = {c: Counts() for c in CATEGORY_COUNTS}
        overall = Counts()
        axes = AxisScore()
        c_pass = c_total = 0
        for doc in tier_docs:
            for cat in CATEGORY_COUNTS:
                for unit in doc.units:
                    per_category[cat] = per_category[cat] + unit.per_category.get(
                        cat, Counts()
                    )
            overall = overall + doc.overall
            axes = axes + doc.axes
            if doc.contamination_check:
                c_total += 1
                c_pass += 1 if doc.abstained else 0
        reports[tier] = TierReport(
            tier=tier,
            per_category=per_category,
            overall=overall,
            axes=axes,
            contamination_pass=c_pass,
            contamination_total=c_total,
        )
    return reports


def _fmt_metric(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "  --"


def _fmt_counts(label: str, c: Counts) -> str:
    p = f"P {_fmt_metric(c.precision)} ({c.tp}/{c.tp + c.fp})"
    r = f"R {_fmt_metric(c.recall)} ({c.tp}/{c.tp + c.fn})"
    f = f"F1 {_fmt_metric(c.f1)}"
    return f"  {label:<24} {p}  {r}  {f}"


def format_report(reports: dict[str, TierReport]) -> str:
    lines: list[str] = []
    for tier in sorted(reports):
        report = reports[tier]
        lines.append(f"=== {tier} ===")
        for cat in CATEGORY_COUNTS:
            lines.append(_fmt_counts(cat, report.per_category[cat]))
        lines.append(_fmt_counts("OVERALL", report.overall))
        acc = report.axes.accuracy
        acc_str = _fmt_metric(acc)
        lines.append(
            f"  assertion axes: {acc_str} "
            f"({report.axes.correct}/{report.axes.total} matched fields)"
        )
        if report.contamination_total:
            lines.append(
                f"  contamination abstention: "
                f"{report.contamination_pass}/{report.contamination_total} passed"
            )
        lines.append("")
    return "\n".join(lines)
