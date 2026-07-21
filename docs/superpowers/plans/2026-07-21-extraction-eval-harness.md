# Extraction Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure, offline harness that scores the extraction pipeline's per-field accuracy on real case reports against hand-authored answer keys, reporting precision/recall/F1 per category, a separate assertion-axis accuracy number, and a contamination-abstention check.

**Architecture:** A dependency-free scorer (`src/vexas/eval.py`) reads gold and prediction JSON files from a new `eval/` directory, aligns predicted fields to gold fields, and aggregates counts into a tier-grouped report. A `motet eval` CLI subcommand runs it. The scorer never calls a model or the network. Answer keys and predictions are data, authored by hand; the code is validated by TDD on synthetic inputs and never depends on the real data existing.

**Tech Stack:** Python 3.14, stdlib only for the scorer (dataclasses, json), pytest. Reuses existing `vexas.gate`, `vexas.normalize.hgvs`, `vexas.normalize.units`.

## Global Constraints

- Python 3.14.4. No `uv`. src-layout, `pythonpath=["src"]`. Run tests with `.venv/bin/pytest -q`.
- The scorer is pure: no network, no Anthropic calls, no API key. `motet eval` spends nothing.
- The internal Python package remains `vexas` (the public brand is Motet; the package name is deliberately not renamed).
- Reuse existing normalization rather than re-implementing: `gate.normalize_for_match` for strings, `normalize.hgvs.normalize_variant` for variants, `normalize.units.parse_numeric` for numerics.
- The global no-em-dash rule applies inside code/docstrings too -- use `--`, not an em-dash.
- Headline precision/recall/F1 is **lenient**: a field is a true positive if its identifying value matches. Assertion-axis correctness is a **separate** number, never folded into the headline.
- Gold and prediction files share one shape and load through one loader. The shape mirrors `PatientRecord` so a real-run prediction (serialized `PatientRecord`) drops in unchanged.
- Only commit when the executing skill's task cycle calls for it; follow the per-task commit steps below.

---

### Task 1: Eval data model and loader

The foundation: the on-disk shape and how it parses into memory. Everything else consumes `EvalDoc`.

**Files:**
- Create: `src/vexas/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: nothing (stdlib only)
- Produces:
  - `vexas.eval.EvalField` dataclass: `.value: str | None`, `.quote: str`, `.certainty: str`, `.temporality: str`, `.experiencer: str`, `.unit_as_written: str | None`
  - `vexas.eval.EvalUnit` dataclass: `.patient_id: str`, `.text: str`, `.demographics: dict[str, EvalField]`, `.timeline: list[EvalField]`, `.labs: list[EvalField]`, `.treatments: list[EvalField]`, `.prior_diagnoses: list[EvalField]`, `.outcome: EvalField | None`, `.reason_for_publication: EvalField | None`
  - `vexas.eval.EvalDoc` dataclass: `.pmc_id: str`, `.pmid: str | None`, `.tier: str`, `.verified_by: str | None`, `.contamination_check: bool`, `.units: list[EvalUnit]`
  - `vexas.eval.load_eval_file(path: Path) -> EvalDoc`

The loader maps each list item to the `EvalField` built from its identifying sub-key: `timeline` items from their `symptom` object, `labs` items from their `analyte` object, `treatments` items from their `treatment` object, `prior_diagnoses` items are themselves field objects. Extra keys on a list item (e.g. `sequence_index`) are ignored by v1 -- the shape is richer than v1 scores so the scorer can grow without re-annotation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval.py`:

```python
import json
from pathlib import Path

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'vexas.eval'`

- [ ] **Step 3: Write `src/vexas/eval.py`**

```python
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
            k: _field(v) for k, v in raw.get("demographics", {}).items()
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/vexas/eval.py tests/test_eval.py
git commit -m "Add eval data model and loader

One shared JSON shape for answer keys and predictions, mirroring
PatientRecord so a real-run prediction drops in unchanged."
```

---

### Task 2: Value normalization and scalar-dict scoring

Scalar fields (demographics keys, and single optional fields wrapped as a
one-key dict) are scored by key with normalized value equality.

**Files:**
- Modify: `src/vexas/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `EvalField`, `gate.normalize_for_match`, `normalize.hgvs.normalize_variant`, `normalize.units.parse_numeric`
- Produces:
  - `vexas.eval.Counts` dataclass: `.tp: int`, `.fp: int`, `.fn: int`, `__add__`, `.precision`, `.recall`, `.f1` (each metric returns `float | None`; `None` when the denominator is zero)
  - `vexas.eval.values_match(gold: EvalField, pred: EvalField, key: str) -> bool`
  - `vexas.eval.score_scalar_dict(gold: dict[str, EvalField], pred: dict[str, EvalField]) -> Counts`

Matching rule for `values_match`: if `key == "uba1_variant"`, compare `normalize_variant(value)` on both sides (equal, and both non-None). Otherwise, if both normalized-numeric parses succeed and are equal, match. Otherwise compare `normalize_for_match(value)`. A `None` gold or pred value never matches.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from vexas.eval import Counts, EvalField, score_scalar_dict, values_match


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL -- `ImportError: cannot import name 'Counts'`

- [ ] **Step 3: Add to `src/vexas/eval.py`**

Add these imports at the top (below the existing imports):

```python
from vexas.gate import normalize_for_match
from vexas.normalize.hgvs import normalize_variant
from vexas.normalize.units import parse_numeric
```

Add after the dataclasses:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/vexas/eval.py tests/test_eval.py
git commit -m "Add Counts and normalization-aware scalar scoring

Reuses the pipeline's own gate/variant/numeric normalizers so the yardstick
and the system agree on what 'equal' means."
```

---

### Task 3: List-field alignment and scoring

List categories align predicted items to gold items by identifying value, then
count matched pairs as TP and leftovers as FP/FN.

**Files:**
- Modify: `src/vexas/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `EvalField`, `Counts`, `values_match`
- Produces:
  - `vexas.eval.align_list(gold: list[EvalField], pred: list[EvalField], key: str) -> list[tuple[EvalField, EvalField]]` -- returns matched pairs only, one-to-one, greedy in gold order
  - `vexas.eval.score_list(gold: list[EvalField], pred: list[EvalField], key: str) -> tuple[Counts, list[tuple[EvalField, EvalField]]]`

Alignment: for each gold item in order, pick the first not-yet-used pred item whose value matches (`values_match` with the given key); record the pair and mark the pred item used. `Counts.tp` = number of pairs, `fn` = unmatched gold, `fp` = unmatched pred.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from vexas.eval import align_list, score_list


def test_list_matches_regardless_of_order():
    gold = [_f("fever"), _f("chondritis"), _f("anemia")]
    pred = [_f("anemia"), _f("fever"), _f("chondritis")]
    counts, pairs = score_list(gold, pred, "timeline")
    assert counts.tp == 3
    assert counts.fp == 0
    assert counts.fn == 0
    assert len(pairs) == 3


def test_list_extra_prediction_is_false_positive():
    gold = [_f("fever")]
    pred = [_f("fever"), _f("rash")]
    counts, _ = score_list(gold, pred, "timeline")
    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.fn == 0


def test_list_missed_gold_is_false_negative():
    gold = [_f("fever"), _f("chondritis")]
    pred = [_f("fever")]
    counts, _ = score_list(gold, pred, "timeline")
    assert counts.tp == 1
    assert counts.fp == 0
    assert counts.fn == 1


def test_align_pairs_gold_with_matching_pred():
    gold = [_f("ruxolitinib")]
    pred = [_f("Ruxolitinib")]
    pairs = align_list(gold, pred, "treatments")
    assert len(pairs) == 1
    assert pairs[0][0].value == "ruxolitinib"
    assert pairs[0][1].value == "Ruxolitinib"


def test_duplicate_gold_values_consume_distinct_predictions():
    gold = [_f("fever"), _f("fever")]
    pred = [_f("fever")]
    counts, _ = score_list(gold, pred, "timeline")
    assert counts.tp == 1
    assert counts.fn == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL -- `ImportError: cannot import name 'align_list'`

- [ ] **Step 3: Add to `src/vexas/eval.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vexas/eval.py tests/test_eval.py
git commit -m "Add greedy list-field alignment and scoring"
```

---

### Task 4: Assertion-axis accuracy on matched fields

Over matched (true-positive) pairs, measure how often all three axes agree.
Reported separately from the headline P/R/F1.

**Files:**
- Modify: `src/vexas/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `EvalField`
- Produces:
  - `vexas.eval.AxisScore` dataclass: `.correct: int`, `.total: int`, `__add__`, `.accuracy -> float | None`
  - `vexas.eval.axis_accuracy(pairs: list[tuple[EvalField, EvalField]]) -> AxisScore`

A pair is axis-correct only if `certainty`, `temporality`, and `experiencer` all match.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from vexas.eval import AxisScore, axis_accuracy


def test_axis_all_correct():
    pairs = [(_f("fever"), _f("fever"))]  # both default positive/present/patient
    score = axis_accuracy(pairs)
    assert score.correct == 1
    assert score.total == 1
    assert score.accuracy == 1.0


def test_axis_certainty_mismatch_is_incorrect():
    gold = _f("lymphoma", certainty="hypothetical")
    pred = _f("lymphoma", certainty="positive")
    score = axis_accuracy([(gold, pred)])
    assert score.correct == 0
    assert score.total == 1
    assert score.accuracy == 0.0


def test_axis_experiencer_mismatch_is_incorrect():
    gold = _f("rash", experiencer="family_member")
    pred = _f("rash", experiencer="patient")
    assert axis_accuracy([(gold, pred)]).correct == 0


def test_axis_accuracy_none_when_no_pairs():
    assert axis_accuracy([]).accuracy is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL -- `ImportError: cannot import name 'AxisScore'`

- [ ] **Step 3: Add to `src/vexas/eval.py`**

```python
@dataclass
class AxisScore:
    correct: int = 0
    total: int = 0

    def __add__(self, other: "AxisScore") -> "AxisScore":
        return AxisScore(self.correct + other.correct, self.total + other.total)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.total if self.total else None


def axis_accuracy(pairs: list[tuple[EvalField, EvalField]]) -> AxisScore:
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vexas/eval.py tests/test_eval.py
git commit -m "Add assertion-axis accuracy over matched fields"
```

---

### Task 5: Unit and document scoring

Compose scalar + list + axis scoring across all categories of a unit, and
across all units of a document. Add the contamination-abstention check.

**Files:**
- Modify: `src/vexas/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: everything above, `EvalUnit`, `EvalDoc`
- Produces:
  - `vexas.eval.CONTAMINATION_MAX_FIELDS = 2`
  - `vexas.eval.CATEGORY_COUNTS` ordering constant: `("demographics", "timeline", "labs", "treatments", "prior_diagnoses", "outcome", "reason_for_publication")`
  - `vexas.eval.UnitScore` dataclass: `.per_category: dict[str, Counts]`, `.axes: AxisScore`, `.total_pred_fields: int`, and `.overall -> Counts` (sum of per-category)
  - `vexas.eval.score_unit(gold: EvalUnit, pred: EvalUnit) -> UnitScore`
  - `vexas.eval.DocScore` dataclass: `.pmc_id: str`, `.tier: str`, `.contamination_check: bool`, `.units: list[UnitScore]`, `.abstained: bool | None`, and `.overall -> Counts`, `.axes -> AxisScore`
  - `vexas.eval.score_document(gold: EvalDoc, pred: EvalDoc) -> DocScore`

`score_unit` treats `outcome` and `reason_for_publication` as one-key scalar
dicts (`{name: field}` when present, `{}` when None) so they flow through
`score_scalar_dict`. It sums axis pairs from every list category **and** the
matched scalar pairs. `total_pred_fields` counts every predicted field in the
unit (demographics values + list items + non-None singles); used for the
contamination check.

For a contamination document, `score_document` sets `abstained = (sum of
total_pred_fields across units) <= CONTAMINATION_MAX_FIELDS`; otherwise
`abstained = None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from vexas.eval import EvalDoc, EvalUnit, score_document, score_unit


def _unit(**kw):
    base = dict(patient_id="p1", text="t")
    base.update(kw)
    return EvalUnit(**base)


def test_score_unit_counts_across_categories():
    gold = _unit(
        demographics={"age": _f("68"), "sex": _f("male")},
        timeline=[_f("fever"), _f("chondritis")],
    )
    pred = _unit(
        demographics={"age": _f("68"), "sex": _f("male")},
        timeline=[_f("fever")],  # missed chondritis
    )
    score = score_unit(gold, pred)
    assert score.per_category["demographics"].tp == 2
    assert score.per_category["timeline"].tp == 1
    assert score.per_category["timeline"].fn == 1
    assert score.overall.tp == 3
    assert score.overall.fn == 1


def test_score_unit_axes_span_scalars_and_lists():
    gold = _unit(
        demographics={"age": _f("68")},
        timeline=[_f("lymphoma", certainty="hypothetical")],
    )
    pred = _unit(
        demographics={"age": _f("68")},
        timeline=[_f("lymphoma", certainty="positive")],  # axis miss
    )
    score = score_unit(gold, pred)
    assert score.axes.total == 2   # age pair + lymphoma pair
    assert score.axes.correct == 1  # only age's axes agree


def test_score_unit_counts_predicted_fields():
    pred = _unit(
        demographics={"age": _f("68"), "sex": _f("male")},
        treatments=[_f("ruxolitinib")],
        outcome=_f("alive"),
    )
    score = score_unit(_unit(), pred)
    assert score.total_pred_fields == 4


def test_contamination_pass_when_prediction_is_near_empty():
    gold_doc = EvalDoc("PMC1", None, "silver", None, True, [_unit()])
    pred_doc = EvalDoc("PMC1", None, "silver", None, True,
                       [_unit(demographics={"age": _f("70")})])
    result = score_document(gold_doc, pred_doc)
    assert result.abstained is True


def test_contamination_fail_when_prediction_invents_a_patient():
    gold_doc = EvalDoc("PMC1", None, "silver", None, True, [_unit()])
    pred_doc = EvalDoc("PMC1", None, "silver", None, True, [_unit(
        demographics={"age": _f("70"), "sex": _f("f"), "country": _f("US")},
        timeline=[_f("fever")],
    )])
    result = score_document(gold_doc, pred_doc)
    assert result.abstained is False


def test_non_contamination_document_has_no_abstention_verdict():
    gold_doc = EvalDoc("PMC1", None, "silver", None, False, [_unit()])
    pred_doc = EvalDoc("PMC1", None, "silver", None, False, [_unit()])
    assert score_document(gold_doc, pred_doc).abstained is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL -- `ImportError: cannot import name 'score_unit'`

- [ ] **Step 3: Add to `src/vexas/eval.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vexas/eval.py tests/test_eval.py
git commit -m "Add unit and document scoring with contamination check"
```

---

### Task 6: Corpus aggregation and text report

Aggregate document scores across the corpus, grouped by tier, and render a
plain-text report.

**Files:**
- Modify: `src/vexas/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `DocScore`, `Counts`, `AxisScore`, `CATEGORY_COUNTS`
- Produces:
  - `vexas.eval.TierReport` dataclass: `.tier: str`, `.per_category: dict[str, Counts]`, `.overall: Counts`, `.axes: AxisScore`, `.contamination_pass: int`, `.contamination_total: int`
  - `vexas.eval.score_corpus(docs: list[DocScore]) -> dict[str, TierReport]` -- keyed by tier
  - `vexas.eval.format_report(reports: dict[str, TierReport]) -> str`

`format_report` never prints a bare percentage without its denominator implied
by being labelled precision/recall (these are ratios of counts, not corpus
prevalence claims -- the no-bare-percentages rule targets clinical output, not
the tool's own diagnostic metrics). Show counts alongside: e.g.
`timeline   P 0.86 (12/14)  R 0.80 (12/15)  F1 0.83`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from vexas.eval import DocScore, UnitScore, format_report, score_corpus


def _doc_score(tier, per_cat, axes_correct=0, axes_total=0,
               contamination=False, abstained=None):
    unit = UnitScore(per_category=per_cat,
                     axes=AxisScore(axes_correct, axes_total),
                     total_pred_fields=0)
    return DocScore(pmc_id="PMC1", tier=tier, contamination_check=contamination,
                    units=[unit], abstained=abstained)


def test_corpus_groups_by_tier():
    docs = [
        _doc_score("silver", {"timeline": Counts(tp=3, fp=1, fn=0)}),
        _doc_score("gold", {"timeline": Counts(tp=2, fp=0, fn=1)}),
    ]
    reports = score_corpus(docs)
    assert set(reports) == {"silver", "gold"}
    assert reports["silver"].per_category["timeline"].tp == 3
    assert reports["gold"].per_category["timeline"].fn == 1


def test_corpus_sums_within_tier():
    docs = [
        _doc_score("silver", {"labs": Counts(tp=2, fp=0, fn=1)}),
        _doc_score("silver", {"labs": Counts(tp=1, fp=1, fn=0)}),
    ]
    labs = score_corpus(docs)["silver"].per_category["labs"]
    assert (labs.tp, labs.fp, labs.fn) == (3, 1, 1)


def test_corpus_tracks_contamination_passes():
    docs = [
        _doc_score("silver", {}, contamination=True, abstained=True),
        _doc_score("silver", {}, contamination=True, abstained=False),
    ]
    report = score_corpus(docs)["silver"]
    assert report.contamination_pass == 1
    assert report.contamination_total == 2


def test_format_report_shows_counts_with_ratios():
    docs = [_doc_score("silver", {"timeline": Counts(tp=12, fp=2, fn=3)})]
    text = format_report(score_corpus(docs))
    assert "silver" in text
    assert "timeline" in text
    assert "12/14" in text   # precision denominator
    assert "12/15" in text   # recall denominator
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: FAIL -- `ImportError: cannot import name 'score_corpus'`

- [ ] **Step 3: Add to `src/vexas/eval.py`**

```python
@dataclass
class TierReport:
    tier: str
    per_category: dict[str, Counts]
    overall: Counts
    axes: AxisScore
    contamination_pass: int
    contamination_total: int


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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vexas/eval.py tests/test_eval.py
git commit -m "Add tier-grouped corpus aggregation and text report"
```

---

### Task 7: `motet eval` CLI subcommand

Wire the scorer to the CLI. Loads gold + pred files, pairs them by filename,
scores, prints the report. Fully offline.

**Files:**
- Modify: `src/vexas/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `vexas.eval.load_eval_file`, `score_document`, `score_corpus`, `format_report`, `vexas.config.settings`
- Produces:
  - `vexas.cli.run_eval(gold_dir: Path, pred_dir: Path) -> str` -- returns the report text (also printed by `main`)
  - New `eval` subparser in `build_parser`, dispatched in `main`

`run_eval` iterates `gold_dir/*.json`; for each, if a same-named file exists in
`pred_dir` it scores the pair, otherwise it records the paper as skipped and
notes it in a trailing line. Default dirs: `settings.data_dir.parent / "eval" /
"gold"` and `.../ "eval" / "pred"` (i.e. the repo-root `eval/`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
import json
from pathlib import Path

from vexas.cli import run_eval


def _write(path, contamination=False, **units_kw):
    doc = {
        "pmc_id": path.stem, "pmid": None, "tier": "silver",
        "verified_by": None, "contamination_check": contamination,
        "units": [{
            "patient_id": f"{path.stem}-p1", "text": "A 68-year-old man.",
            "demographics": {"age": {"value": "68", "quote": "68-year-old"}},
            "timeline": [], "labs": [], "treatments": [],
            "prior_diagnoses": [], "outcome": None,
            "reason_for_publication": None,
            **units_kw,
        }],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_run_eval_scores_a_matched_pair(tmp_path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    _write(gold_dir / "PMC1.json")
    _write(pred_dir / "PMC1.json")
    report = run_eval(gold_dir, pred_dir)
    assert "silver" in report
    assert "demographics" in report


def test_run_eval_notes_missing_prediction(tmp_path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    _write(gold_dir / "PMC1.json")   # no matching pred
    report = run_eval(gold_dir, pred_dir)
    assert "PMC1" in report
    assert "no prediction" in report.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL -- `ImportError: cannot import name 'run_eval'`

- [ ] **Step 3: Add `run_eval` to `src/vexas/cli.py`**

Add near the top with the other imports:

```python
from pathlib import Path
```

Add this function (above `build_parser`):

```python
def _eval_dirs() -> tuple[Path, Path]:
    root = settings.data_dir.parent
    return root / "eval" / "gold", root / "eval" / "pred"


def run_eval(gold_dir: Path, pred_dir: Path) -> str:
    """Score every gold/pred pair and return the report text. Offline."""
    from vexas.eval import (
        format_report,
        load_eval_file,
        score_corpus,
        score_document,
    )

    doc_scores = []
    skipped = []
    for gold_path in sorted(Path(gold_dir).glob("*.json")):
        pred_path = Path(pred_dir) / gold_path.name
        if not pred_path.exists():
            skipped.append(gold_path.stem)
            continue
        gold = load_eval_file(gold_path)
        pred = load_eval_file(pred_path)
        doc_scores.append(score_document(gold, pred))

    report = format_report(score_corpus(doc_scores))
    if skipped:
        report += "\nno prediction yet: " + ", ".join(skipped)
    return report
```

- [ ] **Step 4: Add the subcommand to `build_parser`**

In `build_parser`, add before `return parser`:

```python
    sub.add_parser("eval", help="Score extraction against gold keys. Offline.")
```

- [ ] **Step 5: Dispatch it in `main`**

In `main`, add after the `check` block (before the `estimate` block):

```python
    if args.command == "eval":
        gold_dir, pred_dir = _eval_dirs()
        if not gold_dir.exists():
            print(f"No eval directory at {gold_dir}")
            return 1
        print(run_eval(gold_dir, pred_dir))
        return 0
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/vexas/cli.py tests/test_cli.py
git commit -m "Add offline 'motet eval' subcommand"
```

---

### Task 8: Gold-quote grounding data-quality test

A test that every gold quote is a literal substring of its unit text. Catches
annotation typos before they distort scores. Runs over whatever is in
`eval/gold/`, so it is vacuously green until Task 9 adds data.

**Files:**
- Create: `tests/test_eval_data.py`

**Interfaces:**
- Consumes: `vexas.gate.is_grounded`, `vexas.eval.load_eval_file`

- [ ] **Step 1: Write the test**

Create `tests/test_eval_data.py`:

```python
"""Data-quality guard for hand-authored answer keys.

Every gold quote must ground in its unit text, or the score is measuring an
annotation typo rather than the pipeline. Runs over eval/gold/*.json; vacuous
until answer keys exist.
"""

from pathlib import Path

import pytest

from vexas.eval import EvalField, load_eval_file
from vexas.gate import is_grounded

GOLD_DIR = Path(__file__).resolve().parents[1] / "eval" / "gold"
GOLD_FILES = sorted(GOLD_DIR.glob("*.json")) if GOLD_DIR.exists() else []


def _all_fields(unit) -> list[EvalField]:
    fields: list[EvalField] = list(unit.demographics.values())
    for cat in ("timeline", "labs", "treatments", "prior_diagnoses"):
        fields.extend(getattr(unit, cat))
    for single in (unit.outcome, unit.reason_for_publication):
        if single is not None:
            fields.append(single)
    return fields


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_every_gold_quote_grounds_in_unit_text(path):
    doc = load_eval_file(path)
    for unit in doc.units:
        for field in _all_fields(unit):
            assert is_grounded(field.quote, unit.text), (
                f"{path.stem} {unit.patient_id}: quote not in text: "
                f"{field.quote!r}"
            )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/pytest tests/test_eval_data.py -v`
Expected: PASS (no gold files yet -- collects zero parametrized cases, green)

- [ ] **Step 3: Commit**

```bash
git add tests/test_eval_data.py
git commit -m "Add gold-quote grounding data-quality guard"
```

---

### Task 9: Annotate a clean single-case report

The first real answer key + prediction. This task authors data, not code. The
values come from reading a real cached paper, so they cannot be pre-written
here; the steps define the process and the acceptance checks.

**Files:**
- Create: `eval/gold/<PMCID>.json`, `eval/pred/<PMCID>.json`

**Process:**

- [ ] **Step 1: Choose a clean single-patient VEXAS case report**

List candidates and read their text from the local cache (no network):

```bash
ls data/cache/pmc_xml/
```

```bash
.venv/bin/python -c "
from vexas.fetch.pmc import parse_article
from pathlib import Path
xml = Path('data/cache/pmc_xml/PMC8068627.cache').read_text()
a = parse_article(xml)
print(a.pmc_id, a.pmid, '::', a.title)
for name, body in a.sections.items():
    print('##', name); print(body[:1200])
"
```

Pick a paper that (a) describes exactly one patient, (b) states a UBA1 variant,
(c) has a clear onset narrative and at least one treatment. Record the PMC id.

- [ ] **Step 2: Author the gold answer key under the rigorous protocol**

Read the WHOLE paper. Author `eval/gold/<PMCID>.json` in the Task 1 shape.
Rules for this pass:
- Every field's `quote` copied verbatim from the section text; set `text` on
  the unit to the exact concatenated section text the pipeline would see
  (build it the same way `Article.full_text` does, or paste the relevant
  sections). Every quote MUST be a substring of `text`.
- Fill certainty/temporality/experiencer deliberately: mark ruled-out
  differentials `hypothetical`, family history `family_member` /
  `past_history`, absent-but-stated findings `negated`. Omit anything the
  paper does not state (do not emit a field for it).
- `tier`: `"silver"`, `verified_by`: `null`, `chosen_because`: one line.

- [ ] **Step 3: Author the prediction under the constrained protocol**

Separately -- WITHOUT looking at the gold file you just wrote -- act as the
pipeline's per-category extraction model on the same unit text. For each of the
five categories (demographics, timeline, labs, treatments, outcome), extract
only what a single-category prompt would surface, following the SHARED_RULES in
`src/vexas/extract/prompts.py` (verbatim quotes, no unit conversion, the
not_stated-vs-negated rule, the three axes). Write `eval/pred/<PMCID>.json` in
the same shape. This is the system-under-test's output.

- [ ] **Step 4: Verify the gold quotes ground**

Run: `.venv/bin/pytest tests/test_eval_data.py -v`
Expected: PASS for `<PMCID>` (every gold quote is a substring of unit text). If
any fails, fix the quote or the `text` field -- a grounding failure here is an
annotation bug, not a pipeline finding.

- [ ] **Step 5: Score it**

Run: `.venv/bin/motet eval`
Expected: a `silver` report block with non-trivial per-category numbers for
this one paper. Sanity-check that categories you know both files agree on show
as true positives.

- [ ] **Step 6: Commit**

```bash
git add eval/gold/<PMCID>.json eval/pred/<PMCID>.json
git commit -m "Add gold+pred answer key: single-case report <PMCID>"
```

---

### Task 10: Annotate a multi-patient case series

Stresses cross-patient contamination: the prediction must keep two patients'
facts separate.

**Files:**
- Create: `eval/gold/<PMCID>.json`, `eval/pred/<PMCID>.json`

**Process:**

- [ ] **Step 1: Find a case series (>= 2 patients) in the cache**

Scan for papers whose text describes multiple patients (search for "Patient 2",
"Case 2", "second patient"):

```bash
.venv/bin/python -c "
from vexas.fetch.pmc import parse_article
from pathlib import Path
for p in sorted(Path('data/cache/pmc_xml').glob('*.cache')):
    a = parse_article(p.read_text())
    ft = a.full_text.lower()
    if 'patient 2' in ft or 'case 2' in ft or 'second patient' in ft:
        print(p.stem, '::', a.title[:80])
"
```

Pick one with two clearly-separable patients. Record the PMC id.

- [ ] **Step 2: Author the gold key with one unit per patient**

Two `units`, `patient_id` suffixed `-p1` / `-p2`, each with its OWN verbatim
`text` slice (that patient's paragraphs only). Author fields per patient under
the rigorous protocol. The point of this paper is that per-patient facts never
cross; keep each patient's quotes inside that patient's `text`.

- [ ] **Step 3: Author the prediction per patient**

Act as the pipeline: segmentation first (split into the same two units), then
per-category extraction on each unit independently. Write both units into
`eval/pred/<PMCID>.json`. Do not let patient 2's treatment appear under
patient 1 -- if the constrained read genuinely would confuse them, record that
honestly; that is exactly the failure the eval is meant to surface.

- [ ] **Step 4: Verify grounding**

Run: `.venv/bin/pytest tests/test_eval_data.py -v`
Expected: PASS for `<PMCID>`.

- [ ] **Step 5: Score and commit**

```bash
.venv/bin/motet eval
git add eval/gold/<PMCID>.json eval/pred/<PMCID>.json
git commit -m "Add gold+pred answer key: case series <PMCID>"
```

---

### Task 11: Annotate the contamination false-positive

A paper that only mentions VEXAS in a differential (not a VEXAS case). The
correct extraction is near-empty; this scores abstention.

**Files:**
- Create: `eval/gold/PMC8056719.json`, `eval/pred/PMC8056719.json`

`PMC8056719` is a known false positive (an Adult-onset Still's disease review
that mentions VEXAS once; recorded in the project progress ledger).

**Process:**

- [ ] **Step 1: Confirm the paper is a non-case mention**

```bash
.venv/bin/python -c "
from vexas.fetch.pmc import parse_article
from pathlib import Path
a = parse_article(Path('data/cache/pmc_xml/PMC8056719.cache').read_text())
print(a.title)
import re
for m in re.finditer('vexas', a.full_text, re.I):
    s = max(0, m.start()-200); print('...', a.full_text[s:m.end()+200], '...')
"
```

Confirm there is no single VEXAS patient being reported.

- [ ] **Step 2: Author the gold key with `contamination_check: true`**

Set `"contamination_check": true`, `tier: "silver"`, one unit whose `text` is
the paper's full text and whose field collections are all empty / null (the
correct answer is "no VEXAS patient here"). `chosen_because`: "differential
mention, not a case -- abstention test".

- [ ] **Step 3: Author the prediction**

Act as the pipeline on this text. A faithful per-category read should surface
almost nothing about a specific VEXAS patient. Record whatever it genuinely
produces into `eval/pred/PMC8056719.json` (ideally empty; if the constrained
read wrongly invents fields, record that -- the abstention check will then
correctly fail and flag a real weakness).

- [ ] **Step 4: Verify grounding, score, commit**

```bash
.venv/bin/pytest tests/test_eval_data.py -v
.venv/bin/motet eval    # expect a 'contamination abstention' line
git add eval/gold/PMC8056719.json eval/pred/PMC8056719.json
git commit -m "Add gold+pred answer key: contamination abstention PMC8056719"
```

---

### Task 12: Annotate a variant-diversity paper

Adds coverage of a UBA1 variant not yet represented (aim for M41V or M41L to
complement the earlier M41T papers).

**Files:**
- Create: `eval/gold/<PMCID>.json`, `eval/pred/<PMCID>.json`

**Process:**

- [ ] **Step 1: Find a paper stating M41V or M41L**

```bash
.venv/bin/python -c "
from vexas.fetch.pmc import parse_article
from pathlib import Path
for p in sorted(Path('data/cache/pmc_xml').glob('*.cache')):
    a = parse_article(p.read_text())
    ft = a.full_text
    if 'M41V' in ft or 'p.Met41Val' in ft or 'M41L' in ft or 'p.Met41Leu' in ft:
        print(p.stem, '::', a.title[:80])
"
```

Pick a single-patient paper if possible (keeps this task focused). Record the
PMC id and which variant.

- [ ] **Step 2-6: Same process as Task 9**

Author gold (rigorous, whole-paper) and pred (constrained, per-category, blind
to gold), verify grounding, score, commit:

```bash
.venv/bin/pytest tests/test_eval_data.py -v
.venv/bin/motet eval
git add eval/gold/<PMCID>.json eval/pred/<PMCID>.json
git commit -m "Add gold+pred answer key: variant-diversity <PMCID>"
```

---

### Task 13: Document the harness and add a floor guard

README section explaining the eval, plus a lenient regression floor so a future
change that tanks extraction fails loudly.

**Files:**
- Modify: `README.md`
- Create: `tests/test_eval_floor.py`

- [ ] **Step 1: Add a README section**

Add after the "How it stays honest" section:

```markdown
## How good is the extraction

`motet eval` scores the pipeline's per-field output against hand-authored
answer keys in `eval/gold/`, fully offline. It reports precision, recall, and
F1 per field category, a separate assertion-axis accuracy number (how often
certainty / temporality / experiencer are right on the fields it did find), and
an abstention check on papers that only mention the disease in a differential.

Answer keys are tiered: `silver` (Claude-authored under a stricter whole-paper
protocol) and `gold` (human-verified). The two tiers are reported separately
and never blended. Silver numbers carry an inherent inflation risk because the
same model can author both the key and the by-hand prediction; the gold tier
exists to remove that risk for any published figure.

```bash
motet eval
```
```

- [ ] **Step 2: Write the floor test**

Create `tests/test_eval_floor.py`:

```python
"""Regression floor: overall silver F1 must not collapse.

Lenient by design -- it guards against a change that badly breaks extraction,
not against small movements. Raise FLOOR as real quality is established.
Skips when there is not yet enough scored data.
"""

from pathlib import Path

from vexas.cli import run_eval

FLOOR = 0.5


def test_overall_silver_f1_stays_above_floor():
    root = Path(__file__).resolve().parents[1]
    gold_dir = root / "eval" / "gold"
    pred_dir = root / "eval" / "pred"
    if not gold_dir.exists():
        return  # nothing to score yet
    from vexas.eval import load_eval_file, score_corpus, score_document

    docs = []
    for gold_path in sorted(gold_dir.glob("*.json")):
        pred_path = pred_dir / gold_path.name
        if pred_path.exists():
            docs.append(score_document(
                load_eval_file(gold_path), load_eval_file(pred_path)
            ))
    reports = score_corpus(docs)
    silver = reports.get("silver")
    if silver is None or silver.overall.f1 is None:
        return  # not enough matched data yet
    assert silver.overall.f1 >= FLOOR, (
        f"overall silver F1 {silver.overall.f1:.2f} fell below {FLOOR}"
    )
```

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all prior tests plus the new floor test)

- [ ] **Step 4: Commit**

```bash
git add README.md tests/test_eval_floor.py
git commit -m "Document motet eval and add a lenient regression floor"
```

---

## Self-Review Notes

- **Spec coverage:** data layout (Task 1), tiering + honesty note (Tasks 1, 13),
  scorer purity (Tasks 1-6), matching algorithm scalar+list (Tasks 2-3),
  lenient headline + separate axes (Tasks 4-6), contamination check (Task 5),
  `motet eval` offline (Task 7), gold-quote grounding test (Task 8), starter
  coverage of 4 stress papers (Tasks 9-12), floor test (Task 13). All spec
  sections map to a task.
- **Non-goals honored:** no segmentation scoring (unit text is given); no
  aggregate-pattern scoring (that is validate.py); v1 scores identifying field
  + axes only, sub-components (sequence_index, lab values, treatment-response
  components) deferred -- the JSON shape already carries them so no
  re-annotation is needed to add later.
- **Type consistency:** `Counts`, `AxisScore`, `EvalField`, `EvalUnit`,
  `EvalDoc`, `UnitScore`, `DocScore`, `TierReport` used consistently across
  tasks; `score_scalar_dict` / `score_list` / `axis_accuracy` / `score_unit` /
  `score_document` / `score_corpus` / `format_report` / `run_eval` signatures
  match between their producing task and consumers.
