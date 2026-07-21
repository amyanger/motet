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
