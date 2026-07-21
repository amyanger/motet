"""Regression floor: overall silver F1 must not collapse.

Lenient by design -- it guards against a change that badly breaks extraction,
not against small movements. Raise FLOOR as real quality is established.
Skips when there is not yet enough scored data.
"""

from pathlib import Path

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
