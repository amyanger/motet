import json

import pytest

from vexas.cli import build_parser, check_environment, estimate_cost, run_eval
from vexas.cli import format_triage_summary
from vexas.pipeline import TriageReport
from vexas.relevance import RelevanceLabel, RelevanceVerdict


def test_limit_defaults_to_none_meaning_full_corpus():
    args = build_parser().parse_args(["run"])
    assert args.limit is None


def test_limit_is_parsed_as_int():
    args = build_parser().parse_args(["run", "--limit", "25"])
    assert args.limit == 25


def test_estimate_subcommand_accepts_limit():
    args = build_parser().parse_args(["estimate", "--limit", "50"])
    assert args.command == "estimate"
    assert args.limit == 50


def test_serve_subcommand_has_a_port():
    args = build_parser().parse_args(["serve"])
    assert args.port == 5001


def test_estimate_scales_with_corpus_size():
    one = estimate_cost([30_000])
    ten = estimate_cost([30_000] * 10)
    assert ten["total_usd"] > one["total_usd"]


def test_estimate_reports_cached_and_uncached_separately():
    """Users need to see what caching saves them."""
    result = estimate_cost([30_000] * 10)
    assert result["uncached_usd"] > result["total_usd"]
    assert result["reports"] == 10


def test_estimate_of_empty_corpus_is_zero():
    result = estimate_cost([])
    assert result["total_usd"] == 0.0
    assert result["reports"] == 0


def test_estimate_includes_the_verification_pass():
    """`vexas run` now verifies every field, so the estimate must price
    the verify calls -- output tokens exceed the five category calls alone."""
    est = estimate_cost([30_000])
    assert est["output_tokens"] > 5 * 800


def test_check_environment_reports_missing_api_key(monkeypatch):
    from dataclasses import replace
    from vexas.cli import settings
    monkeypatch.setattr("vexas.cli.settings",
                        replace(settings, anthropic_api_key=None))
    problems = check_environment()
    assert any("ANTHROPIC_API_KEY" in p for p in problems)


def test_check_environment_message_is_actionable(monkeypatch):
    """A first-run user must be told what to do, not just what is wrong."""
    from dataclasses import replace
    from vexas.cli import settings
    monkeypatch.setattr("vexas.cli.settings",
                        replace(settings, anthropic_api_key=None))
    joined = " ".join(check_environment())
    assert ".env" in joined
    assert "console.anthropic.com" in joined


def test_check_environment_passes_when_key_present(monkeypatch):
    from dataclasses import replace
    from vexas.cli import settings
    monkeypatch.setattr("vexas.cli.settings",
                        replace(settings, anthropic_api_key="sk-ant-x"))
    assert check_environment() == []


def test_estimate_does_not_require_an_api_key(monkeypatch):
    """Estimation is local arithmetic -- users check the price first."""
    from dataclasses import replace
    from vexas.cli import settings
    monkeypatch.setattr("vexas.cli.settings",
                        replace(settings, anthropic_api_key=None))
    assert estimate_cost([30_000])["total_usd"] > 0


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


def test_estimate_includes_the_gate_call():
    # Adding a per-report gate call can only raise the estimate, never lower it.
    from vexas import cli
    baseline = (cli.CATEGORY_CALLS, cli.VERIFY_CALLS, cli.SEGMENT_CALLS)
    est = estimate_cost([40_000])
    assert est["reports"] == 1
    # One gate call of ~10k tokens adds measurable input; assert it is priced.
    assert est["total_usd"] > 0
    # Output budget now includes the gate's short JSON.
    assert est["output_tokens"] >= (
        cli.CATEGORY_CALLS * cli.CATEGORY_OUTPUT
        + cli.VERIFY_CALLS * cli.VERIFY_OUTPUT
        + cli.SEGMENT_CALLS * cli.SEGMENT_OUTPUT
        + cli.GATE_CALLS * cli.GATE_OUTPUT
    )


def test_triage_summary_is_counts_only_no_percentage():
    report = TriageReport()
    report.record("PMC1", RelevanceVerdict(RelevanceLabel.CASE_REPORT, "q", "r"))
    report.record("PMC2", RelevanceVerdict(
        RelevanceLabel.MENTION_ONLY, "VEXAS was considered", "differential"))
    report.record("PMC3", RelevanceVerdict(RelevanceLabel.UNCERTAIN, "", "unclear"))
    text = format_triage_summary(report)
    assert "%" not in text
    assert "3" in text  # total
    assert "PMC2" in text  # excluded paper listed
    assert "VEXAS was considered" in text  # with its quote
    assert "PMC3" in text  # uncertain paper surfaced
