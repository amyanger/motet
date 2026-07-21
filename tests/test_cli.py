import pytest

from vexas.cli import build_parser, check_environment, estimate_cost


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
