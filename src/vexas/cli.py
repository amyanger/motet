"""Command-line interface.

This tool is bring-your-own-key: users supply ANTHROPIC_API_KEY and pay for
their own runs. Two design consequences:

  1. A missing key must produce an actionable message, not a stack trace.
     It is the most likely first-run failure.
  2. `estimate` runs without any key at all, so a user can see the price
     before spending anything.
"""

import argparse
import sys
from pathlib import Path

from vexas.config import settings

# Pricing, claude-opus-4-8 (verified 2026-07-18). Update alongside config.model.
USD_PER_INPUT_TOKEN = 5.00 / 1_000_000
USD_PER_OUTPUT_TOKEN = 25.00 / 1_000_000
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: Rough tokens-per-character for English clinical prose. Good enough for a
#: pre-flight estimate; the real number comes from usage after a run.
CHARS_PER_TOKEN = 4

#: Extraction: 5 category prompts per report.
CATEGORY_CALLS = 5
#: Output tokens per category call -- extraction is verbose.
CATEGORY_OUTPUT = 800

#: Verification: `motet run` verifies every surviving field with a blind
#: second pass (verify.py). Roughly 30 fields per report, so this is the
#: highest-volume stage and dominates the call count. Each call returns a
#: short {"supported", "reason"} object.
VERIFY_CALLS = 30
VERIFY_OUTPUT = 40

#: Segmentation: one call per report to detect patient count / anchors.
SEGMENT_CALLS = 1
SEGMENT_OUTPUT = 60

#: Relevance gate: one call per report before extraction (relevance.py). Priced
#: at the extraction rates here as an upper bound -- it actually runs on a
#: cheaper model, and excluded papers skip extraction entirely, so the real
#: run costs less than this estimate. Modeled as a single uncached call.
GATE_CALLS = 1
GATE_OUTPUT = 60


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


def estimate_cost(char_counts: list[int]) -> dict:
    """Estimate the cost of a run, without calling the API.

    Models the caching structure from extract/prompts.build_prompt: each
    report's text is written to cache once per prompt family and read back
    on every subsequent call.
    """
    if not char_counts:
        return {"reports": 0, "total_usd": 0.0, "uncached_usd": 0.0,
                "input_tokens": 0, "output_tokens": 0}

    total_input = 0.0
    uncached_input = 0.0
    total_output = 0

    def cached_family(tokens: float, calls: int) -> float:
        # Each cache family (extraction, verification) re-sends the same
        # source: one write, the rest read back at 0.1x.
        reads = calls - 1
        return tokens * (CACHE_WRITE_MULTIPLIER + reads * CACHE_READ_MULTIPLIER)

    for chars in char_counts:
        tokens = chars / CHARS_PER_TOKEN

        # Extraction and verification are two separate cache families (their
        # prefixes differ), so each pays its own cache write.
        total_input += cached_family(tokens, CATEGORY_CALLS)
        total_input += cached_family(tokens, VERIFY_CALLS)
        # Segmentation and the relevance gate are single uncached calls.
        total_input += tokens * SEGMENT_CALLS
        total_input += tokens * GATE_CALLS
        uncached_input += tokens * (
            CATEGORY_CALLS + VERIFY_CALLS + SEGMENT_CALLS + GATE_CALLS
        )

        total_output += CATEGORY_CALLS * CATEGORY_OUTPUT
        total_output += VERIFY_CALLS * VERIFY_OUTPUT
        total_output += SEGMENT_CALLS * SEGMENT_OUTPUT
        total_output += GATE_CALLS * GATE_OUTPUT

    output_usd = total_output * USD_PER_OUTPUT_TOKEN
    return {
        "reports": len(char_counts),
        "input_tokens": int(total_input),
        "output_tokens": total_output,
        "total_usd": round(total_input * USD_PER_INPUT_TOKEN + output_usd, 2),
        "uncached_usd": round(
            uncached_input * USD_PER_INPUT_TOKEN + output_usd, 2
        ),
    }


def check_environment() -> list[str]:
    """Return human-readable setup problems. Empty list means ready to run."""
    problems = []
    if not settings.anthropic_api_key:
        problems.append(
            "ANTHROPIC_API_KEY is not set.\n"
            "  1. Get a key at https://console.anthropic.com/settings/keys\n"
            "  2. Add credits at https://console.anthropic.com/settings/billing\n"
            "  3. cp .env.example .env  and put the key in it\n"
            "  Runs are billed to your own account. Use "
            "'motet estimate --limit 25' to price a run first."
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motet",
        description="Mine rare-disease case reports into traceable patterns.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_limit(p):
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Process only the first N reports. Omit for the full corpus. "
                 "Use a small value (25-50) for a cheap demo run.",
        )

    estimate = sub.add_parser(
        "estimate", help="Estimate run cost. Needs no API key."
    )
    add_limit(estimate)

    run = sub.add_parser("run", help="Fetch, extract, and aggregate.")
    add_limit(run)
    run.add_argument("--min-support", type=int, default=3,
                     help="Reports required before a pattern is surfaced.")
    run.add_argument("--yes", action="store_true",
                     help="Skip the cost confirmation prompt.")

    serve = sub.add_parser("serve", help="Launch the local web UI.")
    add_limit(serve)
    serve.add_argument("--port", type=int, default=5001)
    serve.add_argument("--min-support", type=int, default=3)

    sub.add_parser("check", help="Verify setup without spending anything.")

    sub.add_parser("eval", help="Score extraction against gold keys. Offline.")
    return parser


def _corpus_char_counts(limit: int | None) -> list[int]:
    """Fetch (cached) full text and measure it. Costs nothing but NCBI calls."""
    from vexas.fetch.pmc import fetch_full_text, parse_article
    from vexas.fetch.pubmed import search_pmc_ids

    ids = search_pmc_ids()
    counts = []
    for raw_id in ids[:limit] if limit else ids:
        try:
            article = parse_article(fetch_full_text(f"PMC{raw_id}"))
        except Exception:
            continue
        if article.sections:
            counts.append(len(article.full_text))
    return counts


def _print_estimate(est: dict) -> None:
    print(f"  reports:        {est['reports']}")
    print(f"  input tokens:   {est['input_tokens']:,} (after caching)")
    print(f"  output tokens:  {est['output_tokens']:,}")
    print(f"  estimated cost: ${est['total_usd']:.2f}")
    print(f"  without caching: ${est['uncached_usd']:.2f}")
    print("  Estimate only -- actual usage is reported after the run.")


def format_triage_summary(triage) -> str:
    """One-line counts plus the excluded and uncertain papers with quotes.

    Counts only -- no bare percentages (global constraint). Excluded =
    mention_only + review; uncertain papers are surfaced for human review.
    """
    excluded = triage.mention_only + triage.review
    lines = [
        f"triage: {triage.total} full-text papers -- "
        f"{triage.case_report} case reports, {triage.uncertain} uncertain, "
        f"{excluded} excluded"
    ]
    for entry in triage.excluded + triage.flagged:
        quote = entry.quote[:80] if entry.quote else "(no quote)"
        lines.append(f"  {entry.pmc_id} [{entry.label}]: {quote}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check":
        problems = check_environment()
        if problems:
            print("Setup incomplete:\n")
            for p in problems:
                print(p)
            return 1
        print("Ready. API key found.")
        print("Try: motet estimate --limit 25")
        return 0

    if args.command == "eval":
        gold_dir, pred_dir = _eval_dirs()
        if not gold_dir.exists():
            print(f"No eval directory at {gold_dir}")
            return 1
        print(run_eval(gold_dir, pred_dir))
        return 0

    if args.command == "estimate":
        print("Measuring corpus (no API calls, no cost)...")
        _print_estimate(estimate_cost(_corpus_char_counts(args.limit)))
        return 0

    # run and serve both spend money -- check setup first.
    problems = check_environment()
    if problems:
        print("Cannot run:\n")
        for p in problems:
            print(p)
        return 1

    if args.command == "run" and not args.yes:
        est = estimate_cost(_corpus_char_counts(args.limit))
        print("Estimated cost of this run:")
        _print_estimate(est)
        if input("\nProceed? [y/N] ").strip().lower() != "y":
            print("Aborted. Nothing was spent.")
            return 1

    from vexas.aggregate.odyssey import rank_misdiagnoses
    from vexas.aggregate.sequence import rank_precedence
    from vexas.aggregate.treatment import rank_by_line
    from vexas.extract.client import CACHE_STATS
    from vexas.pipeline import run as run_pipeline

    result = run_pipeline(limit=args.limit)
    print(f"\n{len(result.records)} records from "
          f"{result.full_text_count} of {result.corpus_size} reports")
    if result.triage:
        print(format_triage_summary(result.triage))
    if result.gate_stats:
        print(f"gate rejection rate: "
              f"{result.gate_stats.rejection_rate * 100:.1f}%")
    if result.verify_stats:
        print(f"verify rejection rate: "
              f"{result.verify_stats.rejection_rate * 100:.1f}%")
    print(f"cache: {CACHE_STATS['read']:,} read, "
          f"{CACHE_STATS['write']:,} written, "
          f"{CACHE_STATS['uncached']:,} uncached")
    if CACHE_STATS["read"] == 0 and len(result.records) > 1:
        print("WARNING: no cache reads. Prompt block ordering may be broken "
              "-- see extract/prompts.build_prompt. This run cost ~5x more "
              "than it should have.")

    patterns = {
        "sequence": rank_precedence(result.records, args.min_support),
        "treatment": rank_by_line(result.records, args.min_support),
        "odyssey": rank_misdiagnoses(result.records, args.min_support),
    }
    for kind, found in patterns.items():
        print(f"{kind}: {len(found)} patterns")

    from vexas.validate import KNOWN_FINDINGS, check_genotype_gradient
    genotypes = check_genotype_gradient(result.records)
    if genotypes:
        print("\ngenotype gradient "
              "(validation against published NIH-cohort findings):")
        for variant, count in sorted(genotypes.items(),
                                     key=lambda kv: kv[1].n, reverse=True):
            prior = KNOWN_FINDINGS.get(variant)
            line = f"  {variant}: {count.as_text()}"
            if prior:
                line += f"  [{prior}]"
            print(line)

    if args.command == "serve":
        from vexas.web.app import create_app
        print(f"\nOpen http://localhost:{args.port} in Safari")
        create_app(patterns=patterns, records=result.records).run(
            debug=True, port=args.port
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
