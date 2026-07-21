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

from vexas.config import settings

# Pricing, claude-opus-4-8 (verified 2026-07-18). Update alongside config.model.
USD_PER_INPUT_TOKEN = 5.00 / 1_000_000
USD_PER_OUTPUT_TOKEN = 25.00 / 1_000_000
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

#: Rough tokens-per-character for English clinical prose. Good enough for a
#: pre-flight estimate; the real number comes from usage after a run.
CHARS_PER_TOKEN = 4

#: Calls per report: 5 category prompts + ~30 verification calls.
CATEGORY_CALLS = 5
VERIFY_CALLS = 30
#: Output tokens per call -- extraction is verbose, verification is a verdict.
CATEGORY_OUTPUT = 800
VERIFY_OUTPUT = 50


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

    for chars in char_counts:
        tokens = chars / CHARS_PER_TOKEN
        calls = CATEGORY_CALLS + VERIFY_CALLS

        # Two cache families (extraction prefix, verification prefix):
        # one write each, everything else reads.
        writes = 2
        reads = calls - writes
        total_input += tokens * (
            writes * CACHE_WRITE_MULTIPLIER + reads * CACHE_READ_MULTIPLIER
        )
        uncached_input += tokens * calls

        total_output += (
            CATEGORY_CALLS * CATEGORY_OUTPUT + VERIFY_CALLS * VERIFY_OUTPUT
        )

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
            "'vexas estimate --limit 25' to price a run first."
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vexas",
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
        print("Try: vexas estimate --limit 25")
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
    if result.gate_stats:
        print(f"gate rejection rate: "
              f"{result.gate_stats.rejection_rate * 100:.1f}%")
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

    if args.command == "serve":
        from vexas.web.app import create_app
        print(f"\nOpen http://localhost:{args.port} in Safari")
        create_app(patterns=patterns, records=result.records).run(
            debug=True, port=args.port
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
