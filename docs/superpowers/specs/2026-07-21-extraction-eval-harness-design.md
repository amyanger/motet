# Extraction Evaluation Harness — Design

**Date:** 2026-07-21
**Project:** Motet (rare-disease case-report miner)
**Status:** Approved design, pending implementation plan

## Goal

Measure how accurately the extraction pipeline pulls structured facts out of
real case reports, field by field, against a human-checkable answer key.
Produce precision / recall / F1 per category and overall, plus a separate
assertion-axis accuracy number, so we can state -- with evidence -- how good
the extraction actually is rather than only that the plumbing is correct.

## Why this exists (the gap)

Every existing test validates machinery against stubs. 211 tests pass, but the
pipeline has never produced a single record from a real paper, and nothing
measures whether the facts it would extract are *right*. `validate.py` checks a
different thing: whether the aggregate output recovers already-published
findings (the genotype gradient). That is downstream of, and assumes, correct
per-field extraction. This harness measures the per-field extraction itself,
which is the more fundamental and currently unmeasured property.

For a tool whose entire pitch is "every claim is grounded and trustworthy," a
measured extraction-accuracy number is the single most credibility-building
artifact it can carry.

## Non-goals

- **Not** a segmentation evaluation. The harness assumes each patient unit's
  text is already correctly split; per-patient text spans are provided in the
  gold file. Scoring `detect_patient_count` is a possible later addition.
- **Not** an aggregate-pattern evaluation. That is `validate.py`'s job.
- **Not** a live-cost or throughput benchmark.
- **Not** a replacement for eventual human/clinician verification. It is the
  harness and format that make such verification cheap to fold in.

## Ground-truth model: tiered, hybrid

Every answer-key file is tagged with a `tier`:

- `silver` -- Claude-authored under a deliberately stricter protocol than the
  pipeline (whole paper read at once, all categories together, no per-category
  tunnel vision, no substring-gate shortcut, authored **blind to the
  pipeline's prediction** for that paper).
- `gold` -- a human domain expert has reviewed and corrected the entry, and
  filled in `verified_by`.

The scorer reports the two tiers as **separate numbers** and never blends them.
Promotion from silver to gold is editing one file's `tier` and `verified_by`
fields -- no rework, no reformatting.

### Honest limitation, recorded in the output

On this project the predictions are produced by Claude acting as the extraction
model by hand (the creator's workflow never uses an API key). Because the same
model authors both the silver key and the by-hand prediction, silver-tier
agreement carries an inherent inflation risk. Two things mitigate it, and the
harness states it plainly rather than hiding it:

1. The two passes use genuinely different protocols (rigorous whole-paper
   annotation vs. constrained per-category extraction) and are authored
   independently, so the silver key can still catch the pipeline's signature
   failure modes (per-category tunnel vision, cross-patient bleed, paraphrase
   creep).
2. The gold tier exists precisely to remove this risk for any published number.

A forker who wants a publishable headline figure should regenerate the
prediction files from a real `motet run` and promote keys to gold. This does
not change the creator's by-hand workflow.

## Data layout and format

A new top-level `eval/` directory:

```
eval/
  gold/PMC<id>.json     one answer key per paper
  pred/PMC<id>.json     the pipeline's prediction per paper, same shape
```

### File shape (both gold and pred)

```json
{
  "pmc_id": "PMC8068627",
  "pmid": "34......",
  "tier": "silver",
  "verified_by": null,
  "chosen_because": "clean single-case report, M41T variant",
  "notes": "optional annotator notes",
  "units": [
    {
      "patient_id": "PMC8068627-p1",
      "text": "<the exact per-patient source text the pipeline sees>",
      "demographics": {
        "age": {"value": "68", "unit_as_written": "years",
                "quote": "A 68-year-old man",
                "certainty": "positive", "temporality": "present",
                "experiencer": "patient"}
      },
      "timeline": [
        {"symptom": {"value": "fever", "quote": "recurrent fevers",
                     "certainty": "positive", "temporality": "present",
                     "experiencer": "patient"},
         "sequence_index": 1}
      ],
      "labs": [ ... ],
      "treatments": [ ... ],
      "prior_diagnoses": [ ... ],
      "outcome": { ... } | null,
      "reason_for_publication": { ... } | null
    }
  ]
}
```

Notes on the shape:

- It mirrors `PatientRecord` but as plain JSON so it is hand-editable and
  diff-reviewable. It is deliberately not the pydantic model on disk.
- `text` is included per unit because the pipeline scores extraction *given a
  unit*; for a case series the annotator defines the canonical split here so
  segmentation variance does not contaminate the extraction score. `text` must
  be a verbatim slice of the cached JATS full text so every gold quote grounds.
- A gold quote that does not ground in `text` is an annotation bug; the harness
  surfaces it (see Testing).

## The scorer (`src/vexas/eval.py`)

A pure, deterministic module. No network, no model calls, no key.

### Public surface

- `load_eval_file(path) -> EvalDoc` -- parse and lightly validate a gold/pred
  JSON file.
- `score_unit(gold_unit, pred_unit) -> UnitScore` -- score one patient unit.
- `score_document(gold_doc, pred_doc) -> DocScore` -- score one paper.
- `score_corpus(pairs) -> CorpusReport` -- aggregate across papers, grouped by
  tier.

### Matching algorithm

Fields are compared after normalization that reuses the pipeline's own code so
the yardstick and the system agree on what "equal" means:

- Strings: `gate.normalize_for_match` (case / whitespace / publisher-punctuation
  insensitive).
- Variants: `normalize.hgvs.normalize_variant` (so `p.Met41Thr` == `M41T`).
- Numerics where a unit is involved: `normalize.units.parse_numeric` (so `68`
  and `68 years` agree on the number).

**Scalar fields** (age, sex, country, uba1_variant, vaf, outcome,
reason_for_publication) match by key. A predicted scalar is a *true positive*
if the gold has that key and the normalized values are equal, a *false
positive* if gold lacks it or values differ, a *false negative* if gold has it
and the prediction does not.

**List fields** (timeline, treatments, labs, prior_diagnoses) align predicted
items to gold items greedily by their identifying surface value (symptom name /
drug / analyte / diagnosis text, normalized). Each aligned pair is a true
positive. Unaligned predicted items are false positives; unaligned gold items
are false negatives. Alignment is one-to-one; ties resolve by first match in
gold order (deterministic).

### Metrics

- **Headline: lenient.** A field counts as a true positive if its identifying
  value matches. Precision = TP / (TP + FP); recall = TP / (TP + FN);
  F1 = harmonic mean. Reported per category and overall.
- **Assertion-axis accuracy, reported separately.** Over the set of matched
  (true-positive) fields, the fraction whose `certainty`, `temporality`, and
  `experiencer` all equal the gold's. This isolates "labeled it right" from
  "found it," mirroring the schema's own orthogonal design. A field can be a
  headline true positive and an axis miss; that is the point.
- **Contamination / abstention check.** For a file flagged
  `"contamination_check": true` (the differential-mention non-case papers), the
  correct behavior is to emit essentially nothing. Scored as pass if the
  prediction's total field count is at or below a small threshold (e.g. <= 2),
  fail otherwise. Reported as its own line, not folded into P/R/F1. A
  contamination paper's field-level false positives are also deliberately
  counted into the tier's aggregate precision, in addition to the separate
  abstention line, because spurious extraction from a non-case paper is a real
  precision failure.

Quotes are **not** required to match between prediction and gold. The fact is
what is scored, and the substring gate already guarantees a predicted quote
grounds to source. (The harness may still assert that every *gold* quote
grounds in the unit `text`, as an annotation-quality check.)

## How it runs

A `motet eval` CLI subcommand:

- Loads every pair in `eval/gold` + `eval/pred` (matched by filename).
- Runs `score_corpus`.
- Prints a per-category and overall report, grouped by tier, plus the
  assertion-axis and contamination lines.
- Fully offline. Needs no API key. Spends nothing.

A prediction file may be missing while an answer key is being drafted; the
harness reports that paper as "no prediction" rather than crashing.

## Testing strategy

- The scorer is unit-tested with **synthetic** gold/pred pairs (TDD, failing
  test first): exact match, missed field (FN), hallucinated field (FP), value
  mismatch, list alignment with reordering, an axis-only miss, and the
  contamination-abstention pass/fail. These guard the harness independent of
  any real data.
- A data-quality test asserts every real gold quote grounds in its unit `text`
  via `gate.is_grounded`. This catches annotation typos.
- A later floor test (added once real numbers exist) asserts overall silver F1
  does not drop below a chosen floor, so a prompt or code change that degrades
  extraction fails loudly. Kept lenient at first.

## Starter coverage (this session)

Build the harness plus a first set of ~4 hand-authored papers, each chosen to
stress one thing:

1. A clean single-case report (baseline).
2. A multi-patient case series (the cross-patient-contamination test).
3. The known differential-mention false positive, e.g. PMC8056719 (the
   abstention test).
4. One chosen for variant diversity (M41V or M41L to complement M41T).

All four have their JATS text already cached in `data/cache/pmc_xml/`, so the
whole loop runs offline. Expanding toward the 6-8 papers discussed is then
purely adding more `eval/gold` + `eval/pred` files -- no code change.

## What this buys

After this ships, the project can say, with a number behind it, how accurately
it extracts each category of fact from real papers, and how often it gets the
certainty/temporality/experiencer labels right -- and it can detect the day a
change makes that worse. That is the bridge from "provably-correct plumbing" to
"a result a scientist would trust."
