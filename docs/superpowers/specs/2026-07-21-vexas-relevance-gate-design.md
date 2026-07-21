# VEXAS Relevance Gate -- Design

Date: 2026-07-21
Status: Approved (brainstorm), pending implementation plan
Feature branch (proposed): `relevance-gate`

## Problem

The corpus query (`pmc_query` in `config.py`) returns every open-access paper
that mentions "VEXAS", not only papers that report a VEXAS patient. Reviews,
editorials, and papers that name VEXAS in a differential all come back. The
extraction prompts pull "any single patient's case report", so a false-positive
paper yields a full, wrong-disease patient record that inflates every downstream
denominator.

This was demonstrated concretely by the eval harness (Task 11, PMC8056719): an
Adult-onset Still's disease paper that mentions VEXAS once in a differential was
extracted into a full 37-field record, and the abstention check scored 0/1.

There is currently no stage that decides whether a paper is actually about VEXAS
before extraction runs. This design adds one.

## Goals

- Stop non-VEXAS papers from producing patient records.
- Make the decision auditable: every triage call carries a verbatim supporting
  quote and a reason, in keeping with the tool's traceability discipline.
- Turn the raw corpus count into an honest, reported breakdown
  (fetched -> case reports extracted / uncertain / excluded), refining the
  denominator rather than silently filtering.
- Be provable offline (creator runs no-spend); validate the model's judgment out
  of band, as with existing smoke tests.

## Non-goals

- Per-patient-unit gating. The corpus false positives are whole papers; splitting
  a genuine VEXAS series and gating individual patients is YAGNI here.
- A scored eval of the gate itself (label-accuracy metric). Possible later; out of
  scope now.
- Web UI surfacing of excluded/uncertain papers. Deferred; the triage data lands
  on `PipelineResult` and prints in the `motet run` summary, which is enough to
  make the gate usable and auditable.
- Changing the extraction prompts or the existing field-level substring gate.

## Design

### Placement

A new stage 2.5, per paper, inside `pipeline.run()`, between the
`article.sections` check and extraction:

```
fetch -> parse -> [has sections?] -> RELEVANCE GATE -> segment -> extract -> gate -> verify -> normalize -> aggregate -> serve
```

`records_from_article` is unchanged; the gate lives in `run()` so that function
stays focused on per-article extraction.

### New module: `src/vexas/relevance.py`

Mirrors the shape of `verify.py`.

- `RelevanceLabel` (enum): `CASE_REPORT`, `MENTION_ONLY`, `REVIEW`, `UNCERTAIN`.
- `RelevanceVerdict` (frozen dataclass): `label: RelevanceLabel`,
  `quote: str` (verbatim), `reason: str`.
- `build_relevance_prompt(text) -> list[dict]`: a neutral instruction block (task
  + JSON shape) and the source-text block. It does NOT reuse extraction's
  `SHARED_RULES` (see Prompt below), so it does not share extraction's cached
  prefix. It uses NO cache breakpoint at all: the gate is one call per paper, and
  a cache write (1.25x) only pays off when later calls read it back at 0.1x, which
  never happens here. So the gate is a single uncached call.
- `classify_relevance(article, complete_fn=...) -> RelevanceVerdict`: one model
  call over `article.full_text`; parses `{label, quote, reason}`.

Label semantics (encoded in the prompt):

- `CASE_REPORT` -- reports at least one patient diagnosed with VEXAS. **Precedence
  rule:** a paper that both reviews the literature AND reports a VEXAS patient is
  `CASE_REPORT`.
- `MENTION_ONLY` -- VEXAS appears only in a differential, as background, or in an
  editorial; no VEXAS patient is reported.
- `REVIEW` -- review / meta-analysis with no newly reported patient.
- `UNCERTAIN` -- the model is not confident, OR the fail-closed path fired.

### Fail-closed behavior

`classify_relevance` returns `UNCERTAIN` (never `CASE_REPORT`, never a silent
exclude) when any of these hold:

- `complete_fn` raises.
- The response is unparseable JSON or the label is not one of the four.
- The returned `quote` is not a verbatim substring of the source (reuses the same
  whitespace-collapsing grounding check the field-level gate uses).

`UNCERTAIN` papers are surfaced for human review: not extracted, not silently
dropped. A gate outage therefore stalls into "review", it never fabricates
records and never discards papers unseen.

### Prompt and model

The preamble is deliberately neutral. It must NOT open with extraction's
`SHARED_RULES` (*"You are extracting structured data from a single patient's case
report"*), which presupposes the paper is a case report and would bias the
classifier toward `CASE_REPORT`. The gate therefore uses its own preamble and
does not share extraction's cached source block. Because it is one call per
paper, it uses no cache breakpoint at all -- a single uncached call.

`complete_json` currently hardcodes `settings.model`, so it gains an optional
`model` parameter (defaulting to `settings.model`) and the gate passes
`settings.relevance_model`. This is the only change to the shared client.

The instruction asks for the single best verbatim quote supporting the label
(for `CASE_REPORT`, the sentence stating the patient's VEXAS/UBA1 diagnosis; for
`MENTION_ONLY`, the sentence showing VEXAS is only referenced), plus a short
`reason`. Returns JSON only. No separate chain-of-thought (consistent with the
extraction finding that CoT gave little advantage); the short `reason` doubles as
lightweight justification and audit text.

Model: **Haiku 4.5** (`RELEVANCE_MODEL` in `config.py`). A four-way label with a
supporting quote is reading comprehension, not structured extraction, and Haiku
handles it at a fraction of Opus cost. Every excluded paper skips ~5 extraction
plus ~30 verification calls, so the gate is strongly net-negative on cost for the
papers it removes and adds only one cheap call for the papers it keeps.

### Wiring, caching, toggle

- `run(limit, verify=True, gate=True)`. `gate=True` by default; `gate=False`
  restores the pre-gate behavior for debugging. Symmetric with `verify`.
- The verdict is cached in `DiskCache("relevance")` keyed by pmc_id and checked
  before the extraction cache lookup -- resume-not-restart, same as every stage.
- Control flow in `run()` per paper: classify; if `CASE_REPORT`, extract as today;
  otherwise record the verdict in the triage report, produce no records, and do
  not call extraction.

### Result surface

`PipelineResult` gains a triage report:

- counts per label (`case_report`, `mention_only`, `review`, `uncertain`);
- a list of `(pmc_id, label, quote, reason)` for the excluded and uncertain
  papers.

`motet run` prints a triage summary: the counts, then the excluded/uncertain
papers with their quotes. Counts only -- no bare percentages (honors the global
constraint). The triage total is the full-text set (papers with sections), not
the raw fetched count: papers without full text are skipped before the gate, as
they are today, and are reported by the existing `full_text_count`.

### Cost model

`estimate_cost` adds one gate call per paper. It keeps extraction priced for all
papers (an honest upper bound, since the keep-rate is unknown before the run) and
notes that the gate can only reduce the real figure. This preserves the existing
"treat the estimate/denominators as an upper bound" stance.

## Testing

The offline suite proves the plumbing; the model's judgment is validated out of
band. This split is explicit because the creator runs no-spend and an offline
test cannot exercise the model.

Offline (automated, injected fake `complete_fn`):

- `build_relevance_prompt` block ordering (neutral preamble present; source-text
  block present; classification instruction last).
- Routing: `CASE_REPORT` -> extraction runs and records are produced;
  `MENTION_ONLY` / `REVIEW` / `UNCERTAIN` -> no records, extraction never called,
  entry recorded in triage.
- Fail-closed: quote-not-in-source, unknown label, and `complete_fn` raising each
  yield `UNCERTAIN`.
- Caching: verdict served from cache on a second run (classify called exactly
  once across two runs).
- Triage counts sum to the number of papers with sections.
- Given a hand-authored `mention_only` verdict for PMC8056719 (fixture), the
  pipeline excludes it and the abstention case holds.

Out of band (manual / live, recorded in the task report, not the CI suite):

- The creator (or a real-key run) authors the actual `classify_relevance` verdicts
  for the three eval papers by hand under the no-spend pattern, confirming
  PMC8056719 -> `MENTION_ONLY`, PMC8313797 -> `CASE_REPORT`, PMC11582098 ->
  `CASE_REPORT`. This validates the model's real judgment; the offline tests above
  validate that the code routes any given verdict correctly.

## Invariants respected

- Verbatim-quote grounding on the gate's own output (same discipline as
  extraction; fail-closed if the quote does not ground).
- `not_stated != negated` unaffected -- the gate operates above the field level.
- No bare percentages in any printed output (triage prints counts).
- Extraction's prompt-cache block ordering is untouched; the gate uses its own
  neutral prefix and does not perturb the extraction cache.
- Denominator honesty: the triage report is the denominator refinement itself
  (fetched -> case reports / uncertain / excluded), all auditable via quotes.

## Files

- New: `src/vexas/relevance.py`, `tests/test_relevance.py`.
- Edit: `src/vexas/pipeline.py` (wire gate into `run()`, add triage to
  `PipelineResult`, relevance cache), `src/vexas/cli.py` (triage summary print;
  `estimate_cost` adds the gate call), `src/vexas/config.py` (`RELEVANCE_MODEL`).
- Edit: `tests/test_pipeline.py`, `tests/test_cli.py` (gate wiring + summary +
  estimate coverage).

## Open questions

None blocking. A future iteration could add a scored gate eval (label accuracy
against a hand-labeled set) and a web view of excluded/uncertain papers.

## Manual validation (2026-07-21)

Acting as the relevance model by hand (no-spend, no Anthropic call made), the
three eval papers' cached PMC source text were read in full and each verdict
was authored and checked for verbatim quote grounding:

- `PMC8056719` -> `MENTION_ONLY`. Quote: "the recently described VEXAS
  (vacuoles, E1 enzyme, X-linked, autoinflammatory, somatic) syndrome". This is
  an Adult-onset Still's disease case; VEXAS appears only once, in a list of
  differential diagnoses to consider for genetic testing, never as this
  patient's diagnosis. Matches expectation.
- `PMC8313797` -> `CASE_REPORT`. Quote: "Genomic sequencing revealed a somatic
  mutation in UBA1 c.121 A>C (p.Met41Leu). He was subsequently diagnosed with
  VEXAS syndrome". Single patient, UBA1 M41L, explicit VEXAS diagnosis.
  Matches expectation.
- `PMC11582098` -> `CASE_REPORT`. Quote: "We described 8 patients with VEXAS
  syndrome with at least one available PET scan". Eight-patient VEXAS series
  (the precedence rule -- review-plus-case is `CASE_REPORT` -- is not actually
  exercised here since the paper is a case series, not a review, but the label
  still confirms as expected). Matches expectation.

All three verdicts match the design's expected judgments; no surprises. Each
quote above was confirmed to be a verbatim substring of the paper's fetched
`full_text` before being recorded.

The live `classify_relevance` HTTP path (the actual Claude API call inside
`classify_relevance`) remains unexercised by this validation -- it is
unexercised until a real-key run, the same status as the extraction smoke
tests. This manual pass validates the labels and quotes a careful reader
assigns to the source text, standing in for that call under the no-spend
constraint; it does not prove the live prompt/parsing path end-to-end.
