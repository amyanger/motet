# Rare Disease Case-Report Miner — Design

Date: 2026-07-18
Demo case: VEXAS syndrome
Status: approved design, pre-implementation

## Problem

Rare disease knowledge accumulates one case report at a time. No single condition
has the volume to justify a dedicated database, so the reports are never
aggregated. A clinician reading one paper never sees the pattern spread across
twenty others.

## The gap this targets

A published systematic review already aggregates VEXAS case reports by hand:
*Clinical features in VEXAS syndrome: a systematic review*, Rheumatology 64(10):5217,
Oct 2025, PMID 40570089 — 720 patients from 33 case reports and 21 case series
across 32 countries. It reports feature prevalences: cutaneous 81.8%,
constitutional 69.4%, respiratory 61.3%, VTE 41.8%, MDS 35.8%.

There is also a 367-patient treatment-response meta-analysis (Ann Hematol 2025,
PMC12141402) and a 59-patient UK multicentre cohort (Lancet Rheumatol 2025;7(7):e472).

Every one of these is **cross-sectional**. They report what fraction of patients
ever had a feature. None tabulate per-case longitudinal structure, because doing
that by hand across 200 papers is labor nobody will spend.

That is the gap. This tool answers *in what order, after which failed treatment,
and how long until diagnosis* — not *how common*.

Consequence for positioning: cross-sectional prevalence output is not a finding,
it is a reproduction of published work. The tool cites the 720-patient review as
its baseline and validates against it.

## Scope

### In scope — the three pattern types

1. **Symptom onset sequencing.** Which manifestation appears first, and the
   interval to hematologic involvement. Existing work establishes that VEXAS
   evolves progressively; per-case ordering is not tabulated anywhere.

2. **Treatment response by line of therapy.** Published meta-analyses pool by
   drug (azacitidine CR 67%, JAK inhibitors CR 42%, IL-6 CR 24%). None report
   response conditional on sequence position. "Ruxolitinib after tocilizumab
   failure" is a different question from "ruxolitinib", and it is the question
   a treating clinician actually faces.

3. **Diagnostic odyssey.** Preceding misdiagnoses (relapsing polychondritis,
   GCA, Sweet syndrome, PMR, MDS alone), count of prior diagnoses, time from
   first symptom to UBA1 testing, and what triggered testing.

### Validation check, not a headline finding

Genotype-phenotype splits. M41V worst prognosis, M41L most indolent, M41T more
ocular disease, ear chondritis protective, higher VAF worse — all published in
the NIH cohort. Recovering these is a credibility result ("the tool reproduces
known findings"), never presented as discovery.

### Out of scope

- Prevalence, incidence, or any rate. Case reports have no denominator.
- Treatment efficacy or comparative effectiveness. Publication bias toward
  responders makes this indefensible.
- Survival or prognosis estimates. Case reports truncate follow-up unpredictably.
- Generalization to a second disease. VEXAS end-to-end first.

## Corpus

PMC open-access full text only: **197 reports** (measured 2026-07-18).

Query: `"VEXAS"[All Fields] AND "open access"[filter] AND "case reports"[pt]`
against the `pmc` database.

An earlier estimate of ~76 was wrong — it searched the phrase `"VEXAS syndrome"`
rather than `"VEXAS"[All Fields]`, which misses papers using only the acronym.
Corrected counts:

| Query | Count |
|---|---|
| PubMed, VEXAS + case-reports[pt] | 209 |
| PubMed, + case report/series in title | 233 |
| PMC, VEXAS + case-reports[pt], any license | 218 |
| **PMC, open access + case-reports[pt]** | **197** |

Full text is therefore available for ~94% of the case-report literature, not 38%.
The corpus is effectively the whole literature, which removes the thin-corpus
risk to treatment-sequencing patterns.

Rationale: onset sequencing and treatment ordering require full text. Abstracts
do not carry the temporal detail. A smaller honest corpus with a stated
limitation is more defensible than a padded one with two denominators to explain.

## Architecture

```
fetch → segment → extract → gate → verify → normalize → aggregate → serve
```

Each stage caches to disk keyed by PMID, so a mid-run failure never re-burns
API spend and development iterations do not re-fetch.

### fetch
PubMed E-utilities esearch/efetch for metadata and abstracts; PMC OA for full
text. Retain section structure — section-aware extraction gave a measurable
efficiency gain in CaseReportBench. Respect NCBI rate limits; use an API key.

### segment
Split each paper into per-patient units with stable `patient_id` before
extraction. Case series describe multiple patients per paper. A prompt that sees
two patients and is asked to sort them out is where silent cross-patient
contamination occurs. This is a structural fix, not a prompting fix.

### extract
Claude, **one call per field category**, not one global call per paper.
Category-specific prompting is the largest measured lever in the benchmark
literature: 28.4% → ~52% token-set-ratio fidelity.

Categories: demographics, timeline, labs, treatments, outcome.

Negative result worth not rediscovering: zero-shot chain-of-thought gave little
advantage over plain zero-shot on this task in CaseReportBench. Do not assume
CoT is free accuracy here — category-specific prompting is the lever, not CoT.

### gate
Every field carries a verbatim `quote`. Code asserts the quote is a literal
substring of source text, whitespace-normalized. Failure discards the field —
never repairs it.

This is the highest-value component in the system. It is deterministic, cheap,
and structurally eliminates the dominant frontier-model failure mode documented
in CaseReportBench, where GPT-4o "merged details and introduced inferences
rather than extracting verbatim information."

Build this first.

### verify
Second Claude pass, blind to pass 1's reasoning, seeing only source text plus
one candidate field and its span: "is this supported?" The generation/verification
asymmetry is the mechanism, so pass 2 must not see pass 1's chain of thought.
Disagreement escalates.

Confidence-weighted self-consistency sampling applies **only** to escalated or
disagreeing fields, not universally — CISC matches plain self-consistency at
10 samples instead of 18.6.

### normalize
- **Units: deterministic code, never the model.** Lab/imaging is the weakest
  extraction category across every model tested. LLM arithmetic on lab values
  is an unforced error. Extract `{value, unit_as_written, analyte, quote}`
  verbatim; convert in code.
- **Ontology: the model never emits an identifier.** Retrieval proposes
  candidates, code validates the ID exists in the ontology, Claude may re-rank
  only. GPT-4 scores 57.9% on HPO ID assignment versus 90.6% for a plain
  embedding/cosine baseline, because it generates plausible IDs with no
  mechanism to check they are real.
- Mutation nomenclature normalized to HGVS (p.Met41Thr / M41T / c.122T>C all
  collapse), verbatim string retained.
- Ontology versions pinned. Reproducibility of the mapping is a defensibility
  requirement.

Resources: HPO and ORDO via OLS4 (https://www.ebi.ac.uk/ols4, free, no key,
verified live). RxNorm via rxnav.nlm.nih.gov (free, no key, verified live).
LOINC requires free registration. For rare disease, a local embedding index over
downloaded HPO/ORDO release files beats live API calls: faster, versionable,
pinnable.

### aggregate
Counts with explicit denominators attached. CARE reporting-completeness and JBI
critical-appraisal metadata per report.

### serve
Local web app. Enter disease, watch extraction stream, click any pattern to
expand its supporting reports, and expand any field to its verbatim source quote.
Traceability is clickable — that is the product.

## Schema

Every clinical finding:

```
{
  "value": ...,
  "unit_as_written": "",
  "quote": "",              // literal substring of source, gate-enforced
  "section": "",
  "certainty": "positive | negated | possible | hypothetical",
  "temporality": "present | past_history",
  "experiencer": "patient | family_member"
}
```

### Three orthogonal assertion axes

Not one presence flag. Certainty, temporality, and experiencer handle negation,
family history, and "we considered lymphoma but ruled it out" in a single schema
move.

This is load-bearing: fine-tuned assertion models beat GPT-4o by **+23.4% on
hypotheticals** and +8.4% on absent findings. "We considered Y" recorded as
"patient had Y" is the failure mode most likely to discredit the tool in front
of a clinician.

Consider routing the assertion decision to a dedicated classifier rather than
folding it into the extraction prompt.

### `not_stated` is never `absent`

Silent omission and explicit denial must never collapse to the same value.
This is a scientific requirement, not schema tidiness: authors under-report
common features, so `not_stated` counts carry no information about true absence.
Merging them manufactures false negatives.

### Treatment response — raw components only

No consensus response criteria exist in VEXAS. Every paper defines "complete
response" ad hoc; this is the stated main limitation of the published
meta-analysis (Lancet Rheumatol 2025, "response is in the eye of the definer").

Therefore extract the components, not the verdict:
steroid dose before/after, CRP before/after, Hb, platelets, symptom resolution
(yes/no), **follow-up duration** (required-with-null — response claims are
meaningless without it).

Apply one consistent response definition across all reports in the aggregation
layer. Store the author's own label separately as an explicitly low-trust field.

This is both the honest approach and a methodological contribution the published
meta-analysis structurally cannot make.

### Per-case publication-bias metadata

Record a `reason_for_publication` field. Publication bias toward severe,
atypical, and treatment-responsive cases is the primary confounder in every
downstream claim, and encoding it per case makes the bias inspectable rather
than invisible.

## Claims discipline

Enforced in the output layer, not left to prose.

**Legitimate:** "Reported in N of M reports meeting inclusion criteria."
Hypothesis generation. Signal detection. Establishing an association has been
described. Mapping the reported phenotypic spectrum. Identifying literature gaps.

**Never emitted:** bare percentages, prevalence, incidence, "X% of patients",
efficacy, comparative effectiveness, causal language, pooled effect sizes.

Every surfaced pattern carries its denominator and a standing caveat that
absence in reports does not imply absence in patients.

Only 43% of systematic reviews mention publication bias at all. Addressing it by
default is a genuine differentiator, not boilerplate.

## Model strategy

Claude, per project brief, plus the substring gate.

Noted and deliberately not acted on: CaseReportBench found Qwen2.5:7B (56.5%)
outperformed GPT-4o (31.1%) at verbatim extraction, because instruction-tuned
models infer rather than quote. The substring gate makes this failure mode moot
by construction — anything inferred rather than quoted is mechanically rejected.
Revisit only if gate rejection rates prove high.

One caveat to carry forward: CaseReportBench's reported 84–93% "hallucination
rate" for GPT-4o is a metric artifact counting any non-verbatim added text. It is
evidence about verbatim fidelity, not fabrication. Do not quote that number to
clinicians without the caveat.

## Testing

- **Gate**: unit tests on substring matching with whitespace/unicode normalization
  edge cases. Deterministic, so fully testable.
- **Segmentation**: fixture case series with known patient counts.
- **Units**: property tests on conversion, both directions.
- **Ontology**: assert every emitted ID resolves in the pinned release.
- **End-to-end**: a small hand-annotated ground-truth set for field-level accuracy.
- **Validation run**: confirm the pipeline recovers the published genotype-survival
  gradient. If it cannot reproduce a known finding, it cannot be trusted on a
  novel one.

## Error handling

Extraction is expected to fail partially and often — case reports are messy.
Every stage degrades to a null field with a recorded reason rather than throwing.
Gate and verify rejections are logged per field and surfaced in the UI as
coverage statistics, because rejection rate is itself a quality signal worth
showing a judge.

## Definition of done

Disease name in → real case reports fetched → structured data extracted with
every field traceable to a verbatim quote → at least one non-obvious longitudinal
pattern surfaced → the exact source reports backing it, clickable.

Demo narrative: "Here is a pattern no single paper states, that the published
720-patient review could not have found because it never extracted sequence, and
here are the N reports that support it — with the sentence from each one."

## Sources

- Rheumatology 2025 systematic review (baseline to beat): https://academic.oup.com/rheumatology/article/64/10/5217/8164134
- CaseReportBench: https://arxiv.org/abs/2505.17265
- Self-verification for clinical IE: https://arxiv.org/abs/2306.00024
- Assertion detection beyond negation: https://arxiv.org/abs/2503.17425
- HPO mapping accuracy: https://arxiv.org/html/2408.01214v2
- LLM ontology ID failures: https://arxiv.org/pdf/2509.04458
- CISC: https://aclanthology.org/2025.findings-acl.1030.pdf
- VEXAS prevalence (JAMA 2023): https://jamanetwork.com/journals/jama/fullarticle/2800661
- VEXAS GeneReviews: https://www.ncbi.nlm.nih.gov/books/NBK614471/
- Treatment meta-analysis: https://pmc.ncbi.nlm.nih.gov/articles/PMC12141402/
- UK cohort: https://orca.cardiff.ac.uk/id/eprint/178875/1/PIIS2665991325000347.pdf
- OLS4: https://www.ebi.ac.uk/ols4
- RxNorm: https://www.nlm.nih.gov/research/umls/rxnorm/index.html
