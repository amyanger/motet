# Motet

*Many scattered case reports, read as one score.*

Mines published case reports for a rare disease and surfaces longitudinal
patterns (symptom onset order, treatment response by line of therapy, and
diagnostic delay) with every claim traceable to a verbatim source quote.

Built and demonstrated against VEXAS syndrome (197 open-access case reports).

## Why this exists

Published reviews of rare diseases are cross-sectional: they report what
fraction of patients ever had a feature. None tabulate per-case longitudinal
structure, because doing that by hand across hundreds of papers is labor
nobody spends. This tool answers *in what order, after which failed
treatment, and how long until diagnosis*.

## Setup

You supply your own Anthropic API key and pay for your own runs.

1. Get a key: https://console.anthropic.com/settings/keys
2. Add credits: https://console.anthropic.com/settings/billing
3. Configure and install:

```bash
git clone https://github.com/amyanger/motet.git
cd motet
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env    # then put your key in .env
.venv/bin/motet check
```

`NCBI_API_KEY` is optional. Without it NCBI allows 3 requests/second;
with it, 10. Free from https://www.ncbi.nlm.nih.gov/account/

## Usage

```bash
motet check                  # verify setup, spends nothing
motet estimate --limit 25    # price a run, needs no API key
motet run --limit 25         # extract 25 reports
motet serve --limit 25       # extract, then open the web UI
motet run                    # full corpus
```

**Always estimate first.** `--limit` is the main cost control: a 25-report
run is enough to see the pipeline work end to end.

## Cost

Billed to your own Anthropic account, per token. Measured on the VEXAS
corpus (197 reports, ~7,700 tokens each), with the blind verification pass
enabled:

| Run | Approximate cost |
|---|---|
| `--limit 25` | $10 |
| `--limit 50` | $20 |
| full corpus (197) | $77 |

Prompt caching is doing most of the work: without it a full run is ~$299.
Verification is the largest share of the bill (roughly one call per surviving
field), so it can be turned off in code for a cheaper extraction-only pass.
Every stage caches to disk, so re-running after a failure resumes rather than
restarts, and a report already processed costs nothing.

## What it does not claim

Case reports have no denominator: you know how many *reports* describe a
feature, never how many *patients* have it. Authors also preferentially
publish unusual cases. So this tool reports raw counts with explicit
denominators and never emits prevalence, incidence, efficacy, or causal
claims. Absence in reports does not imply absence in patients.

The corpus query also catches papers that merely mention the disease in a
differential rather than reporting a case, so the raw report count overstates
the true number of cases. Treat the denominators as an upper bound.

Output is hypothesis-generating. It is not evidence of treatment effect.

## How it stays honest

- **Substring gate.** Every extracted field carries a verbatim quote that
  code verifies is a literal substring of the source. Fields that fail are
  discarded, never repaired.
- **Blind verification.** A second pass judges each surviving field seeing
  only the source text and the claim, never the first pass's reasoning. A
  field it cannot support is dropped.
- **Per-patient segmentation.** Case series are split into one unit per
  patient before extraction, so no prompt ever sees two patients at once and
  details cannot bleed between them.
- **Three assertion axes.** Certainty, temporality, and experiencer are
  modelled separately, so "we considered lymphoma but ruled it out" is never
  recorded as "patient had lymphoma".
- **No model-generated identifiers.** Ontology terms are retrieved and
  validated against pinned releases; the model may re-rank but never invent
  an ID. Units are converted in code, never by the model.

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

## How it works

An eight-stage disk-cached pipeline:

1. **fetch** PMC identifiers and full JATS XML from NCBI E-utilities
2. **segment** case series into per-patient units
3. **extract** one Claude call per field category, per patient
4. **gate** every field against its verbatim quote
5. **verify** each surviving field with a blind second pass
6. **normalize** units, ontology terms, and variant nomenclature in code
7. **aggregate** onset order, treatment response by line, diagnostic odyssey
8. **serve** a local web UI with click-through traceability to source quotes

## Other diseases

The pipeline is disease-agnostic; only the corpus query and a few
VEXAS-specific normalization tables are tuned. Change `pmc_query` in
`src/vexas/config.py` to point at another condition. The misdiagnosis
aliases in `aggregate/odyssey.py` and the variant parser in
`normalize/hgvs.py` are VEXAS-specific and would need replacing.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## License

MIT
