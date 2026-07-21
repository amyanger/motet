# Rare Disease Case-Report Miner

Mines published case reports for a rare disease and surfaces longitudinal
patterns — symptom onset order, treatment response by line of therapy, and
diagnostic delay — with every claim traceable to a verbatim source quote.

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
git clone <repo> && cd rare-diseases
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env    # then put your key in .env
.venv/bin/vexas check
```

`NCBI_API_KEY` is optional. Without it NCBI allows 3 requests/second;
with it, 10. Free from https://www.ncbi.nlm.nih.gov/account/

## Usage

```bash
vexas check                  # verify setup, spends nothing
vexas estimate --limit 25    # price a run, needs no API key
vexas run --limit 25         # extract 25 reports
vexas serve --limit 25       # extract, then open the web UI
vexas run                    # full corpus
```

**Always estimate first.** `--limit` is the main cost control: a 25-report
run is enough to see the pipeline work end to end.

## Cost

Billed to your own Anthropic account, per token. Measured on the VEXAS
corpus (197 reports, ~7,700 tokens each):

| Run | Approximate cost |
|---|---|
| `--limit 25` | $8 |
| `--limit 50` | $16 |
| full corpus (197) | $62 |

Prompt caching is doing most of the work — without it a full run is ~$292.
Every stage caches to disk, so re-running after a failure resumes rather
than restarts, and a report already extracted costs nothing.

## What it does not claim

Case reports have no denominator: you know how many *reports* describe a
feature, never how many *patients* have it. Authors also preferentially
publish unusual cases. So this tool reports raw counts with explicit
denominators and never emits prevalence, incidence, efficacy, or causal
claims. Absence in reports does not imply absence in patients.

Output is hypothesis-generating. It is not evidence of treatment effect.

## How it stays honest

- **Substring gate.** Every extracted field carries a verbatim quote that
  code verifies is a literal substring of the source. Fields that fail are
  discarded, never repaired.
- **Blind verification.** A second pass judges each field seeing only the
  source text and the claim — never the first pass's reasoning.
- **Three assertion axes.** Certainty, temporality, and experiencer are
  modelled separately, so "we considered lymphoma but ruled it out" is
  never recorded as "patient had lymphoma".
- **No model-generated identifiers.** Ontology terms are retrieved and
  validated against pinned releases; the model may re-rank but never invent
  an ID. Units are converted in code, never by the model.

## Other diseases

The pipeline is disease-agnostic; only the corpus query and a few
VEXAS-specific normalization tables are tuned. Change `pmc_query` in
`src/vexas/config.py` to point at another condition. The misdiagnosis
aliases in `aggregate/odyssey.py` and the variant parser in
`normalize/hgvs.py` are VEXAS-specific and would need replacing.

## License

MIT
