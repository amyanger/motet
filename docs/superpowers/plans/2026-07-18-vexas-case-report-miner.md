# VEXAS Case-Report Miner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine 197 PMC open-access VEXAS syndrome case reports into structured, quote-grounded records, then surface longitudinal patterns (symptom onset order, treatment response by line of therapy, diagnostic odyssey) that published cross-sectional reviews could not extract.

**Architecture:** An eight-stage disk-cached pipeline — fetch, segment, extract, gate, verify, normalize, aggregate, serve. Every extracted field carries a verbatim source quote that is mechanically validated as a literal substring of the source text; fields failing that check are discarded, never repaired. Extraction runs one Claude call per field category (not per paper), and a blind second pass verifies each surviving field independently.

**Tech Stack:** Python 3.14, pydantic v2, httpx, pytest, Flask, Anthropic SDK, NCBI E-utilities, EMBL-EBI OLS4 (HPO/ORDO), RxNorm.

## Global Constraints

- Python 3.14.4. No `uv` on this machine — use `python3 -m venv` and `pip`.
- Corpus query, verbatim: `"VEXAS"[All Fields] AND "open access"[filter] AND "case reports"[pt]` against db `pmc`. Expected count 197 (measured 2026-07-18).
- **The model never emits an ontology identifier.** Retrieval proposes, code validates, Claude may only re-rank.
- **The model never converts units.** Extract `value` + `unit_as_written` verbatim; convert in Python.
- **`not_stated` is never `absent`.** Missing field ≠ `certainty=negated`. These must never collapse.
- **No bare percentages anywhere in output.** Every count carries its denominator. No prevalence, incidence, efficacy, comparative-effectiveness, or causal language.
- Every clinical finding carries three assertion axes: `certainty`, `temporality`, `experiencer`.
- Ontology release versions are pinned and recorded in output metadata.
- NCBI rate limit: 3 req/s without an API key, 10 req/s with one. Sleep accordingly.
- `CLAUDE.md` must be in `.gitignore` before the first commit.
- Model id for extraction: `claude-opus-4-8`. Do not hardcode elsewhere; read from `config.py`.
- **Prompt caching is mandatory, not an optimization.** Every prompt builder
  returns content blocks ordered stable-first with a `cache_control`
  breakpoint after the source text. Reordering blocks so the source text
  follows varying content silently disables caching and costs ~$230 per
  corpus run. Tests in `tests/extract/test_prompts.py` and
  `tests/test_verify.py` guard the ordering.

### Cost model (measured 2026-07-18)

Corpus: 197 reports, mean ~7,700 tokens each (range 2,100–17,300).
Each report's text is re-sent ~35 times: 5 category calls + ~30 verify calls.

Pricing, `claude-opus-4-8`: $5 / 1M input, $25 / 1M output.
Cached reads bill at 0.1x, cache writes at 1.25x.

| Configuration | Input tokens | Cost / full run |
|---|---|---|
| No caching (block order wrong) | 53M | ~$292 |
| **Caching (as specified)** | **7M** | **~$62** |
| Caching + Batch API | 7M | ~$31 |

Output is ~$27 in every configuration.

Development runs cost far less than this — `DiskCache` means a report is
extracted once per prompt version, not once per run. Budget roughly two to
three full runs plus incremental work.

Verify caching is live by checking `CACHE_STATS` after the Task 8 smoke test.
If `read` is 0 across multiple calls on one report, the prefix is being
invalidated — check block ordering before running the full corpus.

## File Structure

```
pyproject.toml               deps, pytest config
.gitignore                   includes CLAUDE.md
.env.example                 ANTHROPIC_API_KEY, NCBI_API_KEY
src/vexas/
  config.py                  paths, keys, model id, constants
  gate.py                    substring gate — pure, no deps
  schema.py                  pydantic models, assertion axes
  cache.py                   disk cache keyed by pmid
  fetch/pubmed.py            esearch/efetch metadata
  fetch/pmc.py               full text + section structure
  segment.py                 per-patient units
  extract/prompts.py         one prompt per field category
  extract/client.py          Claude calls, structured output
  extract/runner.py          orchestration, applies gate
  verify.py                  blind second pass
  normalize/units.py         deterministic conversion
  normalize/ontology.py      HPO/RxNorm retrieve + validate
  normalize/hgvs.py          mutation nomenclature
  aggregate/denominators.py  claims-discipline guard
  aggregate/sequence.py      onset ordering
  aggregate/treatment.py     response by line of therapy
  aggregate/odyssey.py       diagnostic odyssey
  web/app.py                 Flask app
  web/templates/             index, pattern detail
tests/                       mirrors src layout
```

Rationale for boundaries: `gate.py` is a pure function with zero dependencies so it can be built and trusted first. `normalize/` is split by data kind because units, ontology terms, and variant nomenclature have nothing in common beyond being deterministic. `aggregate/` is split per pattern type because each is independently demoable and a reviewer could reject one while accepting another.

---

### Task 1: Project scaffolding and the substring gate

The gate is the highest-value component in the system and has no dependencies, so it ships first. Scaffolding is folded in here because the gate's tests need it.

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`
- Create: `src/vexas/__init__.py`, `src/vexas/config.py`, `src/vexas/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `vexas.gate.normalize_for_match(text: str) -> str`
  - `vexas.gate.is_grounded(quote: str, source: str) -> bool`
  - `vexas.config.Settings` with attrs `anthropic_api_key: str | None`, `ncbi_api_key: str | None`, `model: str`, `cache_dir: Path`, `data_dir: Path`
  - `vexas.config.settings` (module-level singleton)

- [ ] **Step 1: Create the project skeleton**

```bash
mkdir -p ~/Dev/rare-diseases/{src/vexas,tests}
cd ~/Dev/rare-diseases
git init
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
CLAUDE.md
data/cache/
data/raw/
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "vexas-miner"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "pydantic>=2.9",
    "httpx>=0.27",
    "anthropic>=0.40",
    "flask>=3.0",
    "python-dotenv>=1.0",
    "defusedxml>=0.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 4: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
NCBI_API_KEY=
```

- [ ] **Step 5: Install dependencies**

```bash
cd ~/Dev/rare-diseases
.venv/bin/pip install -e ".[dev]"
```

Expected: `Successfully installed ... vexas-miner-0.1.0`

- [ ] **Step 6: Write `src/vexas/config.py`**

```python
"""Central configuration. Read once, imported everywhere."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    ncbi_api_key: str | None = field(
        default_factory=lambda: os.getenv("NCBI_API_KEY")
    )
    model: str = "claude-opus-4-8"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    data_dir: Path = PROJECT_ROOT / "data"

    # Corpus definition — see plan Global Constraints.
    pmc_query: str = (
        '"VEXAS"[All Fields] AND "open access"[filter] AND "case reports"[pt]'
    )

    @property
    def ncbi_rate_delay(self) -> float:
        """Seconds to sleep between NCBI calls (3/s without key, 10/s with)."""
        return 0.11 if self.ncbi_api_key else 0.34


settings = Settings()
```

- [ ] **Step 7: Create `src/vexas/__init__.py`**

```python
"""VEXAS syndrome case-report miner."""
```

- [ ] **Step 8: Write the failing test**

Create `tests/test_gate.py`:

```python
from vexas.gate import is_grounded, normalize_for_match


def test_exact_substring_is_grounded():
    source = "The patient developed recurrent fevers and ear chondritis."
    assert is_grounded("recurrent fevers", source) is True


def test_absent_text_is_not_grounded():
    source = "The patient developed recurrent fevers."
    assert is_grounded("ear chondritis", source) is False


def test_whitespace_differences_are_tolerated():
    source = "The patient developed\n  recurrent   fevers\tand chondritis."
    assert is_grounded("recurrent fevers and chondritis", source) is True


def test_case_differences_are_tolerated():
    source = "Macrocytic Anemia was noted."
    assert is_grounded("macrocytic anemia", source) is True


def test_unicode_dashes_are_normalized():
    source = "Treatment was steroid–sparing throughout."
    assert is_grounded("steroid-sparing", source) is True


def test_curly_quotes_are_normalized():
    source = "The authors described a “classic” presentation."
    assert is_grounded('"classic" presentation', source) is True


def test_paraphrase_is_rejected():
    """The core failure mode: model infers instead of quoting."""
    source = "The patient was treated with ruxolitinib 10 mg twice daily."
    assert is_grounded("patient received a JAK inhibitor", source) is False


def test_empty_quote_is_not_grounded():
    assert is_grounded("", "some source text") is False


def test_whitespace_only_quote_is_not_grounded():
    assert is_grounded("   \n  ", "some source text") is False


def test_normalize_collapses_runs_of_whitespace():
    assert normalize_for_match("a  \n\t b") == "a b"
```

- [ ] **Step 9: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_gate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.gate'`

- [ ] **Step 10: Write `src/vexas/gate.py`**

```python
"""The substring gate.

Every extracted field carries a verbatim quote. This module mechanically
verifies that quote is a literal substring of the source text. A field whose
quote fails is DISCARDED, never repaired.

This exists because the dominant frontier-model failure on case-report
extraction is merging details and introducing inferences rather than
extracting verbatim text (CaseReportBench, arXiv:2505.17265). A soft prompt
instruction cannot prevent that; a substring assertion can.

Keep this module dependency-free and pure.
"""

import re
import unicodedata

# Unicode punctuation that publishers substitute for ASCII equivalents.
_PUNCT_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    " ": " ",
}

_WHITESPACE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Canonicalize text for substring comparison.

    Applies NFKC normalization, maps publisher punctuation to ASCII,
    collapses whitespace runs, lowercases, and strips.

    Deliberately permissive about formatting, strict about wording: a
    paraphrase will still fail.
    """
    text = unicodedata.normalize("NFKC", text)
    for source_char, ascii_char in _PUNCT_MAP.items():
        text = text.replace(source_char, ascii_char)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().lower()


def is_grounded(quote: str, source: str) -> bool:
    """True if `quote` appears verbatim in `source` after normalization.

    An empty or whitespace-only quote is never grounded — that is the
    degenerate case where a model returns a field with no evidence.
    """
    normalized_quote = normalize_for_match(quote)
    if not normalized_quote:
        return False
    return normalized_quote in normalize_for_match(source)
```

- [ ] **Step 11: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_gate.py -v
```

Expected: PASS, 10 passed

- [ ] **Step 12: Commit**

```bash
cd ~/Dev/rare-diseases
git add pyproject.toml .gitignore .env.example src/vexas/__init__.py src/vexas/config.py src/vexas/gate.py tests/test_gate.py
git commit -m "Add substring gate for verbatim quote validation

The gate mechanically rejects any extracted field whose quote is not a
literal substring of the source. This is the structural defense against
models inferring rather than quoting."
```

---

### Task 2: Schema with assertion axes

**Files:**
- Create: `src/vexas/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `vexas.gate.is_grounded`
- Produces:
  - Enums `Certainty`, `Temporality`, `Experiencer`
  - `GroundedField` with `.value`, `.unit_as_written`, `.quote`, `.section`, `.certainty`, `.temporality`, `.experiencer`, and method `.is_grounded_in(source: str) -> bool`
  - `PatientRecord` with `.pmid`, `.patient_id`, `.demographics`, `.timeline`, `.labs`, `.treatments`, `.prior_diagnoses`, `.outcome`, `.reason_for_publication`
    (NOT `.author_response_label` — that is per-treatment and lives on
    `TreatmentCourse`. One patient can have several treatments, each with the
    paper's own response verdict, so a single label on the record would be
    ambiguous.)
  - `TimelineEvent`, `TreatmentCourse`, `LabValue`
  - `NOT_STATED` sentinel

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from vexas.schema import (
    Certainty,
    Experiencer,
    GroundedField,
    PatientRecord,
    Temporality,
    TimelineEvent,
)


def _field(**kwargs):
    defaults = dict(value="fever", quote="recurrent fevers", section="Case")
    defaults.update(kwargs)
    return GroundedField(**defaults)


def test_defaults_are_positive_present_patient():
    f = _field()
    assert f.certainty is Certainty.POSITIVE
    assert f.temporality is Temporality.PRESENT
    assert f.experiencer is Experiencer.PATIENT


def test_quote_is_required():
    with pytest.raises(ValidationError):
        GroundedField(value="fever")


def test_negated_finding_is_a_real_field_carrying_evidence():
    """'The paper said no' is a present field with a supporting quote."""
    f = _field(value="ear chondritis",
               quote="no evidence of ear chondritis",
               certainty=Certainty.NEGATED)
    assert f.certainty is Certainty.NEGATED
    assert f.quote == "no evidence of ear chondritis"


def test_unstated_finding_is_absent_not_negated():
    """'The paper didn't say' is a missing field, never certainty=negated.

    Collapsing these two is the failure the schema exists to prevent:
    authors under-report common features, so silence carries no information
    about true absence.
    """
    record = PatientRecord(pmid="12345678", patient_id="12345678-p1")
    assert record.outcome is None
    assert record.prior_diagnoses == []
    assert record.demographics == {}


def test_hypothetical_certainty_is_representable():
    f = _field(certainty=Certainty.HYPOTHETICAL,
               quote="we considered lymphoma")
    assert f.certainty is Certainty.HYPOTHETICAL


def test_family_member_experiencer_is_representable():
    f = _field(experiencer=Experiencer.FAMILY_MEMBER,
               quote="his father had a similar rash")
    assert f.experiencer is Experiencer.FAMILY_MEMBER


def test_is_grounded_in_delegates_to_gate():
    source = "The patient had recurrent fevers for six months."
    assert _field().is_grounded_in(source) is True
    assert _field(quote="ear chondritis").is_grounded_in(source) is False


def test_unit_is_stored_as_written():
    f = _field(value="12.4", unit_as_written="mg/dL", quote="CRP 12.4 mg/dL")
    assert f.unit_as_written == "mg/dL"


def test_timeline_event_allows_unordered_sequence():
    e = TimelineEvent(
        symptom=_field(value="fever", quote="fevers"),
        sequence_index=None,
        interval_value=None,
        interval_unit=None,
    )
    assert e.sequence_index is None


def test_patient_record_requires_pmid_and_patient_id():
    with pytest.raises(ValidationError):
        PatientRecord(pmid="12345678")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_schema.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.schema'`

- [ ] **Step 3: Write `src/vexas/schema.py`**

```python
"""Extraction schema.

Two design decisions here are load-bearing and must not be simplified away:

1. Three orthogonal assertion axes (certainty, temporality, experiencer)
   rather than a single presence flag. Fine-tuned assertion models beat
   general LLMs by +23.4% on hypotheticals and +8.4% on absent findings
   (arXiv:2503.17425). "We considered lymphoma but ruled it out" recorded
   as "patient had lymphoma" is the failure most likely to discredit the
   tool in front of a clinician.

2. `not_stated` is never `absent`. A missing field means the paper did not
   say. `certainty=NEGATED` means the paper said no. Authors under-report
   common features, so missing-ness carries no information about true
   absence. Collapsing these manufactures false negatives.
"""

from enum import Enum

from pydantic import BaseModel, Field

from vexas.gate import is_grounded

#: Explicit marker for "the source did not state this". Distinct from a
#: negated finding. Prefer leaving the field as None; use this only where a
#: string is structurally required.
NOT_STATED = "not_stated"


class Certainty(str, Enum):
    POSITIVE = "positive"
    NEGATED = "negated"
    POSSIBLE = "possible"
    HYPOTHETICAL = "hypothetical"


class Temporality(str, Enum):
    PRESENT = "present"
    PAST_HISTORY = "past_history"


class Experiencer(str, Enum):
    PATIENT = "patient"
    FAMILY_MEMBER = "family_member"


class GroundedField(BaseModel):
    """A single extracted value with its verbatim source evidence."""

    value: str | None = None
    unit_as_written: str | None = None
    quote: str = Field(..., min_length=1)
    section: str | None = None
    certainty: Certainty = Certainty.POSITIVE
    temporality: Temporality = Temporality.PRESENT
    experiencer: Experiencer = Experiencer.PATIENT

    def is_grounded_in(self, source: str) -> bool:
        return is_grounded(self.quote, source)


class TimelineEvent(BaseModel):
    """One symptom with its position in the onset sequence.

    `sequence_index` is None when the paper describes the symptom but does
    not place it relative to others. Many case reports narrate loosely
    ("months of fevers, then a rash"), so unordered must be representable.
    """

    symptom: GroundedField
    sequence_index: int | None = None
    interval_value: float | None = None
    interval_unit: str | None = None


class LabValue(BaseModel):
    analyte: GroundedField
    #: Never converted by the model. See normalize/units.py.
    value_as_written: str | None = None
    unit_as_written: str | None = None


class TreatmentCourse(BaseModel):
    """One treatment, with the raw components of response.

    No consensus response criteria exist in VEXAS — every paper defines
    "complete response" ad hoc. So we record the components and apply one
    consistent definition in the aggregation layer. The author's own label
    is retained separately and treated as low-trust.
    """

    treatment: GroundedField
    line_of_therapy: int | None = None
    dose_as_written: str | None = None
    steroid_dose_before: GroundedField | None = None
    steroid_dose_after: GroundedField | None = None
    crp_before: GroundedField | None = None
    crp_after: GroundedField | None = None
    hemoglobin_after: GroundedField | None = None
    platelets_after: GroundedField | None = None
    symptom_resolution: GroundedField | None = None
    #: Required-with-null. Response claims are meaningless without it.
    follow_up_duration: GroundedField | None = None
    #: Low-trust. The paper's own verdict, kept for comparison only.
    author_response_label: GroundedField | None = None


class PatientRecord(BaseModel):
    """One patient. Case series yield several of these per PMID."""

    pmid: str
    patient_id: str
    demographics: dict[str, GroundedField] = Field(default_factory=dict)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    labs: list[LabValue] = Field(default_factory=list)
    treatments: list[TreatmentCourse] = Field(default_factory=list)
    prior_diagnoses: list[GroundedField] = Field(default_factory=list)
    outcome: GroundedField | None = None
    #: Publication bias is the primary confounder. Encoding it per case
    #: makes the bias inspectable rather than invisible.
    reason_for_publication: GroundedField | None = None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_schema.py -v
```

Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/schema.py tests/test_schema.py
git commit -m "Add extraction schema with three assertion axes

Certainty, temporality and experiencer are modelled orthogonally so that
negation, family history and hypotheticals are all representable. Missing
fields stay distinct from negated findings."
```

---

### Task 3: Disk cache

**Files:**
- Create: `src/vexas/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `vexas.config.settings`
- Produces:
  - `vexas.cache.DiskCache(namespace: str, root: Path | None = None)`
  - `.get(key: str) -> str | None`
  - `.put(key: str, value: str) -> None`
  - `.get_json(key: str) -> object | None`
  - `.put_json(key: str, value: object) -> None`
  - `.has(key: str) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache.py`:

```python
import pytest

from vexas.cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache("test_ns", root=tmp_path)


def test_get_returns_none_when_missing(cache):
    assert cache.get("39012345") is None


def test_put_then_get_roundtrips(cache):
    cache.put("39012345", "<article>body</article>")
    assert cache.get("39012345") == "<article>body</article>"


def test_has_reflects_presence(cache):
    assert cache.has("39012345") is False
    cache.put("39012345", "x")
    assert cache.has("39012345") is True


def test_json_roundtrips(cache):
    payload = {"pmid": "39012345", "patients": 2}
    cache.put_json("39012345", payload)
    assert cache.get_json("39012345") == payload


def test_namespaces_are_isolated(tmp_path):
    a = DiskCache("raw", root=tmp_path)
    b = DiskCache("extracted", root=tmp_path)
    a.put("39012345", "raw text")
    assert b.get("39012345") is None


def test_unsafe_keys_are_sanitized(cache):
    cache.put("../../etc/passwd", "value")
    assert cache.get("../../etc/passwd") == "value"


def test_unicode_survives_roundtrip(cache):
    cache.put("39012345", "chondritis – M41T “classic”")
    assert cache.get("39012345") == "chondritis – M41T “classic”"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_cache.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.cache'`

- [ ] **Step 3: Write `src/vexas/cache.py`**

```python
"""Disk cache keyed by PMID.

Every pipeline stage caches its output so a mid-run failure never re-burns
API spend and development iterations never re-fetch. Namespaces keep stage
outputs from colliding.
"""

import hashlib
import json
import re
from pathlib import Path

from vexas.config import settings

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


class DiskCache:
    def __init__(self, namespace: str, root: Path | None = None) -> None:
        base = root if root is not None else settings.cache_dir
        self.dir = Path(base) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are usually PMIDs, but never trust that — a key containing
        # path separators would otherwise escape the cache directory.
        if not _SAFE_KEY.match(key):
            key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / f"{key}.cache"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def put(self, key: str, value: str) -> None:
        # Write-then-rename so an interrupted write never leaves a
        # half-written file that looks like a valid cache hit.
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)

    def get_json(self, key: str) -> object | None:
        raw = self.get(key)
        return None if raw is None else json.loads(raw)

    def put_json(self, key: str, value: object) -> None:
        self.put(key, json.dumps(value, ensure_ascii=False, indent=2))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_cache.py -v
```

Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/cache.py tests/test_cache.py
git commit -m "Add namespaced disk cache with atomic writes

Each stage caches by PMID so failures never re-burn API spend."
```

---

### Task 4: PubMed metadata fetch

**Files:**
- Create: `src/vexas/fetch/__init__.py`, `src/vexas/fetch/pubmed.py`
- Test: `tests/fetch/test_pubmed.py`

**Interfaces:**
- Consumes: `vexas.config.settings`, `vexas.cache.DiskCache`
- Produces:
  - `vexas.fetch.pubmed.search_pmc_ids(query: str, retmax: int = 500) -> list[str]`
  - `vexas.fetch.pubmed.build_eutils_url(endpoint: str, **params) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/fetch/__init__.py` (empty) and `tests/fetch/test_pubmed.py`:

```python
from urllib.parse import parse_qs, urlparse

from vexas.fetch.pubmed import build_eutils_url, parse_id_list


def test_url_encodes_query_with_spaces_and_brackets():
    url = build_eutils_url("esearch.fcgi", db="pmc",
                           term='"VEXAS"[All Fields]')
    parsed = urlparse(url)
    assert parsed.netloc == "eutils.ncbi.nlm.nih.gov"
    assert " " not in url
    assert parse_qs(parsed.query)["term"] == ['"VEXAS"[All Fields]']


def test_url_includes_db_param():
    url = build_eutils_url("esearch.fcgi", db="pmc", term="x")
    assert parse_qs(urlparse(url).query)["db"] == ["pmc"]


def test_parse_id_list_extracts_ids_from_json():
    payload = {"esearchresult": {"count": "2", "idlist": ["111", "222"]}}
    assert parse_id_list(payload) == ["111", "222"]


def test_parse_id_list_handles_empty_result():
    payload = {"esearchresult": {"count": "0", "idlist": []}}
    assert parse_id_list(payload) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/fetch/test_pubmed.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.fetch'`

- [ ] **Step 3: Write `src/vexas/fetch/__init__.py`**

```python
"""NCBI retrieval."""
```

- [ ] **Step 4: Write `src/vexas/fetch/pubmed.py`**

```python
"""PubMed / PMC identifier search via NCBI E-utilities.

Rate limits: 3 req/s anonymous, 10 req/s with an API key. Exceeding them
gets the IP blocked, so every network call sleeps.
"""

import time
from urllib.parse import urlencode

import httpx

from vexas.config import settings

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def build_eutils_url(endpoint: str, **params) -> str:
    """Build a fully URL-encoded E-utilities URL.

    Encoding matters: PubMed query syntax contains spaces and square
    brackets, which produce an invalid URL if passed through raw.
    """
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return f"{BASE}{endpoint}?{urlencode(params)}"


def parse_id_list(payload: dict) -> list[str]:
    return list(payload.get("esearchresult", {}).get("idlist", []))


def search_pmc_ids(query: str | None = None, retmax: int = 500) -> list[str]:
    """Return PMC IDs matching the corpus query."""
    term = query if query is not None else settings.pmc_query
    url = build_eutils_url(
        "esearch.fcgi", db="pmc", term=term, retmax=retmax, retmode="json"
    )
    response = httpx.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(settings.ncbi_rate_delay)
    return parse_id_list(response.json())
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/fetch/test_pubmed.py -v
```

Expected: PASS, 4 passed

- [ ] **Step 6: Verify against the live API and confirm the corpus size**

```bash
cd ~/Dev/rare-diseases && .venv/bin/python -c "
from vexas.fetch.pubmed import search_pmc_ids
ids = search_pmc_ids()
print('corpus size:', len(ids))
print('first five:', ids[:5])
"
```

Expected: `corpus size: 197` (±a few as PMC grows). If it is dramatically lower, the query is wrong — stop and re-check it against the Global Constraints before continuing.

- [ ] **Step 7: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/fetch/__init__.py src/vexas/fetch/pubmed.py tests/fetch/
git commit -m "Add PMC identifier search via E-utilities

URL encoding is explicit because PubMed query syntax contains spaces and
brackets that break naive URL construction."
```

---

### Task 5: PMC full-text fetch with section structure

**Files:**
- Create: `src/vexas/fetch/pmc.py`
- Test: `tests/fetch/test_pmc.py`, `tests/fixtures/sample_pmc.xml`

**Interfaces:**
- Consumes: `build_eutils_url`, `DiskCache`
- Produces:
  - `vexas.fetch.pmc.parse_sections(xml: str) -> dict[str, str]`
  - `vexas.fetch.pmc.fetch_full_text(pmc_id: str) -> str` (raw XML, cached)
  - `vexas.fetch.pmc.Article` dataclass with `.pmc_id`, `.pmid`, `.title`, `.sections`, `.full_text`

Section structure is retained because section-aware extraction gave a measured efficiency gain in CaseReportBench, and because a quote's section is useful provenance in the UI.

- [ ] **Step 1: Create the test fixture**

Create `tests/fixtures/sample_pmc.xml`:

```xml
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmc">PMC9999999</article-id>
      <article-id pub-id-type="pmid">39012345</article-id>
      <title-group>
        <article-title>VEXAS syndrome presenting with chondritis</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>VEXAS syndrome is caused by somatic <italic>UBA1</italic> mutations.</p>
    </sec>
    <sec>
      <title>Case presentation</title>
      <p>A 68-year-old man presented with recurrent fevers for six months,
      followed by ear chondritis.</p>
      <p>He was treated with prednisolone 40 mg daily, then ruxolitinib.</p>
    </sec>
  </body>
</article>
```

- [ ] **Step 2: Write the failing test**

Create `tests/fetch/test_pmc.py`:

```python
from pathlib import Path

import pytest

from vexas.fetch.pmc import parse_article, parse_sections

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_pmc.xml"


@pytest.fixture
def xml():
    return FIXTURE.read_text(encoding="utf-8")


def test_parse_sections_keys_by_section_title(xml):
    sections = parse_sections(xml)
    assert "Introduction" in sections
    assert "Case presentation" in sections


def test_section_text_joins_paragraphs(xml):
    sections = parse_sections(xml)
    case = sections["Case presentation"]
    assert "recurrent fevers for six months" in case
    assert "ruxolitinib" in case


def test_inline_markup_is_flattened(xml):
    """UBA1 is wrapped in <italic> and must not break the text."""
    sections = parse_sections(xml)
    assert "somatic UBA1 mutations" in sections["Introduction"]


def test_parse_article_extracts_identifiers(xml):
    article = parse_article(xml)
    assert article.pmc_id == "PMC9999999"
    assert article.pmid == "39012345"
    assert article.title == "VEXAS syndrome presenting with chondritis"


def test_full_text_concatenates_all_sections(xml):
    article = parse_article(xml)
    assert "recurrent fevers" in article.full_text
    assert "somatic UBA1 mutations" in article.full_text


def test_quotes_from_sections_survive_the_gate(xml):
    """Guards the seam between fetch and gate: text extracted here must
    still match quotes the model will produce from it."""
    from vexas.gate import is_grounded

    article = parse_article(xml)
    assert is_grounded("recurrent fevers for six months", article.full_text)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/fetch/test_pmc.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.fetch.pmc'`

- [ ] **Step 4: Write `src/vexas/fetch/pmc.py`**

```python
"""PMC full-text retrieval and section parsing.

JATS XML wraps inline markup (<italic>, <sup>, <xref>) inside paragraph
text. Naive `.text` extraction silently truncates at the first child tag,
which would produce quotes the gate then rejects. `itertext()` is used
throughout to flatten inline markup correctly.

Parsing uses defusedxml rather than the stdlib parser. JATS documents carry
DTD references, and stdlib ElementTree is vulnerable to entity-expansion
attacks (billion laughs, quadratic blowup). This XML comes from NCBI over
HTTPS so the risk is low, but the swap is one import and removes the
question entirely.
"""

import time
from dataclasses import dataclass, field
# defusedxml exposes the safe parser functions but not the Element type,
# which it returns unchanged from the stdlib. Import parsing from
# defusedxml, the type from stdlib. Never call stdlib fromstring/parse.
from xml.etree.ElementTree import Element

import httpx
from defusedxml import ElementTree as ET

from vexas.cache import DiskCache
from vexas.config import settings
from vexas.fetch.pubmed import build_eutils_url

_cache = DiskCache("pmc_xml")


@dataclass
class Article:
    pmc_id: str
    pmid: str | None
    title: str
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        parts = [f"{name}\n{body}" for name, body in self.sections.items()]
        return "\n\n".join(parts)


def _flatten(element: Element) -> str:
    """Collect all descendant text, flattening inline markup."""
    return " ".join(t.strip() for t in element.itertext() if t.strip())


def parse_sections(xml: str) -> dict[str, str]:
    """Map section title to flattened section text.

    Untitled sections are keyed 'Body'. Repeated titles are suffixed so
    later sections never silently overwrite earlier ones.
    """
    root = ET.fromstring(xml)
    sections: dict[str, str] = {}
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        title = _flatten(title_el) if title_el is not None else "Body"
        paragraphs = [_flatten(p) for p in sec.findall("p")]
        text = "\n".join(p for p in paragraphs if p)
        if not text:
            continue
        key = title
        suffix = 2
        while key in sections:
            key = f"{title} ({suffix})"
            suffix += 1
        sections[key] = text
    return sections


def _article_id(root: Element, id_type: str) -> str | None:
    for el in root.iter("article-id"):
        if el.get("pub-id-type") == id_type:
            return (el.text or "").strip()
    return None


def parse_article(xml: str) -> Article:
    root = ET.fromstring(xml)
    pmc_id = _article_id(root, "pmc") or ""
    if pmc_id and not pmc_id.startswith("PMC"):
        pmc_id = f"PMC{pmc_id}"
    title_el = root.find(".//article-title")
    return Article(
        pmc_id=pmc_id,
        pmid=_article_id(root, "pmid"),
        title=_flatten(title_el) if title_el is not None else "",
        sections=parse_sections(xml),
    )


def fetch_full_text(pmc_id: str) -> str:
    """Fetch raw JATS XML for one PMC article. Cached by PMC id."""
    cached = _cache.get(pmc_id)
    if cached is not None:
        return cached
    numeric = pmc_id.removeprefix("PMC")
    url = build_eutils_url("efetch.fcgi", db="pmc", id=numeric, retmode="xml")
    response = httpx.get(url, timeout=120)
    response.raise_for_status()
    time.sleep(settings.ncbi_rate_delay)
    _cache.put(pmc_id, response.text)
    return response.text
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/fetch/test_pmc.py -v
```

Expected: PASS, 6 passed

- [ ] **Step 6: Download the real corpus**

```bash
cd ~/Dev/rare-diseases && .venv/bin/python -c "
from vexas.fetch.pubmed import search_pmc_ids
from vexas.fetch.pmc import fetch_full_text, parse_article
ids = search_pmc_ids()
ok = failed = 0
for i, pid in enumerate(ids, 1):
    try:
        article = parse_article(fetch_full_text('PMC' + pid))
        ok += 1 if article.sections else 0
    except Exception as exc:
        failed += 1
        print('FAIL', pid, type(exc).__name__)
    if i % 25 == 0:
        print(f'{i}/{len(ids)} fetched')
print(f'parsed with sections: {ok}, failed: {failed}, total: {len(ids)}')
"
```

Expected: most articles parse with sections. Record the number that yield no sections — some PMC records are abstract-only despite the filter, and that number is the real full-text denominator. Write it down; the aggregation layer reports it.

- [ ] **Step 7: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/fetch/pmc.py tests/fetch/test_pmc.py tests/fixtures/
git commit -m "Add PMC full-text fetch with JATS section parsing

Uses itertext() throughout so inline markup does not truncate paragraph
text and produce quotes the gate would later reject."
```

---

### Task 6: Per-patient segmentation

Case series describe several patients in one paper. Letting a single prompt see two patients and sort them out is where silent cross-patient contamination happens, so this is a structural split done before extraction.

**Files:**
- Create: `src/vexas/segment.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: `vexas.fetch.pmc.Article`, `vexas.extract.client.complete_json` (Task 7)
- Produces:
  - `vexas.segment.PatientUnit` dataclass with `.patient_id`, `.pmid`, `.text`
  - `vexas.segment.split_patients(article: Article, unit_texts: list[str]) -> list[PatientUnit]`
  - `vexas.segment.detect_patient_count(article: Article) -> int`

Note on ordering: `split_patients` is pure and testable now. `detect_patient_count` calls Claude and is exercised in Task 8. Build and test the pure part here.

- [ ] **Step 1: Write the failing test**

Create `tests/test_segment.py`:

```python
from vexas.fetch.pmc import Article
from vexas.segment import PatientUnit, split_patients


def _article():
    return Article(
        pmc_id="PMC9999999",
        pmid="39012345",
        title="Two cases of VEXAS",
        sections={"Case presentation": "Patient 1 ... Patient 2 ..."},
    )


def test_single_unit_gets_suffix_p1():
    units = split_patients(_article(), ["A 68-year-old man ..."])
    assert len(units) == 1
    assert units[0].patient_id == "39012345-p1"


def test_multiple_units_get_sequential_ids():
    units = split_patients(_article(), ["first case text", "second case text"])
    assert [u.patient_id for u in units] == ["39012345-p1", "39012345-p2"]


def test_unit_text_is_preserved_verbatim():
    """Segmentation must not alter text, or the gate breaks downstream."""
    text = "A 68-year-old man  with\nrecurrent fevers."
    units = split_patients(_article(), [text])
    assert units[0].text == text


def test_empty_units_are_dropped():
    units = split_patients(_article(), ["real text", "   ", ""])
    assert len(units) == 1


def test_pmid_falls_back_to_pmc_id_when_absent():
    article = Article(pmc_id="PMC9999999", pmid=None, title="t",
                      sections={"Body": "x"})
    units = split_patients(article, ["text"])
    assert units[0].patient_id == "PMC9999999-p1"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_segment.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.segment'`

- [ ] **Step 3: Write `src/vexas/segment.py`**

```python
"""Split multi-patient papers into per-patient units.

Extraction runs per unit, never per paper. A prompt that sees two patients
and is asked to keep them straight will occasionally merge them, and the
resulting record looks entirely plausible — there is no downstream check
that catches it. Splitting first makes the failure impossible rather than
unlikely.
"""

from dataclasses import dataclass

from vexas.fetch.pmc import Article


@dataclass
class PatientUnit:
    patient_id: str
    pmid: str
    text: str


def split_patients(article: Article, unit_texts: list[str]) -> list[PatientUnit]:
    """Wrap per-patient text blocks in units with stable ids.

    Text is preserved byte-for-byte. Any normalization here would desync
    quotes from source and cause spurious gate rejections.
    """
    identifier = article.pmid or article.pmc_id
    units: list[PatientUnit] = []
    for text in unit_texts:
        if not text.strip():
            continue
        units.append(
            PatientUnit(
                patient_id=f"{identifier}-p{len(units) + 1}",
                pmid=identifier,
                text=text,
            )
        )
    return units
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_segment.py -v
```

Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/segment.py tests/test_segment.py
git commit -m "Add per-patient segmentation

Case series are split into units before extraction so no prompt ever sees
two patients at once."
```

---

### Task 7: Claude client and per-category prompts

Category-specific prompting is the largest measured lever in the benchmark literature (28.4% → ~52% extraction fidelity). Chain-of-thought is deliberately not used: it showed little advantage on this task and costs tokens.

**Files:**
- Create: `src/vexas/extract/__init__.py`, `src/vexas/extract/prompts.py`, `src/vexas/extract/client.py`
- Test: `tests/extract/test_prompts.py`, `tests/extract/test_client.py`

**Interfaces:**
- Consumes: `vexas.config.settings`
- Produces:
  - `vexas.extract.prompts.CATEGORIES: dict[str, CategorySpec]` with keys `demographics`, `timeline`, `labs`, `treatments`, `outcome`
  - `CategorySpec` with `.name`, `.instruction`, `.json_shape`
  - `vexas.extract.prompts.build_prompt(category: str, text: str) -> str`
  - `vexas.extract.client.complete_json(prompt: str, max_tokens: int = 4096) -> dict`
  - `vexas.extract.client.extract_json_block(raw: str) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/extract/__init__.py` (empty) and `tests/extract/test_prompts.py`:

```python
import pytest

from vexas.extract.prompts import CATEGORIES, build_prompt


def _flat(category: str, text: str) -> str:
    """Concatenate all block text for content assertions."""
    return "\n".join(b["text"] for b in build_prompt(category, text))


def test_all_five_categories_exist():
    assert set(CATEGORIES) == {
        "demographics", "timeline", "labs", "treatments", "outcome"
    }


def test_prompt_embeds_the_source_text():
    assert "A 68-year-old man with fevers." in _flat(
        "timeline", "A 68-year-old man with fevers."
    )


def test_prompt_demands_verbatim_quotes():
    prompt = _flat("labs", "CRP 12 mg/dL").lower()
    assert "verbatim" in prompt
    assert "quote" in prompt


def test_prompt_forbids_unit_conversion():
    assert "do not convert" in _flat("labs", "CRP 12 mg/dL").lower()


def test_prompt_explains_not_stated_versus_negated():
    prompt = _flat("timeline", "text").lower()
    assert "not_stated" in prompt
    assert "negated" in prompt


def test_prompt_requires_assertion_axes():
    prompt = _flat("timeline", "text")
    for axis in ("certainty", "temporality", "experiencer"):
        assert axis in prompt


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        build_prompt("nonexistent", "text")


# --- Caching structure. These guard ~$230 per corpus run. ---

def test_source_text_precedes_the_category_instruction():
    """Caching is a prefix match: stable content must come first."""
    blocks = build_prompt("timeline", "SOURCE_MARKER")
    source_index = next(
        i for i, b in enumerate(blocks) if "SOURCE_MARKER" in b["text"]
    )
    task_index = next(
        i for i, b in enumerate(blocks) if b["text"].startswith("TASK:")
    )
    assert source_index < task_index


def test_cache_breakpoint_sits_on_the_source_block():
    blocks = build_prompt("timeline", "SOURCE_MARKER")
    cached = [b for b in blocks if "cache_control" in b]
    assert len(cached) == 1
    assert "SOURCE_MARKER" in cached[0]["text"]


def test_cached_prefix_is_identical_across_categories():
    """The whole saving depends on byte-identical prefixes."""
    a = build_prompt("timeline", "same text")
    b = build_prompt("labs", "same text")
    assert a[:2] == b[:2]
    assert a[2] != b[2]
```

Create `tests/extract/test_client.py`:

```python
import pytest

from vexas.extract.client import extract_json_block


def test_parses_bare_json():
    assert extract_json_block('{"a": 1}') == {"a": 1}


def test_parses_json_inside_fenced_block():
    raw = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
    assert extract_json_block(raw) == {"a": 1}


def test_parses_json_with_leading_prose():
    raw = 'I found the following:\n{"a": 1}'
    assert extract_json_block(raw) == {"a": 1}


def test_raises_on_unparseable_output():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_block("I could not find anything.")


def test_handles_nested_braces():
    raw = '{"outer": {"inner": [1, 2]}}'
    assert extract_json_block(raw) == {"outer": {"inner": [1, 2]}}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/extract/ -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.extract'`

- [ ] **Step 3: Write `src/vexas/extract/__init__.py`**

```python
"""Claude-based extraction."""
```

- [ ] **Step 4: Write `src/vexas/extract/prompts.py`**

```python
"""One prompt per field category.

Category-specific prompting rather than one global prompt is the largest
measured accuracy lever for this task (28.4% -> ~52% token set ratio,
CaseReportBench arXiv:2505.17265). Chain-of-thought is deliberately absent:
the same benchmark found it gave little advantage here.
"""

from dataclasses import dataclass

SHARED_RULES = """\
You are extracting structured data from a single patient's case report.

Absolute rules:
1. Every field you emit MUST include a "quote" that is copied VERBATIM,
   character for character, from the source text below. Do not paraphrase,
   summarize, correct, or normalize the quote. If you cannot find verbatim
   supporting text, omit the field entirely.
2. Do NOT convert units, compute values, or infer anything not stated.
   Copy numbers and units exactly as written.
3. Do NOT emit ontology identifiers (HPO, RxNorm, LOINC, OMIM codes).
   Emit only the surface phrase as written.
4. Distinguish these three cases and never conflate them:
   - The report states the finding is present -> certainty "positive"
   - The report states the finding is ABSENT -> certainty "negated"
   - The report does not mention it -> OMIT the field (not_stated)
   Omission and negation are different facts. Never emit a negated field
   for something simply not mentioned.
5. Every finding carries three assertion axes:
   - certainty: positive | negated | possible | hypothetical
     Use "hypothetical" for conditions merely considered or ruled out
     ("we considered lymphoma", "differential included GCA").
   - temporality: present | past_history
   - experiencer: patient | family_member
     Use "family_member" for family history. Never attribute a relative's
     finding to the patient.

Return ONLY a JSON object. No commentary.
"""


@dataclass(frozen=True)
class CategorySpec:
    name: str
    instruction: str
    json_shape: str


CATEGORIES: dict[str, CategorySpec] = {
    "demographics": CategorySpec(
        name="demographics",
        instruction=(
            "Extract patient age, sex, and country of the reporting centre. "
            "Also extract the UBA1 variant exactly as written (e.g. 'p.Met41Thr', "
            "'M41T', 'c.122T>C') and the variant allele frequency if stated."
        ),
        json_shape="""{
  "fields": {
    "age": {"value": "68", "unit_as_written": "years", "quote": "...",
            "certainty": "positive", "temporality": "present",
            "experiencer": "patient"},
    "sex": {"value": "male", "quote": "...", "certainty": "positive",
            "temporality": "present", "experiencer": "patient"},
    "country": {"value": "Japan", "quote": "...", "certainty": "positive",
                "temporality": "present", "experiencer": "patient"},
    "uba1_variant": {"value": "p.Met41Thr", "quote": "...",
                     "certainty": "positive", "temporality": "present",
                     "experiencer": "patient"},
    "vaf": {"value": "62", "unit_as_written": "%", "quote": "...",
            "certainty": "positive", "temporality": "present",
            "experiencer": "patient"}
  }
}""",
    ),
    "timeline": CategorySpec(
        name="timeline",
        instruction=(
            "Extract each symptom or manifestation AND its position in the "
            "onset sequence. This is the most important category.\n"
            "- sequence_index: 1 for the first manifestation, 2 for the next, "
            "and so on. Use null when the report does not place the symptom "
            "relative to others. Do NOT guess an order that is not stated.\n"
            "- interval_value / interval_unit: time since the PREVIOUS event, "
            "as written ('six months' -> 6, 'months'). Null if not stated.\n"
            "Also extract any diagnoses the patient carried BEFORE VEXAS was "
            "confirmed (e.g. relapsing polychondritis, giant cell arteritis, "
            "Sweet syndrome, polymyalgia rheumatica, MDS) as prior_diagnoses, "
            "and the time from first symptom to UBA1 testing if stated."
        ),
        json_shape="""{
  "timeline": [
    {"symptom": {"value": "recurrent fevers", "quote": "...",
                 "certainty": "positive", "temporality": "present",
                 "experiencer": "patient"},
     "sequence_index": 1, "interval_value": null, "interval_unit": null}
  ],
  "prior_diagnoses": [
    {"value": "relapsing polychondritis", "quote": "...",
     "certainty": "positive", "temporality": "past_history",
     "experiencer": "patient"}
  ],
  "time_to_diagnosis": {"value": "3", "unit_as_written": "years",
                        "quote": "...", "certainty": "positive",
                        "temporality": "present", "experiencer": "patient"}
}""",
    ),
    "labs": CategorySpec(
        name="labs",
        instruction=(
            "Extract laboratory values. Copy the number and the unit exactly "
            "as printed. DO NOT convert between units (mg/L and mg/dL, g/dL "
            "and g/L). DO NOT compute derived values. If a value is given as "
            "a range or with a qualifier, copy it as written."
        ),
        json_shape="""{
  "labs": [
    {"analyte": {"value": "CRP", "quote": "...", "certainty": "positive",
                 "temporality": "present", "experiencer": "patient"},
     "value_as_written": "12.4", "unit_as_written": "mg/dL"}
  ]
}""",
    ),
    "treatments": CategorySpec(
        name="treatments",
        instruction=(
            "Extract every treatment given, IN THE ORDER GIVEN.\n"
            "- line_of_therapy: 1 for the first treatment tried, 2 for the "
            "next, and so on. This ordering is the point of the extraction. "
            "Use null only if the report genuinely does not indicate order.\n"
            "- Record the RAW COMPONENTS of response, not a verdict: steroid "
            "dose before and after, CRP before and after, haemoglobin and "
            "platelets after, whether symptoms resolved, and follow-up "
            "duration.\n"
            "- author_response_label: if the authors themselves call it a "
            "'complete response' or 'partial response', record that phrase. "
            "Do NOT invent a label they did not use, and do not infer one "
            "from the components."
        ),
        json_shape="""{
  "treatments": [
    {"treatment": {"value": "ruxolitinib", "quote": "...",
                   "certainty": "positive", "temporality": "present",
                   "experiencer": "patient"},
     "line_of_therapy": 2,
     "dose_as_written": "10 mg twice daily",
     "steroid_dose_before": {"value": "40", "unit_as_written": "mg/day",
                             "quote": "...", "certainty": "positive",
                             "temporality": "present", "experiencer": "patient"},
     "steroid_dose_after": null,
     "crp_before": null, "crp_after": null,
     "hemoglobin_after": null, "platelets_after": null,
     "symptom_resolution": {"value": "yes", "quote": "...",
                            "certainty": "positive", "temporality": "present",
                            "experiencer": "patient"},
     "follow_up_duration": {"value": "12", "unit_as_written": "months",
                            "quote": "...", "certainty": "positive",
                            "temporality": "present", "experiencer": "patient"},
     "author_response_label": null}
  ]
}""",
    ),
    "outcome": CategorySpec(
        name="outcome",
        instruction=(
            "Extract the patient's final reported outcome and the reason the "
            "case was published. For reason_for_publication, quote the "
            "authors' own statement of novelty if present (e.g. 'the first "
            "reported case of', 'we describe an unusual'). This records "
            "publication bias per case; omit the field if the authors make "
            "no such claim rather than inferring one."
        ),
        json_shape="""{
  "outcome": {"value": "alive at last follow-up", "quote": "...",
              "certainty": "positive", "temporality": "present",
              "experiencer": "patient"},
  "reason_for_publication": {"value": "first reported case with cardiac involvement",
                             "quote": "...", "certainty": "positive",
                             "temporality": "present", "experiencer": "patient"}
}""",
    ),
}


def build_prompt(category: str, text: str) -> list[dict]:
    """Build a cache-friendly content block list.

    ORDER IS LOAD-BEARING. Prompt caching is a prefix match, so the stable
    content must come first and the varying content last:

        SHARED_RULES  (identical for every call, ever)
        SOURCE TEXT   (identical for all ~35 calls about one report)  <-- cached
        INSTRUCTION   (varies per category)

    Each report's text is re-sent ~35 times (5 category calls + ~30
    verification calls). Cached reads cost 0.1x, so getting this ordering
    right takes a full corpus run from roughly $292 to roughly $62.

    Putting SOURCE TEXT last would make it uncacheable and silently cost
    ~$230 per run. Do not reorder these blocks.
    """
    spec = CATEGORIES[category]
    return [
        {"type": "text", "text": SHARED_RULES},
        {
            "type": "text",
            "text": f"SOURCE TEXT:\n---\n{text}\n---",
            # Breakpoint: everything above this is reused across all calls
            # about this report. Reports average ~7,700 tokens, comfortably
            # over Opus 4.8's 4,096-token minimum cacheable prefix.
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"TASK: {spec.instruction}\n\n"
                f"Return JSON in exactly this shape:\n{spec.json_shape}\n\n"
                f"Omit any field you cannot support with a verbatim quote."
            ),
        },
    ]
```

- [ ] **Step 5: Write `src/vexas/extract/client.py`**

```python
"""Anthropic API wrapper.

Kept deliberately thin: one call, one JSON object out. Retries handle
transient API errors only — a malformed response is a data-quality signal
and is surfaced, not silently retried into existence.
"""

import json
import time

from anthropic import Anthropic, APIError

from vexas.config import settings

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                "and fill it in."
            )
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


def extract_json_block(raw: str) -> dict:
    """Pull the first complete JSON object out of a model response.

    Handles bare JSON, fenced blocks, and leading prose by scanning for a
    balanced brace span.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : index + 1])

    raise ValueError(f"no JSON object in response: {raw[:200]!r}")


def complete_json(prompt: str | list[dict], max_tokens: int = 4096,
                  max_retries: int = 3) -> dict:
    """Send one prompt, return the parsed JSON object.

    `prompt` is either a plain string or a list of content blocks. The block
    form carries the `cache_control` breakpoint that makes repeated calls
    about the same report cost 0.1x — see extract/prompts.build_prompt.
    """
    client = _get_client()
    content = prompt if isinstance(prompt, list) else [
        {"type": "text", "text": prompt}
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=settings.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            _log_cache_usage(message)
            return extract_json_block(message.content[0].text)
        except APIError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error


#: Running totals, so a silently broken cache is visible rather than
#: just expensive. If cache_read stays at zero across a corpus run, the
#: prefix is being invalidated somewhere — check block ordering first.
CACHE_STATS = {"write": 0, "read": 0, "uncached": 0}


def _log_cache_usage(message) -> None:
    usage = message.usage
    CACHE_STATS["write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    CACHE_STATS["read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    CACHE_STATS["uncached"] += usage.input_tokens
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/extract/ -v
```

Expected: PASS, 12 passed

- [ ] **Step 7: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/extract/ tests/extract/
git commit -m "Add per-category extraction prompts and Claude client

One prompt per field category rather than one global prompt, which is the
largest measured accuracy lever for this task. No chain-of-thought."
```

---

### Task 8: Extraction runner with the gate applied

**Files:**
- Create: `src/vexas/extract/runner.py`
- Test: `tests/extract/test_runner.py`

**Interfaces:**
- Consumes: `build_prompt`, `complete_json`, `is_grounded`, `PatientUnit`, schema models
- Produces:
  - `vexas.extract.runner.GateStats` dataclass with `.kept`, `.rejected`, `.rejected_quotes`
  - `vexas.extract.runner.apply_gate(payload: dict, source: str) -> tuple[dict, GateStats]`
  - `vexas.extract.runner.extract_unit(unit: PatientUnit, complete=complete_json) -> tuple[PatientRecord, GateStats]`

`complete` is injected so tests never hit the network.

- [ ] **Step 1: Write the failing test**

Create `tests/extract/test_runner.py`:

```python
from vexas.extract.runner import GateStats, apply_gate, extract_unit
from vexas.segment import PatientUnit

SOURCE = (
    "A 68-year-old man presented with recurrent fevers for six months, "
    "followed by ear chondritis. He was treated with ruxolitinib."
)


def test_grounded_field_survives_the_gate():
    payload = {"fields": {"age": {"value": "68", "quote": "68-year-old man"}}}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert "age" in cleaned["fields"]
    assert stats.kept == 1
    assert stats.rejected == 0


def test_ungrounded_field_is_removed():
    payload = {"fields": {"age": {"value": "70", "quote": "70-year-old woman"}}}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert cleaned["fields"] == {}
    assert stats.rejected == 1


def test_paraphrased_quote_is_removed():
    """The exact failure the gate exists to stop."""
    payload = {"treatments": [
        {"treatment": {"value": "ruxolitinib",
                       "quote": "treated with a JAK inhibitor"}}
    ]}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert cleaned["treatments"] == []
    assert stats.rejected == 1


def test_rejected_quotes_are_recorded_for_inspection():
    payload = {"fields": {"age": {"value": "70", "quote": "invented text"}}}
    _, stats = apply_gate(payload, SOURCE)
    assert "invented text" in stats.rejected_quotes


def test_gate_recurses_into_nested_lists():
    payload = {"timeline": [
        {"symptom": {"value": "fevers", "quote": "recurrent fevers"},
         "sequence_index": 1},
        {"symptom": {"value": "rash", "quote": "widespread rash"},
         "sequence_index": 2},
    ]}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert len(cleaned["timeline"]) == 1
    assert cleaned["timeline"][0]["symptom"]["value"] == "fevers"


def test_field_missing_a_quote_is_rejected():
    payload = {"fields": {"age": {"value": "68"}}}
    cleaned, stats = apply_gate(payload, SOURCE)
    assert cleaned["fields"] == {}
    assert stats.rejected == 1


def test_non_field_scalars_pass_through_untouched():
    payload = {"timeline": [
        {"symptom": {"value": "fevers", "quote": "recurrent fevers"},
         "sequence_index": 1, "interval_value": 6, "interval_unit": "months"}
    ]}
    cleaned, _ = apply_gate(payload, SOURCE)
    assert cleaned["timeline"][0]["interval_value"] == 6


def test_extract_unit_builds_a_record_from_stubbed_responses():
    unit = PatientUnit(patient_id="39012345-p1", pmid="39012345", text=SOURCE)

    def fake_complete(prompt, max_tokens=4096):
        # prompt is now a list of content blocks, not a string
        flat = "\n".join(b["text"] for b in prompt)
        if "age, sex" in flat:
            return {"fields": {"age": {"value": "68",
                                       "quote": "68-year-old man"}}}
        if "onset sequence" in flat:
            return {"timeline": [{"symptom": {"value": "recurrent fevers",
                                              "quote": "recurrent fevers"},
                                  "sequence_index": 1}]}
        return {}

    record, stats = extract_unit(unit, complete=fake_complete)
    assert record.pmid == "39012345"
    assert record.patient_id == "39012345-p1"
    assert record.demographics["age"].value == "68"
    assert record.timeline[0].sequence_index == 1
    assert stats.rejected == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/extract/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.extract.runner'`

- [ ] **Step 3: Write `src/vexas/extract/runner.py`**

```python
"""Extraction orchestration: prompt per category, then gate everything.

The gate runs on raw JSON before pydantic parsing, so an ungrounded field
never becomes a typed object at all. Rejections are counted rather than
repaired — rejection rate is a quality signal we surface in the UI.
"""

from dataclasses import dataclass, field

from vexas.extract.client import complete_json
from vexas.extract.prompts import CATEGORIES, build_prompt
from vexas.gate import is_grounded
from vexas.schema import (
    GroundedField,
    LabValue,
    PatientRecord,
    TimelineEvent,
    TreatmentCourse,
)
from vexas.segment import PatientUnit

#: Keys that identify a dict as an extracted field rather than a container.
_FIELD_MARKER = "quote"


@dataclass
class GateStats:
    kept: int = 0
    rejected: int = 0
    rejected_quotes: list[str] = field(default_factory=list)

    def merge(self, other: "GateStats") -> None:
        self.kept += other.kept
        self.rejected += other.rejected
        self.rejected_quotes.extend(other.rejected_quotes)

    @property
    def rejection_rate(self) -> float:
        total = self.kept + self.rejected
        return 0.0 if total == 0 else self.rejected / total


def _looks_like_field(value: object) -> bool:
    return isinstance(value, dict) and (
        _FIELD_MARKER in value or "value" in value
    )


def apply_gate(payload: object, source: str, stats: GateStats | None = None):
    """Recursively strip every field whose quote is not verbatim in source.

    Returns (cleaned_payload, stats). Containers are preserved; only field
    dicts are judged.
    """
    stats = stats if stats is not None else GateStats()

    if isinstance(payload, dict):
        if _looks_like_field(payload):
            quote = payload.get(_FIELD_MARKER)
            if isinstance(quote, str) and is_grounded(quote, source):
                stats.kept += 1
                return payload, stats
            stats.rejected += 1
            stats.rejected_quotes.append(quote if isinstance(quote, str) else "")
            return None, stats

        cleaned: dict = {}
        for key, value in payload.items():
            result, stats = apply_gate(value, source, stats)
            if result is not None:
                cleaned[key] = result
        return cleaned, stats

    if isinstance(payload, list):
        cleaned_list = []
        for item in payload:
            result, stats = apply_gate(item, source, stats)
            # A list entry whose only field was rejected collapses to {}.
            if result is not None and result != {}:
                cleaned_list.append(result)
        return cleaned_list, stats

    return payload, stats


def _to_field(raw: object) -> GroundedField | None:
    if not isinstance(raw, dict) or "quote" not in raw:
        return None
    return GroundedField(**raw)


def extract_unit(unit: PatientUnit, complete=complete_json):
    """Run all five category prompts over one patient unit.

    `complete` is injected so tests can stub the API.
    """
    stats = GateStats()
    payloads: dict[str, dict] = {}

    for category in CATEGORIES:
        prompt = build_prompt(category, unit.text)
        raw = complete(prompt)
        cleaned, stats_for_category = apply_gate(raw, unit.text)
        stats.merge(stats_for_category)
        payloads[category] = cleaned or {}

    record = PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id)

    for name, raw in payloads.get("demographics", {}).get("fields", {}).items():
        parsed = _to_field(raw)
        if parsed is not None:
            record.demographics[name] = parsed

    for raw in payloads.get("timeline", {}).get("timeline", []):
        symptom = _to_field(raw.get("symptom"))
        if symptom is None:
            continue
        record.timeline.append(
            TimelineEvent(
                symptom=symptom,
                sequence_index=raw.get("sequence_index"),
                interval_value=raw.get("interval_value"),
                interval_unit=raw.get("interval_unit"),
            )
        )

    for raw in payloads.get("timeline", {}).get("prior_diagnoses", []):
        parsed = _to_field(raw)
        if parsed is not None:
            record.prior_diagnoses.append(parsed)

    for raw in payloads.get("labs", {}).get("labs", []):
        analyte = _to_field(raw.get("analyte"))
        if analyte is None:
            continue
        record.labs.append(
            LabValue(
                analyte=analyte,
                value_as_written=raw.get("value_as_written"),
                unit_as_written=raw.get("unit_as_written"),
            )
        )

    for raw in payloads.get("treatments", {}).get("treatments", []):
        treatment = _to_field(raw.get("treatment"))
        if treatment is None:
            continue
        record.treatments.append(
            TreatmentCourse(
                treatment=treatment,
                line_of_therapy=raw.get("line_of_therapy"),
                dose_as_written=raw.get("dose_as_written"),
                steroid_dose_before=_to_field(raw.get("steroid_dose_before")),
                steroid_dose_after=_to_field(raw.get("steroid_dose_after")),
                crp_before=_to_field(raw.get("crp_before")),
                crp_after=_to_field(raw.get("crp_after")),
                hemoglobin_after=_to_field(raw.get("hemoglobin_after")),
                platelets_after=_to_field(raw.get("platelets_after")),
                symptom_resolution=_to_field(raw.get("symptom_resolution")),
                follow_up_duration=_to_field(raw.get("follow_up_duration")),
                author_response_label=_to_field(
                    raw.get("author_response_label")
                ),
            )
        )

    outcome_payload = payloads.get("outcome", {})
    record.outcome = _to_field(outcome_payload.get("outcome"))
    record.reason_for_publication = _to_field(
        outcome_payload.get("reason_for_publication")
    )

    return record, stats
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/extract/test_runner.py -v
```

Expected: PASS, 8 passed

- [ ] **Step 5: Run the full suite to check for regressions**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest -v
```

Expected: all tests pass

- [ ] **Step 6: Smoke-test against one real article**

```bash
cd ~/Dev/rare-diseases && .venv/bin/python -c "
from vexas.fetch.pubmed import search_pmc_ids
from vexas.fetch.pmc import fetch_full_text, parse_article
from vexas.segment import split_patients
from vexas.extract.runner import extract_unit

pmc_id = 'PMC' + search_pmc_ids()[0]
article = parse_article(fetch_full_text(pmc_id))
unit = split_patients(article, [article.full_text])[0]
record, stats = extract_unit(unit)
print('kept:', stats.kept, 'rejected:', stats.rejected)
print('rejection rate: %.1f%%' % (stats.rejection_rate * 100))
print('timeline events:', len(record.timeline))
print('treatments:', len(record.treatments))
for q in stats.rejected_quotes[:5]:
    print('REJECTED:', q[:80])
"
```

Expected: a rejection rate under roughly 20%. If it is much higher, inspect the rejected quotes before proceeding — the usual causes are section text being altered during parsing, or the model normalizing whitespace inside quotes.

- [ ] **Step 7: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/extract/runner.py tests/extract/test_runner.py
git commit -m "Add extraction runner applying the gate before parsing

Ungrounded fields are stripped from raw JSON so they never become typed
objects. Rejection counts are retained as a quality signal."
```

---

### Task 9: Blind verification pass

**Files:**
- Create: `src/vexas/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Consumes: `complete_json`, `GroundedField`
- Produces:
  - `vexas.verify.Verdict` dataclass with `.supported: bool`, `.reason: str`
  - `vexas.verify.build_verify_prompt(field_name: str, field: GroundedField, source: str) -> str`
  - `vexas.verify.verify_field(field_name, field, source, complete=complete_json) -> Verdict`

The verification prompt must contain only the source text and one candidate field. It must never see the extraction prompt's output or reasoning — the independence between generating and verifying is the entire mechanism.

- [ ] **Step 1: Write the failing test**

Create `tests/test_verify.py`:

```python
from vexas.schema import Certainty, GroundedField
from vexas.verify import Verdict, build_verify_prompt, verify_field

SOURCE = "The patient had no evidence of ear chondritis at presentation."


def _field():
    return GroundedField(value="ear chondritis",
                         quote="no evidence of ear chondritis",
                         certainty=Certainty.NEGATED)


def _flat(field_name="symptom", field=None, source=SOURCE) -> str:
    blocks = build_verify_prompt(field_name, field or _field(), source)
    return "\n".join(b["text"] for b in blocks)


def test_prompt_contains_source_and_candidate():
    prompt = _flat()
    assert SOURCE in prompt
    assert "ear chondritis" in prompt


def test_prompt_does_not_leak_extraction_reasoning():
    """Independence between passes is the mechanism. Guard it."""
    lowered = _flat().lower()
    assert "step by step" not in lowered
    assert "you previously" not in lowered
    assert "extraction prompt" not in lowered


def test_prompt_asks_about_assertion_axes():
    assert "certainty" in _flat().lower()


def test_source_precedes_the_claim_and_carries_the_breakpoint():
    """Caching: ~30 verify calls per report share this prefix."""
    blocks = build_verify_prompt("symptom", _field(), SOURCE)
    source_index = next(i for i, b in enumerate(blocks) if SOURCE in b["text"])
    claim_index = next(
        i for i, b in enumerate(blocks) if b["text"].startswith("CLAIM:")
    )
    assert source_index < claim_index
    cached = [b for b in blocks if "cache_control" in b]
    assert len(cached) == 1 and SOURCE in cached[0]["text"]


def test_cached_prefix_is_identical_across_different_claims():
    a = build_verify_prompt("symptom", _field(), SOURCE)
    b = build_verify_prompt("treatment", _field(), SOURCE)
    assert a[:2] == b[:2]


def test_supported_verdict_is_parsed():
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"supported": True,
                                                    "reason": "explicit"})
    assert verdict.supported is True
    assert verdict.reason == "explicit"


def test_unsupported_verdict_is_parsed():
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"supported": False,
                                                    "reason": "wrong certainty"})
    assert verdict.supported is False


def test_missing_supported_key_defaults_to_unsupported():
    """Fail closed: an unparseable verdict must not silently pass."""
    verdict = verify_field("symptom", _field(), SOURCE,
                           complete=lambda p, **k: {"reason": "unclear"})
    assert verdict.supported is False


def test_api_failure_defaults_to_unsupported():
    def boom(prompt, **kwargs):
        raise RuntimeError("API down")

    verdict = verify_field("symptom", _field(), SOURCE, complete=boom)
    assert verdict.supported is False
    assert "error" in verdict.reason.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_verify.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.verify'`

- [ ] **Step 3: Write `src/vexas/verify.py`**

```python
"""Blind second pass.

Self-verification improves few-shot clinical IE (arXiv:2306.00024). The
mechanism is the asymmetry between generating and checking, which only
holds if the verifier is genuinely independent. So this prompt contains
the source text and ONE candidate field, and nothing else — no extraction
prompt, no prior reasoning, no sibling fields.

Fails closed: anything unparseable or errored is treated as unsupported.
A field wrongly dropped costs recall; a field wrongly kept costs
credibility, which is the more expensive failure here.
"""

from dataclasses import dataclass

from vexas.extract.client import complete_json
from vexas.schema import GroundedField

VERIFY_INSTRUCTIONS = """\
You are checking one extracted claim against a source text.

Answer only this: does the source text support this claim exactly as
stated, including its certainty, temporality and experiencer?

Reject the claim if:
- The quote does not appear in the source.
- The certainty is wrong (e.g. marked positive when the source says the
  finding was absent, or when the source only raises it as a possibility
  or something considered and excluded).
- The temporality is wrong (current finding vs past history).
- The experiencer is wrong (the finding belongs to a relative, not the
  patient).

Return ONLY: {"supported": true or false, "reason": "<short reason>"}
"""


@dataclass
class Verdict:
    supported: bool
    reason: str


def build_verify_prompt(field_name: str, field: GroundedField,
                        source: str) -> list[dict]:
    """Build a cache-friendly content block list.

    Same prefix rule as extract/prompts.build_prompt: instructions and
    source text are stable across every verification of a given report,
    the claim varies. This is the highest-volume call in the pipeline —
    roughly 30 per report — so the cached prefix matters most here.
    """
    return [
        {"type": "text", "text": VERIFY_INSTRUCTIONS},
        {
            "type": "text",
            "text": f"SOURCE TEXT:\n---\n{source}\n---",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"CLAIM:\n"
                f"  field: {field_name}\n"
                f"  value: {field.value}\n"
                f"  quote: {field.quote}\n"
                f"  certainty: {field.certainty.value}\n"
                f"  temporality: {field.temporality.value}\n"
                f"  experiencer: {field.experiencer.value}"
            ),
        },
    ]


def verify_field(field_name: str, field: GroundedField, source: str,
                 complete=complete_json) -> Verdict:
    prompt = build_verify_prompt(field_name, field, source)
    try:
        payload = complete(prompt, max_tokens=512)
    except Exception as exc:
        return Verdict(supported=False, reason=f"verification error: {exc}")

    supported = payload.get("supported")
    if not isinstance(supported, bool):
        return Verdict(supported=False, reason="unparseable verdict")
    return Verdict(supported=supported,
                   reason=str(payload.get("reason", "")))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_verify.py -v
```

Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/verify.py tests/test_verify.py
git commit -m "Add blind verification pass

The verifier sees only source text and one candidate field, preserving the
independence that makes self-verification work. Fails closed."
```

---

### Task 10: Deterministic unit normalization

Lab/imaging is the weakest extraction category for every model tested. The model never converts; Python does.

**Files:**
- Create: `src/vexas/normalize/__init__.py`, `src/vexas/normalize/units.py`
- Test: `tests/normalize/test_units.py`

**Interfaces:**
- Produces:
  - `vexas.normalize.units.CANONICAL: dict[str, str]` — analyte to canonical unit
  - `vexas.normalize.units.parse_numeric(text: str) -> float | None`
  - `vexas.normalize.units.convert(analyte: str, value: float, unit: str) -> tuple[float, str] | None`

- [ ] **Step 1: Write the failing test**

Create `tests/normalize/__init__.py` (empty) and `tests/normalize/test_units.py`:

```python
import pytest

from vexas.normalize.units import convert, parse_numeric


def test_parses_plain_number():
    assert parse_numeric("12.4") == 12.4


def test_parses_number_with_comparator():
    assert parse_numeric("<0.5") == 0.5


def test_parses_number_with_thousands_separator():
    assert parse_numeric("1,200") == 1200.0


def test_parses_range_by_taking_midpoint():
    assert parse_numeric("10-20") == 15.0


def test_returns_none_for_non_numeric():
    assert parse_numeric("not detected") is None


def test_crp_mg_dl_converts_to_mg_l():
    assert convert("CRP", 1.24, "mg/dL") == (12.4, "mg/L")


def test_crp_already_canonical_is_unchanged():
    assert convert("CRP", 12.4, "mg/L") == (12.4, "mg/L")


def test_hemoglobin_g_l_converts_to_g_dl():
    assert convert("hemoglobin", 124.0, "g/L") == (12.4, "g/dL")


def test_unit_matching_is_case_insensitive():
    assert convert("CRP", 1.0, "MG/DL") == (10.0, "mg/L")


def test_analyte_matching_is_case_insensitive():
    assert convert("crp", 1.0, "mg/dL") == (10.0, "mg/L")


def test_unknown_analyte_returns_none():
    assert convert("unobtainium", 1.0, "mg/dL") is None


def test_unknown_unit_returns_none():
    """Never guess. An unrecognized unit is a data-quality signal."""
    assert convert("CRP", 1.0, "furlongs") is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/normalize/test_units.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.normalize'`

- [ ] **Step 3: Write `src/vexas/normalize/__init__.py`**

```python
"""Deterministic normalization. Nothing in this package calls an LLM."""
```

- [ ] **Step 4: Write `src/vexas/normalize/units.py`**

```python
"""Unit conversion in code, never by the model.

Lab and imaging values are the weakest extraction category for every model
in CaseReportBench, and LLM arithmetic on clinical values is an unforced
error. The model copies `value_as_written` and `unit_as_written`; this
module converts.

Unrecognized analytes and units return None rather than a guess. A missing
conversion is a visible gap; a wrong one is invisible.
"""

import re

#: analyte (lowercase) -> canonical unit
CANONICAL: dict[str, str] = {
    "crp": "mg/L",
    "esr": "mm/h",
    "hemoglobin": "g/dL",
    "haemoglobin": "g/dL",
    "platelets": "10^9/L",
    "ferritin": "ng/mL",
    "mcv": "fL",
}

#: (analyte, from_unit) -> multiplier to reach the canonical unit
_FACTORS: dict[tuple[str, str], float] = {
    ("crp", "mg/l"): 1.0,
    ("crp", "mg/dl"): 10.0,
    ("esr", "mm/h"): 1.0,
    ("esr", "mm/hr"): 1.0,
    ("hemoglobin", "g/dl"): 1.0,
    ("hemoglobin", "g/l"): 0.1,
    ("haemoglobin", "g/dl"): 1.0,
    ("haemoglobin", "g/l"): 0.1,
    ("platelets", "10^9/l"): 1.0,
    ("platelets", "x10^9/l"): 1.0,
    ("platelets", "/ul"): 0.001,
    ("platelets", "10^3/ul"): 1.0,
    ("ferritin", "ng/ml"): 1.0,
    ("ferritin", "ug/l"): 1.0,
    ("mcv", "fl"): 1.0,
}

_NUMBER = re.compile(r"(?<!\d)-?\d+(?:\.\d+)?")


def parse_numeric(text: str) -> float | None:
    """Extract a number from a clinical value string.

    Handles comparators ('<0.5'), thousands separators ('1,200') and
    ranges ('10-20', taken as the midpoint). Returns None when there is
    no number to find.
    """
    if not text:
        return None
    cleaned = text.replace(",", "").strip()
    numbers = _NUMBER.findall(cleaned)
    if not numbers:
        return None
    if len(numbers) >= 2 and re.search(r"\d\s*[-–]\s*\d", cleaned):
        return (float(numbers[0]) + float(numbers[1])) / 2
    return float(numbers[0])


def convert(analyte: str, value: float, unit: str) -> tuple[float, str] | None:
    """Convert to the canonical unit. None if analyte or unit is unknown."""
    key = analyte.strip().lower()
    if key not in CANONICAL:
        return None
    factor = _FACTORS.get((key, unit.strip().lower()))
    if factor is None:
        return None
    converted = round(value * factor, 6)
    # Trim float noise from decimal multipliers (0.1 * 124 -> 12.4).
    return (round(converted, 4), CANONICAL[key])
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/normalize/test_units.py -v
```

Expected: PASS, 12 passed

- [ ] **Step 6: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/normalize/__init__.py src/vexas/normalize/units.py tests/normalize/
git commit -m "Add deterministic unit conversion

Unknown analytes and units return None rather than a guess. A missing
conversion is visible; a wrong one is not."
```

---

### Task 11: Ontology mapping with ID validation

The model never emits an identifier. GPT-4 scores 57.9% on HPO ID assignment versus 90.6% for a plain cosine-similarity baseline, because it produces plausible IDs with no mechanism to check they exist.

**Files:**
- Create: `src/vexas/normalize/ontology.py`
- Test: `tests/normalize/test_ontology.py`

**Interfaces:**
- Consumes: `DiskCache`
- Produces:
  - `vexas.normalize.ontology.OntologyTerm` dataclass with `.term_id`, `.label`, `.ontology`
  - `vexas.normalize.ontology.search_hpo(phrase: str, limit: int = 5) -> list[OntologyTerm]`
  - `vexas.normalize.ontology.lookup_rxnorm(drug: str) -> OntologyTerm | None`
  - `vexas.normalize.ontology.validate_hpo_id(term_id: str) -> bool`
  - `vexas.normalize.ontology.parse_ols_response(payload: dict) -> list[OntologyTerm]`

- [ ] **Step 1: Write the failing test**

Create `tests/normalize/test_ontology.py`:

```python
import pytest

from vexas.normalize.ontology import (
    OntologyTerm,
    is_wellformed_hpo_id,
    parse_ols_response,
    parse_rxnorm_response,
)


def test_parses_ols_search_response():
    payload = {"response": {"docs": [
        {"obo_id": "HP:0000969", "label": "Edema", "ontology_name": "hp"},
        {"obo_id": "HP:0002027", "label": "Abdominal pain",
         "ontology_name": "hp"},
    ]}}
    terms = parse_ols_response(payload)
    assert terms[0] == OntologyTerm(term_id="HP:0000969", label="Edema",
                                    ontology="hp")
    assert len(terms) == 2


def test_ols_docs_without_obo_id_are_skipped():
    payload = {"response": {"docs": [
        {"label": "No id here", "ontology_name": "hp"},
        {"obo_id": "HP:0000969", "label": "Edema", "ontology_name": "hp"},
    ]}}
    assert len(parse_ols_response(payload)) == 1


def test_empty_ols_response_yields_no_terms():
    assert parse_ols_response({"response": {"docs": []}}) == []


def test_malformed_ols_response_yields_no_terms():
    assert parse_ols_response({}) == []


def test_wellformed_hpo_id_accepted():
    assert is_wellformed_hpo_id("HP:0000969") is True


def test_malformed_hpo_ids_rejected():
    """Catches the shape of IDs models invent."""
    for bad in ["HP:969", "HPO:0000969", "HP0000969", "", "HP:abcdefg"]:
        assert is_wellformed_hpo_id(bad) is False


def test_parses_rxnorm_response():
    payload = {"idGroup": {"rxnormId": ["1193326"]}}
    term = parse_rxnorm_response(payload, "ruxolitinib")
    assert term.term_id == "RxCUI:1193326"
    assert term.label == "ruxolitinib"


def test_rxnorm_miss_returns_none():
    assert parse_rxnorm_response({"idGroup": {}}, "unobtainium") is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/normalize/test_ontology.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.normalize.ontology'`

- [ ] **Step 3: Write `src/vexas/normalize/ontology.py`**

```python
"""Ontology mapping: retrieve, then validate.

The division of labour is fixed and must not be relaxed:
  1. The extraction model emits only a surface phrase and its quote.
  2. This module retrieves candidate terms from the real ontology.
  3. Code validates the identifier exists.
  4. A model may re-rank among validated candidates, never invent one.

GPT-4 maps signs to HPO at 57.9% accuracy versus 90.6% for a plain
embedding baseline (arXiv:2408.01214), because it generates plausible IDs
with no mechanism to verify them (arXiv:2509.04458).

APIs used (both free, no key, verified live 2026-07-18):
  OLS4   https://www.ebi.ac.uk/ols4/api/search
  RxNorm https://rxnav.nlm.nih.gov/REST/rxcui.json
"""

import re
from dataclasses import dataclass

import httpx

from vexas.cache import DiskCache

OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
RXNORM_LOOKUP = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

#: Pinned for reproducibility — recorded in output metadata so a clinician
#: can reproduce any mapping. Update deliberately, never incidentally.
HPO_RELEASE = "2025-05-06"

_hpo_cache = DiskCache("ontology_hpo")
_rxnorm_cache = DiskCache("ontology_rxnorm")

_HPO_ID = re.compile(r"^HP:\d{7}$")


@dataclass(frozen=True)
class OntologyTerm:
    term_id: str
    label: str
    ontology: str


def is_wellformed_hpo_id(term_id: str) -> bool:
    """Shape check. Catches invented identifiers before a network call."""
    return bool(_HPO_ID.match(term_id or ""))


def parse_ols_response(payload: dict) -> list[OntologyTerm]:
    docs = payload.get("response", {}).get("docs", [])
    terms = []
    for doc in docs:
        term_id = doc.get("obo_id")
        if not term_id:
            continue
        terms.append(
            OntologyTerm(
                term_id=term_id,
                label=doc.get("label", ""),
                ontology=doc.get("ontology_name", ""),
            )
        )
    return terms


def parse_rxnorm_response(payload: dict, drug: str) -> OntologyTerm | None:
    ids = payload.get("idGroup", {}).get("rxnormId", [])
    if not ids:
        return None
    return OntologyTerm(term_id=f"RxCUI:{ids[0]}", label=drug,
                        ontology="rxnorm")


def search_hpo(phrase: str, limit: int = 5) -> list[OntologyTerm]:
    """Retrieve candidate HPO terms for a surface phrase. Cached."""
    cached = _hpo_cache.get_json(phrase)
    if cached is not None:
        return [OntologyTerm(**item) for item in cached]

    response = httpx.get(
        OLS4_SEARCH,
        params={"q": phrase, "ontology": "hp", "rows": limit},
        timeout=30,
    )
    response.raise_for_status()
    terms = parse_ols_response(response.json())
    # Belt and braces: reject anything malformed even from the real API.
    terms = [t for t in terms if is_wellformed_hpo_id(t.term_id)]
    _hpo_cache.put_json(phrase, [t.__dict__ for t in terms])
    return terms


def lookup_rxnorm(drug: str) -> OntologyTerm | None:
    """Resolve a drug name to an RxCUI. Cached."""
    cached = _rxnorm_cache.get_json(drug)
    if cached is not None:
        return OntologyTerm(**cached) if cached else None

    response = httpx.get(RXNORM_LOOKUP, params={"name": drug}, timeout=30)
    response.raise_for_status()
    term = parse_rxnorm_response(response.json(), drug)
    _rxnorm_cache.put_json(drug, term.__dict__ if term else None)
    return term
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/normalize/test_ontology.py -v
```

Expected: PASS, 8 passed

- [ ] **Step 5: Verify against the live APIs**

```bash
cd ~/Dev/rare-diseases && .venv/bin/python -c "
from vexas.normalize.ontology import search_hpo, lookup_rxnorm
for term in search_hpo('ear chondritis', limit=3):
    print(term.term_id, '-', term.label)
print(lookup_rxnorm('ruxolitinib'))
print(lookup_rxnorm('azacitidine'))
"
```

Expected: real HP: identifiers, and `RxCUI:1193326` for ruxolitinib.

- [ ] **Step 6: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/normalize/ontology.py tests/normalize/test_ontology.py
git commit -m "Add ontology mapping with identifier validation

Candidates come from OLS4 and RxNorm; identifiers are shape-checked and
never model-generated. HPO release pinned for reproducibility."
```

---

### Task 12: UBA1 variant normalization

**Files:**
- Create: `src/vexas/normalize/hgvs.py`
- Test: `tests/normalize/test_hgvs.py`

**Interfaces:**
- Produces:
  - `vexas.normalize.hgvs.normalize_variant(raw: str) -> str | None` — returns canonical short form (`M41T`) or None
  - `vexas.normalize.hgvs.AMINO_ACIDS: dict[str, str]`

- [ ] **Step 1: Write the failing test**

Create `tests/normalize/test_hgvs.py`:

```python
import pytest

from vexas.normalize.hgvs import normalize_variant


@pytest.mark.parametrize("raw", [
    "p.Met41Thr", "p.M41T", "M41T", "Met41Thr", "m41t", "  M41T  ",
])
def test_all_spellings_of_m41t_collapse(raw):
    assert normalize_variant(raw) == "M41T"


def test_m41v_and_m41l_are_distinguished():
    assert normalize_variant("p.Met41Val") == "M41V"
    assert normalize_variant("p.Met41Leu") == "M41L"


def test_non_m41_variant_is_normalized():
    assert normalize_variant("p.Ser56Phe") == "S56F"


def test_splice_variant_is_labelled_not_dropped():
    assert normalize_variant("c.118-1G>C") == "splice:c.118-1G>C"
    assert normalize_variant("c.118-2A>C") == "splice:c.118-2A>C"


def test_unrecognized_input_returns_none():
    assert normalize_variant("wild type") is None
    assert normalize_variant("") is None


def test_cdna_change_without_splice_context_returns_none():
    """c.122T>C is M41T but requires a codon table we do not have.
    Returning None is correct: better a gap than a wrong call."""
    assert normalize_variant("c.122T>C") is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/normalize/test_hgvs.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.normalize.hgvs'`

- [ ] **Step 3: Write `src/vexas/normalize/hgvs.py`**

```python
"""UBA1 variant nomenclature normalization.

Case reports spell the same variant many ways: p.Met41Thr, p.M41T, M41T,
c.122T>C. Grouping by raw string would scatter one variant across four
buckets. This collapses protein-level spellings to a canonical short form
and labels splice variants distinctly.

cDNA-only notation returns None rather than a guess — translating it needs
a codon table and transcript coordinates we deliberately do not carry. A
recorded gap beats a wrong genotype.
"""

import re

AMINO_ACIDS: dict[str, str] = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}

_SINGLE_LETTER = set(AMINO_ACIDS.values())

_THREE_LETTER = re.compile(
    r"^(?:p\.)?([A-Za-z]{3})(\d+)([A-Za-z]{3})$", re.IGNORECASE
)
_ONE_LETTER = re.compile(
    r"^(?:p\.)?([A-Za-z])(\d+)([A-Za-z])$", re.IGNORECASE
)
_SPLICE = re.compile(r"^c\.\d+[-+]\d+[ACGT]>[ACGT]$", re.IGNORECASE)


def normalize_variant(raw: str) -> str | None:
    """Return canonical short form (e.g. 'M41T'), 'splice:...', or None."""
    if not raw:
        return None
    text = raw.strip()

    if _SPLICE.match(text):
        return f"splice:{text}"

    match = _THREE_LETTER.match(text)
    if match:
        ref, position, alt = match.groups()
        ref_letter = AMINO_ACIDS.get(ref.lower())
        alt_letter = AMINO_ACIDS.get(alt.lower())
        if ref_letter and alt_letter:
            return f"{ref_letter}{position}{alt_letter}"
        return None

    match = _ONE_LETTER.match(text)
    if match:
        ref, position, alt = match.groups()
        ref_letter, alt_letter = ref.upper(), alt.upper()
        if ref_letter in _SINGLE_LETTER and alt_letter in _SINGLE_LETTER:
            return f"{ref_letter}{position}{alt_letter}"

    return None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/normalize/test_hgvs.py -v
```

Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/normalize/hgvs.py tests/normalize/test_hgvs.py
git commit -m "Add UBA1 variant normalization

Collapses protein-level spellings to a canonical short form. cDNA-only
notation returns None rather than a guess."
```

---

### Task 13: Claims-discipline guard

Every count carries its denominator. This is enforced by the type system, not by prose discipline.

**Files:**
- Create: `src/vexas/aggregate/__init__.py`, `src/vexas/aggregate/denominators.py`
- Test: `tests/aggregate/test_denominators.py`

**Interfaces:**
- Produces:
  - `vexas.aggregate.denominators.Count` frozen dataclass with `.n`, `.denominator`, `.label`, `.supporting_pmids`
  - `.as_text() -> str` — "reported in 14 of 76 reports"
  - `vexas.aggregate.denominators.Pattern` dataclass with `.title`, `.count`, `.caveat`
  - `vexas.aggregate.denominators.FORBIDDEN_PHRASES: frozenset[str]`
  - `vexas.aggregate.denominators.assert_no_overclaim(text: str) -> None` — raises `OverclaimError`

- [ ] **Step 1: Write the failing test**

Create `tests/aggregate/__init__.py` (empty) and `tests/aggregate/test_denominators.py`:

```python
import pytest

from vexas.aggregate.denominators import (
    Count,
    OverclaimError,
    Pattern,
    assert_no_overclaim,
)


def test_count_renders_with_denominator():
    count = Count(n=14, denominator=76, label="ear chondritis first",
                  supporting_pmids=["1", "2"])
    assert count.as_text() == "reported in 14 of 76 reports"


def test_count_rejects_n_greater_than_denominator():
    with pytest.raises(ValueError):
        Count(n=80, denominator=76, label="x", supporting_pmids=[])


def test_count_rejects_zero_denominator():
    with pytest.raises(ValueError):
        Count(n=0, denominator=0, label="x", supporting_pmids=[])


def test_count_has_no_percentage_property():
    """Deliberate omission. If it is easy to emit, it will be emitted."""
    count = Count(n=14, denominator=76, label="x", supporting_pmids=[])
    assert not hasattr(count, "percentage")
    assert not hasattr(count, "rate")


@pytest.mark.parametrize("text", [
    "This occurs in 18% of patients",
    "The prevalence of chondritis is high",
    "Ruxolitinib is more effective than tocilizumab",
    "Azacitidine causes remission",
    "The incidence rose over time",
])
def test_overclaiming_text_is_rejected(text):
    with pytest.raises(OverclaimError):
        assert_no_overclaim(text)


@pytest.mark.parametrize("text", [
    "reported in 14 of 76 reports",
    "described in 8 of 76 reports meeting inclusion criteria",
    "no report in this corpus described this sequence",
])
def test_legitimate_text_is_accepted(text):
    assert_no_overclaim(text)


def test_pattern_requires_a_caveat():
    pattern = Pattern(
        title="Chondritis precedes cytopenia",
        count=Count(n=14, denominator=76, label="x", supporting_pmids=["1"]),
    )
    assert "absence in reports" in pattern.caveat.lower()


def test_pattern_summary_passes_the_overclaim_guard():
    pattern = Pattern(
        title="Chondritis precedes cytopenia",
        count=Count(n=14, denominator=76, label="x", supporting_pmids=["1"]),
    )
    assert_no_overclaim(pattern.summary())
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_denominators.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.aggregate'`

- [ ] **Step 3: Write `src/vexas/aggregate/__init__.py`**

```python
"""Pattern aggregation."""
```

- [ ] **Step 4: Write `src/vexas/aggregate/denominators.py`**

```python
"""Claims discipline, enforced structurally.

Case reports have no denominator: we know how many REPORTS describe a
feature, never how many PATIENTS have it. Authors also preferentially
publish extraordinary findings and under-report ordinary ones. So any
statement shaped like "X% of patients" is indefensible regardless of how
carefully the extraction was done.

`Count` deliberately has no `.percentage` property. Anything trivial to
emit eventually gets emitted.

`assert_no_overclaim` is a tripwire, not a proof of completeness: it can
only catch phrasing we thought to list, so it will miss novel overclaims
worded differently. The real guarantee is structural -- `Count` always
carries its denominator and has no percentage property.
"""

import re
from dataclasses import dataclass, field

STANDING_CAVEAT = (
    "Counts are of published reports, not patients. Absence in reports "
    "does not imply absence in patients: authors under-report common "
    "features and preferentially publish unusual ones."
)

# Single-word terms are matched on word boundaries so they don't fire
# inside a larger innocent word (e.g. "incidence" inside "coincidence",
# "cures" inside "obscures"). Multi-word phrases are matched as substrings.
FORBIDDEN_WORDS: frozenset[str] = frozenset({
    "prevalence",
    "incidence",
    "causes",
    "efficacy",
    "cures",
    "rate",
})

FORBIDDEN_PHRASES: frozenset[str] = frozenset({
    "% of patients",
    "more effective than",
    "less effective than",
    "leads to remission",
    "superior to",
    "inferior to",
    "outperformed",
    "worked better than",
    "resulted in",
    "due to",
    "responsible for",
    "triggered by",
})


class OverclaimError(ValueError):
    """Raised when output text makes a claim case reports cannot support."""


def assert_no_overclaim(text: str) -> None:
    lowered = text.lower()
    for word in FORBIDDEN_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            raise OverclaimError(
                f"output contains forbidden word {word!r}: {text!r}"
            )
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            raise OverclaimError(
                f"output contains forbidden phrase {phrase!r}: {text!r}"
            )
    # Bare percentages are banned outright; counts carry denominators.
    if re.search(r"\d+\s*%", text):
        raise OverclaimError(f"output contains a bare percentage: {text!r}")


@dataclass(frozen=True)
class Count:
    n: int
    denominator: int
    label: str
    supporting_pmids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("denominator must be positive")
        if self.n > self.denominator:
            raise ValueError(
                f"n ({self.n}) exceeds denominator ({self.denominator})"
            )

    def as_text(self) -> str:
        return f"reported in {self.n} of {self.denominator} reports"


@dataclass
class Pattern:
    title: str
    count: Count
    caveat: str = STANDING_CAVEAT

    def summary(self) -> str:
        return f"{self.title}: {self.count.as_text()}"
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_denominators.py -v
```

Expected: PASS, 13 passed

- [ ] **Step 6: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/aggregate/ tests/aggregate/
git commit -m "Add claims-discipline guard

Count carries its denominator and has no percentage property. Forbidden
phrases raise rather than warn."
```

---

### Task 14: Onset sequence aggregation

The headline pattern. Existing reviews report what fraction of patients ever had a feature; none tabulate the order.

**Files:**
- Create: `src/vexas/aggregate/sequence.py`
- Test: `tests/aggregate/test_sequence.py`

**Interfaces:**
- Consumes: `PatientRecord`, `Count`, `Pattern`
- Produces:
  - `vexas.aggregate.sequence.ordered_pairs(record: PatientRecord) -> set[tuple[str, str]]`
  - `vexas.aggregate.sequence.first_manifestation(record: PatientRecord) -> str | None`
  - `vexas.aggregate.sequence.rank_precedence(records: list[PatientRecord], min_support: int = 3) -> list[Pattern]`

- [ ] **Step 1: Write the failing test**

Create `tests/aggregate/test_sequence.py`:

```python
from vexas.aggregate.sequence import (
    first_manifestation,
    ordered_pairs,
    rank_precedence,
)
from vexas.schema import Certainty, GroundedField, PatientRecord, TimelineEvent


def _event(symptom, index, certainty=Certainty.POSITIVE):
    return TimelineEvent(
        symptom=GroundedField(value=symptom, quote=f"quote for {symptom}",
                              certainty=certainty),
        sequence_index=index,
    )


def _record(pmid, events):
    return PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1", timeline=events)


def test_ordered_pairs_respects_sequence_index():
    record = _record("1", [_event("fever", 1), _event("chondritis", 2)])
    assert ordered_pairs(record) == {("fever", "chondritis")}


def test_three_events_yield_all_ordered_pairs():
    record = _record("1", [_event("fever", 1), _event("chondritis", 2),
                           _event("anemia", 3)])
    assert ordered_pairs(record) == {
        ("fever", "chondritis"), ("fever", "anemia"),
        ("chondritis", "anemia"),
    }


def test_unordered_events_produce_no_pairs():
    record = _record("1", [_event("fever", None), _event("chondritis", None)])
    assert ordered_pairs(record) == set()


def test_negated_symptoms_are_excluded():
    """A symptom the paper says was absent has no place in a sequence."""
    record = _record("1", [_event("fever", 1),
                           _event("chondritis", 2, Certainty.NEGATED)])
    assert ordered_pairs(record) == set()


def test_hypothetical_symptoms_are_excluded():
    record = _record("1", [_event("fever", 1),
                           _event("lymphoma", 2, Certainty.HYPOTHETICAL)])
    assert ordered_pairs(record) == set()


def test_first_manifestation_is_lowest_index():
    record = _record("1", [_event("chondritis", 2), _event("fever", 1)])
    assert first_manifestation(record) == "fever"


def test_first_manifestation_none_when_unordered():
    record = _record("1", [_event("fever", None)])
    assert first_manifestation(record) is None


def test_rank_precedence_requires_minimum_support():
    records = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(2)]
    assert rank_precedence(records, min_support=3) == []


def test_rank_precedence_surfaces_supported_pattern():
    records = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(5)]
    patterns = rank_precedence(records, min_support=3)
    assert len(patterns) == 1
    assert patterns[0].count.n == 5
    assert patterns[0].count.denominator == 5
    assert len(patterns[0].count.supporting_pmids) == 5


def test_denominator_counts_only_records_with_ordering():
    """Records that state no order cannot support or refute a precedence
    claim, so they must not inflate the denominator."""
    ordered = [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
               for i in range(4)]
    unordered = [_record("99", [_event("fever", None)])]
    patterns = rank_precedence(ordered + unordered, min_support=3)
    assert patterns[0].count.denominator == 4


def test_patterns_sorted_by_support_descending():
    records = (
        [_record(str(i), [_event("fever", 1), _event("chondritis", 2)])
         for i in range(5)]
        + [_record(f"b{i}", [_event("rash", 1), _event("anemia", 2)])
           for i in range(3)]
    )
    patterns = rank_precedence(records, min_support=3)
    assert patterns[0].count.n >= patterns[1].count.n
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_sequence.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.aggregate.sequence'`

- [ ] **Step 3: Write `src/vexas/aggregate/sequence.py`**

```python
"""Symptom onset ordering — the headline pattern.

Published VEXAS reviews are cross-sectional: they report what fraction of
patients ever had a feature. None tabulate per-case ordering, because doing
it by hand across ~200 papers is labour nobody spends. This is the gap.

Denominator discipline: a precedence claim is only supported or refuted by
records that state an order. Records with no ordering are excluded from
BOTH numerator and denominator, otherwise the denominator silently inflates
and every pattern looks weaker than the evidence warrants.
"""

from collections import defaultdict
from itertools import combinations

from vexas.aggregate.denominators import Count, Pattern
from vexas.schema import Certainty, PatientRecord

#: Only findings the report affirms belong in a sequence. Negated,
#: hypothetical and merely possible findings are excluded.
_ADMISSIBLE = {Certainty.POSITIVE}


def _ordered_events(record: PatientRecord):
    events = [
        event
        for event in record.timeline
        if event.sequence_index is not None
        and event.symptom.certainty in _ADMISSIBLE
        and event.symptom.value
    ]
    return sorted(events, key=lambda e: e.sequence_index)


def ordered_pairs(record: PatientRecord) -> set[tuple[str, str]]:
    """All (earlier, later) symptom pairs stated by this record."""
    events = _ordered_events(record)
    return {
        (a.symptom.value, b.symptom.value)
        for a, b in combinations(events, 2)
        if a.sequence_index < b.sequence_index
    }


def first_manifestation(record: PatientRecord) -> str | None:
    events = _ordered_events(record)
    return events[0].symptom.value if events else None


def rank_precedence(records: list[PatientRecord],
                    min_support: int = 3) -> list[Pattern]:
    """Rank 'A precedes B' claims by how many reports independently state
    them.

    Only records carrying ordering information count toward the
    denominator.
    """
    informative = [r for r in records if _ordered_events(r)]
    denominator = len(informative)
    if denominator == 0:
        return []

    support: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in informative:
        for pair in ordered_pairs(record):
            support[pair].append(record.pmid)

    patterns = []
    for (earlier, later), pmids in support.items():
        if len(pmids) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"{earlier} precedes {later}",
                count=Count(
                    n=len(pmids),
                    denominator=denominator,
                    label=f"{earlier}->{later}",
                    supporting_pmids=sorted(set(pmids)),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_sequence.py -v
```

Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/aggregate/sequence.py tests/aggregate/test_sequence.py
git commit -m "Add onset sequence aggregation

Ranks 'A precedes B' claims by independent report support. Records without
ordering are excluded from numerator and denominator alike."
```

---

### Task 15: Treatment response by line of therapy

Published meta-analyses pool by drug. None report response conditional on sequence position, which is the question a treating clinician actually faces.

**Files:**
- Create: `src/vexas/aggregate/treatment.py`
- Test: `tests/aggregate/test_treatment.py`

**Interfaces:**
- Consumes: `PatientRecord`, `TreatmentCourse`, `Count`, `Pattern`, `normalize.units.parse_numeric`
- Produces:
  - `vexas.aggregate.treatment.ResponseCall` enum: `RESPONDED`, `NOT_RESPONDED`, `INDETERMINATE`
  - `vexas.aggregate.treatment.call_response(course: TreatmentCourse) -> ResponseCall`
  - `vexas.aggregate.treatment.group_by_line(records) -> dict[tuple[str, int], list[TreatmentCourse]]`
  - `vexas.aggregate.treatment.rank_by_line(records, min_support: int = 3) -> list[Pattern]`

- [ ] **Step 1: Write the failing test**

Create `tests/aggregate/test_treatment.py`:

```python
from vexas.aggregate.treatment import (
    ResponseCall,
    call_response,
    group_by_line,
    rank_by_line,
)
from vexas.schema import GroundedField, PatientRecord, TreatmentCourse


def _f(value, quote="q"):
    return GroundedField(value=value, quote=quote)


def _course(drug, line, **kwargs):
    return TreatmentCourse(treatment=_f(drug), line_of_therapy=line, **kwargs)


def test_responded_requires_steroid_reduction_and_resolution():
    course = _course(
        "ruxolitinib", 2,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
        symptom_resolution=_f("yes"),
        follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.RESPONDED


def test_no_followup_is_indeterminate_even_if_it_looks_good():
    """Response without follow-up duration is meaningless."""
    course = _course(
        "ruxolitinib", 2,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
        symptom_resolution=_f("yes"),
        follow_up_duration=None,
    )
    assert call_response(course) is ResponseCall.INDETERMINATE


def test_missing_components_are_indeterminate_not_failure():
    course = _course("tocilizumab", 1, follow_up_duration=_f("6"))
    assert call_response(course) is ResponseCall.INDETERMINATE


def test_insufficient_steroid_reduction_is_not_a_response():
    course = _course(
        "anakinra", 1,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("35"),
        symptom_resolution=_f("yes"), follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.NOT_RESPONDED


def test_symptoms_unresolved_is_not_a_response():
    course = _course(
        "anakinra", 1,
        steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
        symptom_resolution=_f("no"), follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.NOT_RESPONDED


def test_author_label_is_ignored_when_calling_response():
    """The whole point: we apply one definition, not the paper's."""
    course = _course(
        "anakinra", 1,
        author_response_label=_f("complete response"),
        follow_up_duration=_f("12"),
    )
    assert call_response(course) is ResponseCall.INDETERMINATE


def test_group_by_line_keys_on_drug_and_line():
    records = [
        PatientRecord(pmid="1", patient_id="1-p1",
                      treatments=[_course("ruxolitinib", 2)]),
        PatientRecord(pmid="2", patient_id="2-p1",
                      treatments=[_course("ruxolitinib", 1)]),
    ]
    grouped = group_by_line(records)
    assert set(grouped) == {("ruxolitinib", 2), ("ruxolitinib", 1)}


def test_drug_names_are_lowercased_for_grouping():
    records = [
        PatientRecord(pmid="1", patient_id="1-p1",
                      treatments=[_course("Ruxolitinib", 1)]),
        PatientRecord(pmid="2", patient_id="2-p1",
                      treatments=[_course("ruxolitinib", 1)]),
    ]
    assert list(group_by_line(records)) == [("ruxolitinib", 1)]


def test_courses_without_a_line_are_excluded():
    records = [PatientRecord(pmid="1", patient_id="1-p1",
                             treatments=[_course("ruxolitinib", None)])]
    assert group_by_line(records) == {}


def test_rank_by_line_excludes_indeterminate_from_denominator():
    """Indeterminate courses can neither support nor refute; counting them
    in the denominator would understate every signal."""
    def responded(pmid):
        return PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1", treatments=[
            _course("ruxolitinib", 2,
                    steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
                    symptom_resolution=_f("yes"), follow_up_duration=_f("12"))
        ])

    def indeterminate(pmid):
        return PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1",
                             treatments=[_course("ruxolitinib", 2)])

    records = [responded(str(i)) for i in range(3)]
    records += [indeterminate(f"x{i}") for i in range(10)]
    patterns = rank_by_line(records, min_support=3)
    assert len(patterns) == 1
    assert patterns[0].count.n == 3
    assert patterns[0].count.denominator == 3


def test_rank_by_line_respects_min_support():
    records = [
        PatientRecord(pmid=str(i), patient_id=f"{i}-p1", treatments=[
            _course("ruxolitinib", 2,
                    steroid_dose_before=_f("40"), steroid_dose_after=_f("5"),
                    symptom_resolution=_f("yes"), follow_up_duration=_f("12"))
        ])
        for i in range(2)
    ]
    assert rank_by_line(records, min_support=3) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_treatment.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.aggregate.treatment'`

- [ ] **Step 3: Write `src/vexas/aggregate/treatment.py`**

```python
"""Treatment response by line of therapy.

Two things distinguish this from the published meta-analyses:

1. They pool by drug. We condition on sequence position. "Ruxolitinib as
   second line after tocilizumab failure" is a different question from
   "ruxolitinib", and it is the one a treating clinician faces.

2. No consensus response criteria exist in VEXAS — every paper defines
   "complete response" ad hoc, which is the stated main limitation of the
   published meta-analysis. So we ignore the authors' labels and apply ONE
   definition to the raw components, consistently, across every report.

Our definition, adapted from the momelotinib trial criteria and stated
openly so it can be argued with:
    RESPONDED = symptoms resolved
                AND steroid dose reduced by >= 50%
                AND follow-up duration is stated.
Anything missing a component is INDETERMINATE, never a failure. A report
that does not say is not a report that says no.
"""

from collections import defaultdict
from enum import Enum

from vexas.aggregate.denominators import Count, Pattern
from vexas.normalize.units import parse_numeric
from vexas.schema import PatientRecord, TreatmentCourse

#: Minimum fractional steroid reduction to count as a response.
STEROID_REDUCTION_THRESHOLD = 0.5

RESPONSE_DEFINITION = (
    "Symptoms resolved, steroid dose reduced by at least half, and a "
    "follow-up duration stated. Applied uniformly; authors' own response "
    "labels are not used."
)


class ResponseCall(str, Enum):
    RESPONDED = "responded"
    NOT_RESPONDED = "not_responded"
    INDETERMINATE = "indeterminate"


def _resolved(course: TreatmentCourse) -> bool | None:
    field = course.symptom_resolution
    if field is None or not field.value:
        return None
    return field.value.strip().lower() in {"yes", "true", "resolved",
                                           "complete"}


def call_response(course: TreatmentCourse) -> ResponseCall:
    """Apply one consistent response definition to raw components."""
    # Follow-up is required first: a response claim without it is
    # uninterpretable regardless of the other components.
    if course.follow_up_duration is None or not course.follow_up_duration.value:
        return ResponseCall.INDETERMINATE

    resolved = _resolved(course)
    if resolved is None:
        return ResponseCall.INDETERMINATE

    before = course.steroid_dose_before
    after = course.steroid_dose_after
    if before is None or after is None:
        return ResponseCall.INDETERMINATE

    before_value = parse_numeric(before.value or "")
    after_value = parse_numeric(after.value or "")
    if before_value is None or after_value is None or before_value <= 0:
        return ResponseCall.INDETERMINATE

    reduction = (before_value - after_value) / before_value
    if resolved and reduction >= STEROID_REDUCTION_THRESHOLD:
        return ResponseCall.RESPONDED
    return ResponseCall.NOT_RESPONDED


def group_by_line(
    records: list[PatientRecord],
) -> dict[tuple[str, int], list[tuple[str, TreatmentCourse]]]:
    """Group treatment courses by (drug, line of therapy).

    Values carry the source pmid so patterns stay traceable.
    """
    grouped: dict[tuple[str, int], list[tuple[str, TreatmentCourse]]] = (
        defaultdict(list)
    )
    for record in records:
        for course in record.treatments:
            if course.line_of_therapy is None or not course.treatment.value:
                continue
            key = (course.treatment.value.strip().lower(),
                   course.line_of_therapy)
            grouped[key].append((record.pmid, course))
    return dict(grouped)


def rank_by_line(records: list[PatientRecord],
                 min_support: int = 3) -> list[Pattern]:
    """Rank (drug, line) combinations by reports describing a response.

    Indeterminate courses are excluded from numerator AND denominator —
    they can neither support nor refute, and counting them would
    understate every signal.
    """
    patterns = []
    for (drug, line), entries in group_by_line(records).items():
        called = [
            (pmid, call_response(course))
            for pmid, course in entries
            if call_response(course) is not ResponseCall.INDETERMINATE
        ]
        if not called:
            continue
        responders = [pmid for pmid, call in called
                      if call is ResponseCall.RESPONDED]
        if len(responders) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"{drug} at line {line}: response reported",
                count=Count(
                    n=len(responders),
                    denominator=len(called),
                    label=f"{drug}@L{line}",
                    supporting_pmids=sorted(set(responders)),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_treatment.py -v
```

Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/aggregate/treatment.py tests/aggregate/test_treatment.py
git commit -m "Add treatment response by line of therapy

Applies one consistent response definition to raw components rather than
trusting each paper's ad hoc label. Indeterminate courses are excluded
from both numerator and denominator."
```

---

### Task 16: Diagnostic odyssey aggregation

**Files:**
- Create: `src/vexas/aggregate/odyssey.py`
- Test: `tests/aggregate/test_odyssey.py`

**Interfaces:**
- Consumes: `PatientRecord`, `Count`, `Pattern`
- Produces:
  - `vexas.aggregate.odyssey.MISDIAGNOSIS_ALIASES: dict[str, str]`
  - `vexas.aggregate.odyssey.canonical_diagnosis(raw: str) -> str`
  - `vexas.aggregate.odyssey.rank_misdiagnoses(records, min_support: int = 3) -> list[Pattern]`
  - `vexas.aggregate.odyssey.diagnostic_delays(records) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

Create `tests/aggregate/test_odyssey.py`:

```python
from vexas.aggregate.odyssey import (
    canonical_diagnosis,
    diagnostic_delays,
    rank_misdiagnoses,
)
from vexas.schema import Certainty, GroundedField, PatientRecord


def _f(value, quote="q", certainty=Certainty.POSITIVE):
    return GroundedField(value=value, quote=quote, certainty=certainty)


def _record(pmid, diagnoses, delay=None, delay_unit="years"):
    record = PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1",
                           prior_diagnoses=[_f(d) for d in diagnoses])
    if delay is not None:
        record.demographics["time_to_diagnosis"] = GroundedField(
            value=str(delay), unit_as_written=delay_unit, quote="q"
        )
    return record


def test_aliases_collapse_to_one_canonical_name():
    for raw in ["relapsing polychondritis", "Relapsing Polychondritis", "RP"]:
        assert canonical_diagnosis(raw) == "relapsing polychondritis"


def test_giant_cell_arteritis_aliases_collapse():
    for raw in ["GCA", "giant cell arteritis", "temporal arteritis"]:
        assert canonical_diagnosis(raw) == "giant cell arteritis"


def test_unknown_diagnosis_is_lowercased_not_dropped():
    assert canonical_diagnosis("Some Rare Thing") == "some rare thing"


def test_rank_misdiagnoses_counts_reports():
    records = [_record(str(i), ["relapsing polychondritis"])
               for i in range(4)]
    patterns = rank_misdiagnoses(records, min_support=3)
    assert patterns[0].count.n == 4
    assert "relapsing polychondritis" in patterns[0].title


def test_same_diagnosis_twice_in_one_report_counts_once():
    records = [_record("1", ["RP", "relapsing polychondritis"])]
    records += [_record(str(i), ["RP"]) for i in range(2, 5)]
    patterns = rank_misdiagnoses(records, min_support=3)
    assert patterns[0].count.n == 4


def test_negated_prior_diagnoses_are_excluded():
    record = PatientRecord(pmid="1", patient_id="1-p1", prior_diagnoses=[
        _f("relapsing polychondritis", certainty=Certainty.NEGATED)
    ])
    assert rank_misdiagnoses([record], min_support=1) == []


def test_denominator_counts_only_reports_mentioning_prior_diagnoses():
    with_diagnoses = [_record(str(i), ["RP"]) for i in range(3)]
    silent = [PatientRecord(pmid="9", patient_id="9-p1")]
    patterns = rank_misdiagnoses(with_diagnoses + silent, min_support=3)
    assert patterns[0].count.denominator == 3


def test_diagnostic_delays_normalizes_to_years():
    records = [_record("1", [], delay=24, delay_unit="months")]
    assert diagnostic_delays(records) == [("1", 2.0)]


def test_diagnostic_delays_skips_unparseable():
    records = [_record("1", [], delay="several", delay_unit="years")]
    assert diagnostic_delays(records) == []


def test_diagnostic_delays_skips_unknown_units():
    records = [_record("1", [], delay=3, delay_unit="fortnights")]
    assert diagnostic_delays(records) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_odyssey.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.aggregate.odyssey'`

- [ ] **Step 3: Write `src/vexas/aggregate/odyssey.py`**

```python
"""The diagnostic odyssey: what VEXAS was called before it was called VEXAS.

VEXAS is routinely misdiagnosed as relapsing polychondritis, giant cell
arteritis, Sweet syndrome, polymyalgia rheumatica, or MDS alone. Individual
case reports mention their own patient's wrong turns; nobody has tabulated
them across the literature.
"""

from collections import defaultdict

from vexas.aggregate.denominators import Count, Pattern
from vexas.normalize.units import parse_numeric
from vexas.schema import Certainty, PatientRecord

#: Alias -> canonical name. Lowercased keys.
MISDIAGNOSIS_ALIASES: dict[str, str] = {
    "rp": "relapsing polychondritis",
    "relapsing polychondritis": "relapsing polychondritis",
    "gca": "giant cell arteritis",
    "giant cell arteritis": "giant cell arteritis",
    "temporal arteritis": "giant cell arteritis",
    "pmr": "polymyalgia rheumatica",
    "polymyalgia rheumatica": "polymyalgia rheumatica",
    "sweet syndrome": "sweet syndrome",
    "sweet's syndrome": "sweet syndrome",
    "acute febrile neutrophilic dermatosis": "sweet syndrome",
    "mds": "myelodysplastic syndrome",
    "myelodysplastic syndrome": "myelodysplastic syndrome",
    "aosd": "adult-onset still disease",
    "adult onset still's disease": "adult-onset still disease",
    "adult-onset still's disease": "adult-onset still disease",
    "pan": "polyarteritis nodosa",
    "polyarteritis nodosa": "polyarteritis nodosa",
}

_TO_YEARS: dict[str, float] = {
    "year": 1.0, "years": 1.0, "yr": 1.0, "yrs": 1.0,
    "month": 1 / 12, "months": 1 / 12, "mo": 1 / 12, "mos": 1 / 12,
    "week": 1 / 52, "weeks": 1 / 52, "wk": 1 / 52, "wks": 1 / 52,
    "day": 1 / 365, "days": 1 / 365,
}


def canonical_diagnosis(raw: str) -> str:
    """Collapse aliases. Unknown names are lowercased, never dropped."""
    key = raw.strip().lower()
    return MISDIAGNOSIS_ALIASES.get(key, key)


def rank_misdiagnoses(records: list[PatientRecord],
                      min_support: int = 3) -> list[Pattern]:
    """Rank prior diagnoses by how many reports describe them.

    Denominator counts only reports that mention prior diagnoses at all —
    silence is not evidence the patient had none.
    """
    informative = [
        record for record in records
        if any(d.certainty is Certainty.POSITIVE and d.value
               for d in record.prior_diagnoses)
    ]
    denominator = len(informative)
    if denominator == 0:
        return []

    support: dict[str, set[str]] = defaultdict(set)
    for record in informative:
        for diagnosis in record.prior_diagnoses:
            if diagnosis.certainty is not Certainty.POSITIVE:
                continue
            if not diagnosis.value:
                continue
            # A set per pmid, so one report naming the same condition
            # twice still counts once.
            support[canonical_diagnosis(diagnosis.value)].add(record.pmid)

    patterns = []
    for diagnosis, pmids in support.items():
        if len(pmids) < min_support:
            continue
        patterns.append(
            Pattern(
                title=f"diagnosed as {diagnosis} before VEXAS confirmed",
                count=Count(
                    n=len(pmids),
                    denominator=denominator,
                    label=diagnosis,
                    supporting_pmids=sorted(pmids),
                ),
            )
        )
    return sorted(patterns, key=lambda p: p.count.n, reverse=True)


def diagnostic_delays(records: list[PatientRecord]) -> list[tuple[str, float]]:
    """Time from first symptom to UBA1 testing, in years, per report.

    Unparseable values and unknown units are skipped rather than guessed.
    """
    delays = []
    for record in records:
        field = record.demographics.get("time_to_diagnosis")
        if field is None or not field.value:
            continue
        value = parse_numeric(field.value)
        if value is None:
            continue
        factor = _TO_YEARS.get((field.unit_as_written or "").strip().lower())
        if factor is None:
            continue
        delays.append((record.pmid, round(value * factor, 3)))
    return delays
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/aggregate/test_odyssey.py -v
```

Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/aggregate/odyssey.py tests/aggregate/test_odyssey.py
git commit -m "Add diagnostic odyssey aggregation

Collapses misdiagnosis aliases and counts only reports that mention prior
diagnoses, since silence is not evidence of none."
```

---

### Task 17: Full pipeline runner and the validation check

If the pipeline cannot reproduce a known published finding, it cannot be trusted on a novel one. This task builds the end-to-end run and the credibility check together.

**Files:**
- Create: `src/vexas/pipeline.py`, `src/vexas/validate.py`
- Test: `tests/test_pipeline.py`, `tests/test_validate.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `vexas.pipeline.run(limit: int | None = None, verify: bool = True) -> PipelineResult`
  - `vexas.pipeline.PipelineResult` with `.records`, `.gate_stats`, `.corpus_size`, `.full_text_count`
  - `vexas.validate.check_genotype_gradient(records) -> dict[str, Count]`

Known findings this must recover, from the NIH cohort — these are published, so recovering them is validation, never discovery: M41V has the worst prognosis, M41L the most indolent course, M41T more ocular disease, and ear chondritis is protective for survival.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate.py`:

```python
from vexas.normalize.hgvs import normalize_variant
from vexas.schema import GroundedField, PatientRecord
from vexas.validate import KNOWN_FINDINGS, check_genotype_gradient


def _record(pmid, variant):
    record = PatientRecord(pmid=pmid, patient_id=f"{pmid}-p1")
    record.demographics["uba1_variant"] = GroundedField(value=variant,
                                                        quote="q")
    return record


def test_known_findings_are_documented_as_prior_art():
    """These are published. Recovering them validates; it does not discover."""
    assert "M41V" in KNOWN_FINDINGS
    assert "already published" in KNOWN_FINDINGS["M41V"].lower()


def test_variants_are_grouped_after_normalization():
    records = [_record("1", "p.Met41Thr"), _record("2", "M41T"),
               _record("3", "p.Met41Val")]
    counts = check_genotype_gradient(records)
    assert counts["M41T"].n == 2
    assert counts["M41V"].n == 1


def test_denominator_is_reports_with_a_genotype():
    records = [_record("1", "M41T"), _record("2", "M41T"),
               PatientRecord(pmid="3", patient_id="3-p1")]
    counts = check_genotype_gradient(records)
    assert counts["M41T"].denominator == 2


def test_unparseable_variants_are_excluded():
    records = [_record("1", "M41T"), _record("2", "wild type")]
    counts = check_genotype_gradient(records)
    assert counts["M41T"].denominator == 1


def test_empty_input_yields_no_counts():
    assert check_genotype_gradient([]) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_validate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.validate'`

- [ ] **Step 3: Write `src/vexas/validate.py`**

```python
"""Validation against published findings.

Everything in KNOWN_FINDINGS is already in the literature (NIH cohort,
Ferrada et al., Blood 2022; GeneReviews NBK614471). Recovering these
demonstrates the pipeline works. Presenting them as discoveries would be
misrepresentation, so the prior art is recorded alongside each one.
"""

from collections import defaultdict

from vexas.aggregate.denominators import Count
from vexas.normalize.hgvs import normalize_variant
from vexas.schema import PatientRecord

KNOWN_FINDINGS: dict[str, str] = {
    "M41V": (
        "Worst prognosis; highest mortality; less ear chondritis. "
        "Already published (NIH cohort, Blood 2022)."
    ),
    "M41T": (
        "Most common variant; more inflammatory eye disease. "
        "Already published (NIH cohort, Blood 2022)."
    ),
    "M41L": (
        "Most indolent course; more skin involvement. "
        "Already published (NIH cohort, Blood 2022)."
    ),
    "chondritis": (
        "Ear chondritis is protective for survival. "
        "Already published (NIH cohort, Blood 2022)."
    ),
}


def check_genotype_gradient(
    records: list[PatientRecord],
) -> dict[str, Count]:
    """Count reports per normalized UBA1 variant.

    Denominator is reports carrying a parseable genotype — reports that do
    not state one cannot contribute either way.
    """
    by_variant: dict[str, set[str]] = defaultdict(set)
    genotyped: set[str] = set()

    for record in records:
        field = record.demographics.get("uba1_variant")
        if field is None or not field.value:
            continue
        variant = normalize_variant(field.value)
        if variant is None:
            continue
        genotyped.add(record.pmid)
        by_variant[variant].add(record.pmid)

    denominator = len(genotyped)
    if denominator == 0:
        return {}

    return {
        variant: Count(
            n=len(pmids),
            denominator=denominator,
            label=variant,
            supporting_pmids=sorted(pmids),
        )
        for variant, pmids in by_variant.items()
    }
```

- [ ] **Step 4: Write `tests/test_pipeline.py`**

```python
from vexas.fetch.pmc import Article
from vexas.pipeline import PipelineResult, records_from_article
from vexas.schema import PatientRecord


def test_records_from_article_uses_stubbed_extraction():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t",
                      sections={"Case": "A 68-year-old man with fevers."})

    def fake_extract(unit, complete=None):
        from vexas.extract.runner import GateStats
        return (PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id),
                GateStats(kept=3, rejected=1))

    records, stats = records_from_article(article, extract=fake_extract)
    assert len(records) == 1
    assert records[0].patient_id == "39012345-p1"
    assert stats.kept == 3
    assert stats.rejected == 1


def test_pipeline_result_reports_corpus_and_full_text_counts():
    result = PipelineResult(records=[], gate_stats=None, corpus_size=197,
                            full_text_count=190)
    assert result.corpus_size == 197
    assert result.full_text_count == 190


def test_article_without_sections_yields_no_records():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t", sections={})

    def fake_extract(unit, complete=None):
        raise AssertionError("should not be called for an empty article")

    records, _ = records_from_article(article, extract=fake_extract)
    assert records == []
```

- [ ] **Step 5: Run the tests to verify they fail**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.pipeline'`

- [ ] **Step 6: Write `src/vexas/pipeline.py`**

```python
"""End-to-end pipeline.

Every stage caches, so re-running after a failure resumes rather than
restarts. Extraction is the expensive stage; its output is cached per
patient unit.
"""

from dataclasses import dataclass, field

from vexas.cache import DiskCache
from vexas.extract.runner import GateStats, extract_unit
from vexas.fetch.pmc import Article, fetch_full_text, parse_article
from vexas.fetch.pubmed import search_pmc_ids
from vexas.schema import PatientRecord
from vexas.segment import split_patients

_record_cache = DiskCache("records")


@dataclass
class PipelineResult:
    records: list[PatientRecord] = field(default_factory=list)
    gate_stats: GateStats | None = None
    corpus_size: int = 0
    full_text_count: int = 0
    failed_pmc_ids: list[str] = field(default_factory=list)


def records_from_article(article: Article, extract=extract_unit):
    """Extract every patient record from one article."""
    stats = GateStats()
    if not article.sections:
        return [], stats

    # Single-patient assumption for the MVP: the whole article is one
    # unit. Multi-patient splitting is wired here when detect_patient_count
    # is enabled — segmentation itself already supports N units.
    units = split_patients(article, [article.full_text])
    records = []
    for unit in units:
        record, unit_stats = extract(unit)
        stats.merge(unit_stats)
        records.append(record)
    return records, stats


def run(limit: int | None = None) -> PipelineResult:
    """Fetch, extract and gate the whole corpus."""
    pmc_ids = search_pmc_ids()
    result = PipelineResult(corpus_size=len(pmc_ids))
    stats = GateStats()

    for index, raw_id in enumerate(pmc_ids[:limit] if limit else pmc_ids, 1):
        pmc_id = f"PMC{raw_id}"
        try:
            article = parse_article(fetch_full_text(pmc_id))
        except Exception:
            result.failed_pmc_ids.append(pmc_id)
            continue

        if not article.sections:
            continue
        result.full_text_count += 1

        cached = _record_cache.get_json(pmc_id)
        if cached is not None:
            result.records.extend(PatientRecord(**item) for item in cached)
            continue

        records, article_stats = records_from_article(article)
        stats.merge(article_stats)
        _record_cache.put_json(pmc_id, [r.model_dump(mode="json")
                                        for r in records])
        result.records.extend(records)

        if index % 10 == 0:
            print(f"{index}/{len(pmc_ids)} processed, "
                  f"{len(result.records)} records")

    result.gate_stats = stats
    return result
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_pipeline.py tests/test_validate.py -v
```

Expected: PASS, 8 passed

- [ ] **Step 8: Run the pipeline on ten articles as a checkpoint**

```bash
cd ~/Dev/rare-diseases && .venv/bin/python -c "
from vexas.pipeline import run
from vexas.aggregate.sequence import rank_precedence
from vexas.aggregate.treatment import rank_by_line
from vexas.aggregate.odyssey import rank_misdiagnoses
from vexas.validate import check_genotype_gradient

result = run(limit=10)
print('corpus:', result.corpus_size, 'full text:', result.full_text_count)
print('records:', len(result.records))
print('gate rejection rate: %.1f%%' % (result.gate_stats.rejection_rate * 100))
for p in rank_precedence(result.records, min_support=2)[:5]:
    print(' SEQ ', p.summary())
for p in rank_by_line(result.records, min_support=1)[:5]:
    print(' TRT ', p.summary())
for p in rank_misdiagnoses(result.records, min_support=2)[:5]:
    print(' DX  ', p.summary())
print('genotypes:', {k: v.n for k, v in
                     check_genotype_gradient(result.records).items()})
"
```

Expected: real records, a gate rejection rate under roughly 20%, and at least one pattern. If no timeline events appear, inspect the timeline prompt against a real article before scaling to the full corpus — a full run is the expensive step and should not be launched on a broken extractor.

- [ ] **Step 9: Run the full corpus**

```bash
cd ~/Dev/rare-diseases && .venv/bin/python -c "
from vexas.pipeline import run
result = run()
print('records:', len(result.records))
print('full text:', result.full_text_count, 'of', result.corpus_size)
print('failed:', len(result.failed_pmc_ids))
print('gate rejection rate: %.1f%%' % (result.gate_stats.rejection_rate * 100))
" 2>&1 | tee data/full_run.log
```

- [ ] **Step 10: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/pipeline.py src/vexas/validate.py tests/test_pipeline.py tests/test_validate.py
git commit -m "Add end-to-end pipeline and genotype validation check

Known genotype findings are labelled as published prior art so recovering
them reads as validation rather than discovery."
```

---

### Task 18: Web UI with clickable traceability

Traceability is the product. Every pattern expands to its supporting reports; every field expands to the verbatim sentence that supports it.

**Files:**
- Create: `src/vexas/web/__init__.py`, `src/vexas/web/app.py`
- Create: `src/vexas/web/templates/base.html`, `index.html`, `pattern.html`
- Test: `tests/web/test_app.py`

**Interfaces:**
- Consumes: `pipeline.run`, all aggregators
- Produces: Flask app with routes `/`, `/pattern/<kind>/<index>`, `/report/<pmid>`

- [ ] **Step 1: Write the failing test**

Create `tests/web/__init__.py` (empty) and `tests/web/test_app.py`:

```python
import pytest

from vexas.aggregate.denominators import Count, Pattern
from vexas.web.app import create_app


@pytest.fixture
def app():
    patterns = {
        "sequence": [Pattern(
            title="fever precedes chondritis",
            count=Count(n=14, denominator=76, label="x",
                        supporting_pmids=["39012345", "39012346"]),
        )],
        "treatment": [],
        "odyssey": [],
    }
    application = create_app(patterns=patterns, records=[])
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200


def test_index_shows_pattern_with_denominator(client):
    body = client.get("/").get_data(as_text=True)
    assert "fever precedes chondritis" in body
    assert "14 of 76" in body


def test_index_never_shows_a_bare_percentage(client):
    """Claims discipline must hold in the rendered page, not just in
    the data layer."""
    import re
    body = client.get("/").get_data(as_text=True)
    # Strip the <style> block first: CSS percentages (color-mix, width:100%)
    # are not claims. This guard is about claim text in the rendered page.
    content = re.sub(r"<style.*?</style>", "", body, flags=re.DOTALL)
    assert not re.search(r"\d+\s*%", content)


def test_index_shows_the_standing_caveat(client):
    body = client.get("/").get_data(as_text=True)
    assert "absence in reports" in body.lower()


def test_pattern_detail_lists_supporting_pmids(client):
    body = client.get("/pattern/sequence/0").get_data(as_text=True)
    assert "39012345" in body
    assert "39012346" in body


def test_pattern_detail_links_to_pubmed(client):
    body = client.get("/pattern/sequence/0").get_data(as_text=True)
    assert "pubmed.ncbi.nlm.nih.gov/39012345" in body


def test_unknown_pattern_kind_returns_404(client):
    assert client.get("/pattern/nonexistent/0").status_code == 404


def test_out_of_range_pattern_index_returns_404(client):
    assert client.get("/pattern/sequence/99").status_code == 404
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/web/test_app.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.web'`

- [ ] **Step 3: Write `src/vexas/web/__init__.py`**

```python
"""Local web UI."""
```

- [ ] **Step 4: Write `src/vexas/web/app.py`**

```python
"""Flask app.

Patterns and records are injected rather than computed at request time, so
the UI never triggers an API call and the app starts instantly from cached
pipeline output.
"""

from flask import Flask, abort, render_template

from vexas.aggregate.denominators import STANDING_CAVEAT
from vexas.aggregate.treatment import RESPONSE_DEFINITION


def create_app(patterns: dict, records: list) -> Flask:
    app = Flask(__name__)
    records_by_pmid = {record.pmid: record for record in records}

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            patterns=patterns,
            caveat=STANDING_CAVEAT,
            response_definition=RESPONSE_DEFINITION,
        )

    @app.route("/pattern/<kind>/<int:index>")
    def pattern_detail(kind: str, index: int):
        if kind not in patterns:
            abort(404)
        try:
            pattern = patterns[kind][index]
        except IndexError:
            abort(404)
        supporting = [
            records_by_pmid.get(pmid) for pmid in pattern.count.supporting_pmids
        ]
        return render_template(
            "pattern.html",
            pattern=pattern,
            kind=kind,
            supporting=[(pmid, records_by_pmid.get(pmid))
                        for pmid in pattern.count.supporting_pmids],
            caveat=STANDING_CAVEAT,
        )

    return app
```

- [ ] **Step 5: Write `src/vexas/web/templates/base.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}VEXAS case-report miner{% endblock %}</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, sans-serif;
      max-width: 60rem; margin: 0 auto; padding: 2rem 1.5rem;
    }
    h1 { font-size: 1.6rem; margin-bottom: .25rem; }
    .sub { opacity: .7; margin-top: 0; }
    .caveat {
      border-left: 3px solid currentColor; padding: .75rem 1rem;
      margin: 1.5rem 0; opacity: .8; font-size: .9rem;
    }
    .pattern {
      border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
      border-radius: .5rem; padding: 1rem; margin: .75rem 0;
    }
    .pattern a { text-decoration: none; font-weight: 600; }
    .count { font-variant-numeric: tabular-nums; opacity: .75; }
    .quote {
      border-left: 2px solid color-mix(in srgb, currentColor 30%, transparent);
      padding-left: .75rem; margin: .5rem 0; font-style: italic; opacity: .85;
    }
    table { border-collapse: collapse; width: 100%; }
    td, th { text-align: left; padding: .4rem .6rem;
             border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
    .wrap { overflow-x: auto; }
  </style>
</head>
<body>
{% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 6: Write `src/vexas/web/templates/index.html`**

```html
{% extends "base.html" %}
{% block content %}
<h1>VEXAS syndrome case-report miner</h1>
<p class="sub">Longitudinal patterns extracted from published case reports.
Every claim is traceable to a verbatim source quote.</p>

<div class="caveat">{{ caveat }}</div>

<h2>Onset sequence</h2>
{% for pattern in patterns.get('sequence', []) %}
  <div class="pattern">
    <a href="/pattern/sequence/{{ loop.index0 }}">{{ pattern.title }}</a>
    <div class="count">{{ pattern.count.as_text() }}</div>
  </div>
{% else %}
  <p>No sequence pattern met the support threshold.</p>
{% endfor %}

<h2>Treatment response by line of therapy</h2>
<div class="caveat">Response definition applied uniformly:
{{ response_definition }}</div>
{% for pattern in patterns.get('treatment', []) %}
  <div class="pattern">
    <a href="/pattern/treatment/{{ loop.index0 }}">{{ pattern.title }}</a>
    <div class="count">{{ pattern.count.as_text() }}</div>
  </div>
{% else %}
  <p>No treatment pattern met the support threshold.</p>
{% endfor %}

<h2>Diagnostic odyssey</h2>
{% for pattern in patterns.get('odyssey', []) %}
  <div class="pattern">
    <a href="/pattern/odyssey/{{ loop.index0 }}">{{ pattern.title }}</a>
    <div class="count">{{ pattern.count.as_text() }}</div>
  </div>
{% else %}
  <p>No misdiagnosis pattern met the support threshold.</p>
{% endfor %}
{% endblock %}
```

- [ ] **Step 7: Write `src/vexas/web/templates/pattern.html`**

```html
{% extends "base.html" %}
{% block title %}{{ pattern.title }}{% endblock %}
{% block content %}
<p><a href="/">&larr; all patterns</a></p>
<h1>{{ pattern.title }}</h1>
<p class="count">{{ pattern.count.as_text() }}</p>

<div class="caveat">{{ caveat }}</div>

<h2>Supporting reports</h2>
<div class="wrap">
<table>
  <thead>
    <tr><th>PMID</th><th>Evidence</th></tr>
  </thead>
  <tbody>
  {% for pmid, record in supporting %}
    <tr>
      <td>
        <a href="https://pubmed.ncbi.nlm.nih.gov/{{ pmid }}/"
           target="_blank" rel="noopener">{{ pmid }}</a>
      </td>
      <td>
        {% if record %}
          {% for event in record.timeline %}
            <div class="quote">&ldquo;{{ event.symptom.quote }}&rdquo;</div>
          {% endfor %}
          {% for course in record.treatments %}
            <div class="quote">&ldquo;{{ course.treatment.quote }}&rdquo;</div>
          {% endfor %}
        {% else %}
          <em>record not loaded</em>
        {% endif %}
      </td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% endblock %}
```

- [ ] **Step 8: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/web/test_app.py -v
```

Expected: PASS, 8 passed

- [ ] **Step 9: Write the launcher `run_web.py` in the project root**

```python
"""Launch the UI from cached pipeline output."""

from vexas.aggregate.odyssey import rank_misdiagnoses
from vexas.aggregate.sequence import rank_precedence
from vexas.aggregate.treatment import rank_by_line
from vexas.pipeline import run
from vexas.web.app import create_app

if __name__ == "__main__":
    result = run()
    print(f"{len(result.records)} records, "
          f"{result.full_text_count} of {result.corpus_size} with full text")
    patterns = {
        "sequence": rank_precedence(result.records, min_support=3),
        "treatment": rank_by_line(result.records, min_support=3),
        "odyssey": rank_misdiagnoses(result.records, min_support=3),
    }
    create_app(patterns=patterns, records=result.records).run(
        debug=True, port=5001
    )
```

- [ ] **Step 10: Run the full suite and launch**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest -v && .venv/bin/python run_web.py
```

Expected: all tests pass, server starts on port 5001. Open http://localhost:5001 in Safari.

- [ ] **Step 11: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/web/ tests/web/ run_web.py
git commit -m "Add web UI with clickable traceability

Every pattern expands to supporting reports and the verbatim quotes behind
each field. The bare-percentage ban is tested against rendered HTML."
```

---

### Task 19: CLI, cost estimator, and bring-your-own-key setup

This tool is distributed open-source: users supply their own `ANTHROPIC_API_KEY`
and pay for their own runs. Two consequences shape this task. First, a missing
or invalid key is the single most likely first-run failure, so it must fail with
an actionable message rather than a stack trace. Second, users are spending
their own money, so they get an estimate before committing to a run.

**Files:**
- Create: `src/vexas/cli.py`, `README.md`
- Modify: `pyproject.toml` (add console entry point)
- Modify: `run_web.py` (accept `--limit`, or delete in favour of `vexas serve`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.run`, all aggregators, `config.settings`
- Produces:
  - `vexas.cli.estimate_cost(char_counts: list[int]) -> dict`
  - `vexas.cli.check_environment() -> list[str]` — human-readable problems, empty if ready
  - `vexas.cli.build_parser() -> argparse.ArgumentParser`
  - `vexas.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
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
    monkeypatch.setattr("vexas.cli.settings.anthropic_api_key", None,
                        raising=False)
    problems = check_environment()
    assert any("ANTHROPIC_API_KEY" in p for p in problems)


def test_check_environment_message_is_actionable(monkeypatch):
    """A first-run user must be told what to do, not just what is wrong."""
    monkeypatch.setattr("vexas.cli.settings.anthropic_api_key", None,
                        raising=False)
    joined = " ".join(check_environment())
    assert ".env" in joined
    assert "console.anthropic.com" in joined


def test_check_environment_passes_when_key_present(monkeypatch):
    monkeypatch.setattr("vexas.cli.settings.anthropic_api_key", "sk-ant-x",
                        raising=False)
    assert check_environment() == []


def test_estimate_does_not_require_an_api_key(monkeypatch):
    """Estimation is local arithmetic — users check the price first."""
    monkeypatch.setattr("vexas.cli.settings.anthropic_api_key", None,
                        raising=False)
    assert estimate_cost([30_000])["total_usd"] > 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'vexas.cli'`

- [ ] **Step 3: Write `src/vexas/cli.py`**

```python
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
#: Output tokens per call — extraction is verbose, verification is a verdict.
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
    print("  Estimate only — actual usage is reported after the run.")


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

    # run and serve both spend money — check setup first.
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
              "— see extract/prompts.build_prompt. This run cost ~5x more "
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd ~/Dev/rare-diseases && .venv/bin/pytest tests/test_cli.py -v
```

Expected: PASS, 10 passed

- [ ] **Step 5: Add the console entry point to `pyproject.toml`**

Add after the `[project.optional-dependencies]` block:

```toml
[project.scripts]
vexas = "vexas.cli:main"
```

Then reinstall so the command is available:

```bash
cd ~/Dev/rare-diseases && .venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 6: Write `README.md`**

````markdown
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
````

- [ ] **Step 7: Replace `run_web.py` with a thin shim**

```python
"""Deprecated. Use `vexas serve` instead."""

import sys

from vexas.cli import main

if __name__ == "__main__":
    print("run_web.py is deprecated — use: vexas serve --limit 25\n")
    sys.exit(main(["serve", *sys.argv[1:]]))
```

- [ ] **Step 8: Verify the CLI end to end**

```bash
cd ~/Dev/rare-diseases
.venv/bin/vexas check
.venv/bin/vexas estimate --limit 10
.venv/bin/pytest -v
```

Expected: `check` reports either readiness or actionable setup steps;
`estimate` prints a dollar figure without touching the Anthropic API; full
suite passes.

- [ ] **Step 9: Commit**

```bash
cd ~/Dev/rare-diseases
git add src/vexas/cli.py tests/test_cli.py README.md pyproject.toml run_web.py
git commit -m "Add CLI with cost estimation and bring-your-own-key setup

Users supply their own API key and pay for their own runs, so a missing key
fails with setup instructions rather than a stack trace, and estimation runs
without a key so the price is visible before spending. --limit bounds cost
for demo runs."
```

---

## Self-Review Notes

Spec coverage check, section by section:

| Spec requirement | Task |
|---|---|
| Substring gate, build first | 1 |
| Three assertion axes | 2 |
| `not_stated` ≠ `absent` | 2, 8 |
| Disk cache per stage | 3 |
| PubMed/PMC fetch, section structure | 4, 5 |
| Per-patient segmentation | 6 |
| Per-category prompts, no CoT | 7 |
| Gate applied before parsing | 8 |
| Blind verification pass | 9 |
| Units converted in code | 10 |
| Ontology IDs never model-generated, versions pinned | 11 |
| HGVS variant normalization | 12 |
| Counts with denominators, no bare percentages | 13, 18 |
| Onset sequencing | 14 |
| Treatment response by line, raw components | 15 |
| Diagnostic odyssey | 16 |
| Genotype gradient as validation, labelled prior art | 17 |
| Full-text-only corpus, denominator reported | 5, 17 |
| Publication-bias metadata per case | 2, 7 |
| Clickable traceability | 18 |
| Prompt caching / cost control | 7, 9, 19 |
| Bring-your-own-key distribution | 19 |
| CARE/JBI quality metadata | **not covered — see below** |

**Known gap:** the spec mentions CARE reporting-completeness and JBI critical-appraisal metadata per report. No task implements it. It is genuinely optional for the demo and adds a scoring pass over every report, so it is deferred rather than dropped — add it as Task 19 if time allows after Task 18 lands.

**Deferred by design:** `detect_patient_count` in Task 6 is specified but not wired into the pipeline; Task 17 treats each article as one patient unit. Multi-patient case series will therefore yield one merged record. `split_patients` already supports N units, so enabling it is a small change once single-patient extraction is verified working. This is a deliberate MVP simplification and should be stated in the demo if a case series appears in the results.

**Type consistency:** `GateStats.merge` used in Tasks 8 and 17; `Count.as_text()`
in 13, 14, 15, 16, 18; `Pattern.summary()` in 13 and 17; `parse_numeric` in 10,
15, 16. Names match across all uses.

**Prompt builders return `list[dict]`, not `str`.** Both
`extract.prompts.build_prompt` and `verify.build_verify_prompt` return content
blocks carrying the `cache_control` breakpoint. `complete_json` accepts either
form. Anything asserting on prompt text must flatten first — the tests in Tasks
7, 8, and 9 use a `_flat()` helper for this. A stub `complete` that string-matches
a raw prompt will fail; flatten inside the stub.

**Task 19 depends on Tasks 4, 5, 17, and 18** — `estimate` fetches and measures
the corpus, and `serve` needs the web app. It is last for that reason, but the
`check` and `estimate` subcommands are worth pulling forward if a user other
than you starts using the tool mid-build.
