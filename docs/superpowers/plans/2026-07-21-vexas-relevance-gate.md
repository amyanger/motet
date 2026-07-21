# VEXAS Relevance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-paper relevance triage stage that stops non-VEXAS papers from producing patient records, with every decision carrying a verbatim quote and reported as an auditable breakdown.

**Architecture:** A new `relevance.py` module classifies each article (Haiku 4.5) into `case_report` / `mention_only` / `review` / `uncertain` before extraction. `pipeline.run()` gates on the verdict: only `case_report` papers are extracted; the rest are recorded in a `TriageReport` on `PipelineResult`, cached, and printed in the `motet run` summary. The classifier fails closed to `uncertain` (surfaced for human review, never silently kept or dropped).

**Tech Stack:** Python 3.13+, Anthropic SDK (`claude-opus-4-8` for extraction, `claude-haiku-4-5-20251001` for the gate), pytest, src-layout.

**Design doc:** `docs/superpowers/specs/2026-07-21-vexas-relevance-gate-design.md`

## Global Constraints

- No em-dashes in code, docstrings, or output strings -- use `--`. (Global no-em-dash convention.)
- No bare percentages anywhere in program output. Print counts, not percentages.
- Every decisive relevance verdict (`case_report`, `mention_only`, `review`) carries a `quote` that `vexas.gate.is_grounded(quote, source)` confirms is a verbatim substring of the source. A verdict whose quote is not grounded fails closed to `uncertain`.
- Fail closed to `uncertain`: any exception, unparseable response, unknown label, or ungrounded quote yields `RelevanceLabel.UNCERTAIN`. Never fabricate a record, never silently discard a paper.
- Do NOT change extraction's prompt-cache block ordering (`extract/prompts.build_prompt`). The gate uses its own prompt and must not perturb the extraction or verification cache families.
- Run tests with `.venv/bin/pytest -q` (no `uv`; `pythonpath=["src"]`). One file: `.venv/bin/pytest tests/<path> -v`.
- Model IDs are exact: extraction `claude-opus-4-8`, gate `claude-haiku-4-5-20251001`.
- Commit after each task. Do NOT add Claude/AI co-author or attribution lines.

## File Structure

- Create: `src/vexas/relevance.py` -- relevance labels, verdict, prompt builder, `classify_relevance`.
- Create: `tests/test_relevance.py` -- classifier unit tests.
- Modify: `src/vexas/extract/client.py` -- add optional `model` param to `complete_json`.
- Modify: `src/vexas/config.py` -- add `relevance_model` setting.
- Modify: `src/vexas/pipeline.py` -- `TriageEntry`, `TriageReport`, `process_article`, `classify_cached`, `_relevance_cache`, `PipelineResult.triage`, gate wiring in `run()`.
- Modify: `tests/test_pipeline.py` -- routing, triage counting, caching tests.
- Modify: `src/vexas/cli.py` -- `format_triage_summary`, estimate gate term, print in `main()`.
- Modify: `tests/test_cli.py` -- estimate gate term and summary formatter tests.

---

### Task 1: `complete_json` model override + relevance model setting

**Files:**
- Modify: `src/vexas/extract/client.py` (function `complete_json`, lines 65-91)
- Modify: `src/vexas/config.py` (`Settings`, after line 22)
- Test: `tests/extract/test_client.py` (create if absent) or `tests/test_relevance.py` covers it indirectly -- add a dedicated client test here.

**Interfaces:**
- Consumes: nothing new.
- Produces: `complete_json(prompt, max_tokens=4096, max_retries=3, model=None) -> dict` -- when `model` is None it uses `settings.model`, otherwise the given model id. `settings.relevance_model: str` == `"claude-haiku-4-5-20251001"`.

- [ ] **Step 1: Write the failing test**

Create `tests/extract/test_client.py`:

```python
from vexas.extract import client
from vexas.config import settings


def test_complete_json_defaults_to_settings_model(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            class Msg:
                content = [type("B", (), {"text": '{"ok": true}'})()]
                usage = type("U", (), {"input_tokens": 1})()
            return Msg()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(client, "_client", FakeClient())
    result = client.complete_json([{"type": "text", "text": "hi"}])
    assert result == {"ok": True}
    assert captured["model"] == settings.model


def test_complete_json_uses_explicit_model(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            class Msg:
                content = [type("B", (), {"text": '{"ok": true}'})()]
                usage = type("U", (), {"input_tokens": 1})()
            return Msg()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(client, "_client", FakeClient())
    client.complete_json([{"type": "text", "text": "hi"}], model="claude-haiku-4-5-20251001")
    assert captured["model"] == "claude-haiku-4-5-20251001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/extract/test_client.py -v`
Expected: FAIL (`complete_json() got an unexpected keyword argument 'model'`).

- [ ] **Step 3: Add the `model` parameter**

In `src/vexas/extract/client.py`, change the signature and the `create` call:

```python
def complete_json(prompt: str | list[dict], max_tokens: int = 4096,
                  max_retries: int = 3, model: str | None = None) -> dict:
    """Send one prompt, return the parsed JSON object.

    `prompt` is either a plain string or a list of content blocks. The block
    form carries the `cache_control` breakpoint that makes repeated calls
    about the same report cost 0.1x -- see extract/prompts.build_prompt.
    `model` overrides settings.model (the relevance gate runs on a cheaper
    model than extraction).
    """
    client = _get_client()
    content = prompt if isinstance(prompt, list) else [
        {"type": "text", "text": prompt}
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model or settings.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            _log_cache_usage(message)
            return extract_json_block(message.content[0].text)
        except APIError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"API failed after {max_retries} attempts") from last_error
```

- [ ] **Step 4: Add the setting**

In `src/vexas/config.py`, add after the `model` field (line 22):

```python
    model: str = "claude-opus-4-8"
    #: Relevance triage runs on a cheaper model -- it is a 4-way label with a
    #: supporting quote, not structured extraction.
    relevance_model: str = "claude-haiku-4-5-20251001"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/extract/test_client.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full suite (no regressions)**

Run: `.venv/bin/pytest -q`
Expected: all pass (250 + 2 new = 252).

- [ ] **Step 7: Commit**

```bash
git add src/vexas/extract/client.py src/vexas/config.py tests/extract/test_client.py
git commit -m "feat: allow per-call model override in complete_json for the relevance gate"
```

---

### Task 2: `relevance.py` -- classifier module

**Files:**
- Create: `src/vexas/relevance.py`
- Test: `tests/test_relevance.py`

**Interfaces:**
- Consumes: `complete_json(prompt, max_tokens, model)` (Task 1); `vexas.gate.is_grounded(quote, source)`; `vexas.fetch.pmc.Article` (has `.full_text: str`); `settings.relevance_model`.
- Produces:
  - `class RelevanceLabel(str, Enum)` with members `CASE_REPORT="case_report"`, `MENTION_ONLY="mention_only"`, `REVIEW="review"`, `UNCERTAIN="uncertain"`.
  - `@dataclass(frozen=True) class RelevanceVerdict: label: RelevanceLabel; quote: str; reason: str`.
  - `build_relevance_prompt(text: str) -> list[dict]`.
  - `classify_relevance(article: Article, complete=complete_json) -> RelevanceVerdict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_relevance.py`:

```python
from vexas.fetch.pmc import Article
from vexas.relevance import (
    RelevanceLabel,
    RelevanceVerdict,
    build_relevance_prompt,
    classify_relevance,
)

# A source with a clear VEXAS diagnosis sentence used across tests.
CASE_TEXT = (
    "Case\nA 68-year-old man presented with fevers and chondritis. "
    "Genetic testing confirmed a somatic UBA1 p.Met41Thr variant, "
    "and he was diagnosed with VEXAS syndrome."
)
MENTION_TEXT = (
    "Discussion\nThe differential for this patient with Adult-onset Still's "
    "disease included VEXAS syndrome, which was considered and excluded."
)


def _article(text):
    return Article(pmc_id="PMC1", pmid="1", title="t", sections={"S": text})


def _fake(payload):
    def complete(prompt, max_tokens=512, model=None):
        return payload
    return complete


def test_prompt_includes_source_text_and_instruction():
    blocks = build_relevance_prompt("SRC-BODY")
    joined = " ".join(b["text"] for b in blocks)
    assert "SRC-BODY" in joined
    assert "VEXAS" in joined
    # Single call: no cache breakpoint anywhere.
    assert all("cache_control" not in b for b in blocks)


def test_case_report_with_grounded_quote():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "case_report",
                        "quote": "he was diagnosed with VEXAS syndrome",
                        "reason": "patient diagnosed with VEXAS"}),
    )
    assert verdict.label is RelevanceLabel.CASE_REPORT
    assert verdict.quote == "he was diagnosed with VEXAS syndrome"


def test_mention_only_with_grounded_quote():
    verdict = classify_relevance(
        _article(MENTION_TEXT),
        complete=_fake({"label": "mention_only",
                        "quote": "included VEXAS syndrome, which was considered and excluded",
                        "reason": "only in the differential"}),
    )
    assert verdict.label is RelevanceLabel.MENTION_ONLY


def test_ungrounded_quote_fails_closed_to_uncertain():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "case_report",
                        "quote": "this sentence is not in the source",
                        "reason": "x"}),
    )
    assert verdict.label is RelevanceLabel.UNCERTAIN
    assert verdict.quote == ""


def test_unknown_label_fails_closed_to_uncertain():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "definitely", "quote": "", "reason": "x"}),
    )
    assert verdict.label is RelevanceLabel.UNCERTAIN


def test_exception_fails_closed_to_uncertain():
    def boom(prompt, max_tokens=512, model=None):
        raise RuntimeError("api down")

    verdict = classify_relevance(_article(CASE_TEXT), complete=boom)
    assert verdict.label is RelevanceLabel.UNCERTAIN
    assert "api down" in verdict.reason


def test_model_uncertain_passes_through_without_grounding():
    verdict = classify_relevance(
        _article(CASE_TEXT),
        complete=_fake({"label": "uncertain", "quote": "", "reason": "unclear"}),
    )
    assert verdict.label is RelevanceLabel.UNCERTAIN
    assert verdict.reason == "unclear"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_relevance.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'vexas.relevance'`).

- [ ] **Step 3: Write the module**

Create `src/vexas/relevance.py`:

```python
"""Relevance triage -- stage 2.5.

The corpus query returns every paper that mentions VEXAS, not only papers
that report a VEXAS patient. Extracting a paper that only names VEXAS in a
differential yields a full wrong-disease record that inflates every
denominator. This stage classifies each paper before extraction.

It fails closed to UNCERTAIN: an errored, unparseable, or ungrounded verdict
is surfaced for human review, never silently kept and never silently dropped.
A verdict that keeps or excludes a paper must carry a verbatim quote that the
substring gate confirms is in the source, the same discipline extraction uses.
"""

from dataclasses import dataclass
from enum import Enum

from vexas.config import settings
from vexas.extract.client import complete_json
from vexas.fetch.pmc import Article
from vexas.gate import is_grounded


class RelevanceLabel(str, Enum):
    CASE_REPORT = "case_report"
    MENTION_ONLY = "mention_only"
    REVIEW = "review"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class RelevanceVerdict:
    label: RelevanceLabel
    quote: str
    reason: str


RELEVANCE_INSTRUCTIONS = """\
You are triaging a biomedical paper. Decide what kind of paper this is with
respect to VEXAS syndrome, using ONLY the source text below.

Choose exactly one label:
- "case_report": the paper reports at least one patient who was diagnosed with
  VEXAS syndrome (a somatic UBA1 variant, or an explicit VEXAS diagnosis). If
  the paper both reviews the literature AND reports such a patient, it is still
  case_report.
- "mention_only": VEXAS appears only in a differential diagnosis, as background,
  or in an editorial, and no VEXAS patient is actually reported here.
- "review": a review or meta-analysis with no newly reported VEXAS patient.
- "uncertain": you cannot tell from the text.

For case_report, mention_only, and review you MUST include a "quote" copied
VERBATIM, character for character, from the source text that best supports the
label (for case_report, the sentence stating the patient's VEXAS or UBA1
diagnosis; for mention_only, the sentence showing VEXAS is only referenced). Do
not paraphrase the quote. If you cannot find a supporting verbatim quote, use
"uncertain" with an empty quote.

Return ONLY this JSON object, no commentary:
{"label": "<one of the four>", "quote": "<verbatim supporting text>", "reason": "<short reason>"}
"""


def build_relevance_prompt(text: str) -> list[dict]:
    """Two content blocks: instruction, then source text.

    No cache_control: this is a single call per paper, so there is no later
    call to read a cache write back at 0.1x. A cache write would only add its
    1.25x premium. Extraction's cache families are untouched.
    """
    return [
        {"type": "text", "text": RELEVANCE_INSTRUCTIONS},
        {"type": "text", "text": f"SOURCE TEXT:\n---\n{text}\n---"},
    ]


def classify_relevance(article: Article, complete=complete_json) -> RelevanceVerdict:
    source = article.full_text
    prompt = build_relevance_prompt(source)
    try:
        payload = complete(prompt, max_tokens=512, model=settings.relevance_model)
    except Exception as exc:
        return RelevanceVerdict(RelevanceLabel.UNCERTAIN, "", f"gate error: {exc}")

    raw_label = payload.get("label")
    quote = payload.get("quote", "")
    reason = str(payload.get("reason", ""))
    if not isinstance(quote, str):
        quote = ""

    try:
        label = RelevanceLabel(raw_label)
    except ValueError:
        return RelevanceVerdict(
            RelevanceLabel.UNCERTAIN, "", f"unknown label: {raw_label!r}"
        )

    if label is RelevanceLabel.UNCERTAIN:
        return RelevanceVerdict(label, quote, reason)

    # A decisive label must be backed by a grounded quote, or it fails closed.
    if not is_grounded(quote, source):
        return RelevanceVerdict(
            RelevanceLabel.UNCERTAIN, "", "quote not grounded in source"
        )
    return RelevanceVerdict(label, quote, reason)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_relevance.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (252 + 7 = 259).

- [ ] **Step 6: Commit**

```bash
git add src/vexas/relevance.py tests/test_relevance.py
git commit -m "feat: add VEXAS relevance classifier with fail-closed grounding"
```

---

### Task 3: Pipeline integration -- triage report, per-article gate, caching

**Files:**
- Modify: `src/vexas/pipeline.py` (add types + helpers + `PipelineResult.triage`; imports)
- Test: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `classify_relevance` and `RelevanceLabel`/`RelevanceVerdict` (Task 2); existing `records_from_article`, `GateStats`, `VerifyStats`, `DiskCache`.
- Produces:
  - `@dataclass class TriageEntry: pmc_id: str; label: str; quote: str; reason: str`.
  - `@dataclass class TriageReport` with int fields `case_report`, `mention_only`, `review`, `uncertain` (default 0); `excluded: list[TriageEntry]`; `flagged: list[TriageEntry]`; method `record(pmc_id: str, verdict: RelevanceVerdict) -> None`; property `total -> int`.
  - `process_article(article, pmc_id, classify=classify_relevance, extract=extract_unit, segment=segment_texts, verify=False, verify_fn=verify_record) -> tuple[list[PatientRecord], GateStats, VerifyStats, RelevanceVerdict]`.
  - `classify_cached(article, pmc_id, cache, classify=classify_relevance) -> RelevanceVerdict`.
  - `PipelineResult.triage: TriageReport | None = None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
from vexas.cache import DiskCache
from vexas.pipeline import (
    TriageEntry,
    TriageReport,
    classify_cached,
    process_article,
)
from vexas.relevance import RelevanceLabel, RelevanceVerdict


def _case_verdict():
    return RelevanceVerdict(RelevanceLabel.CASE_REPORT, "diagnosed with VEXAS", "ok")


def test_case_report_article_is_extracted():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t",
                      sections={"Case": "diagnosed with VEXAS"})

    def fake_extract(unit, complete=None):
        return (PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id),
                GateStats(kept=2))

    records, gate_stats, _, verdict = process_article(
        article, "PMC1", classify=lambda a: _case_verdict(),
        extract=fake_extract, segment=_one_unit,
    )
    assert len(records) == 1
    assert verdict.label is RelevanceLabel.CASE_REPORT
    assert gate_stats.kept == 2


def test_mention_only_article_skips_extraction():
    # PMC8056719-style contamination paper: VEXAS only in the differential.
    article = Article(pmc_id="PMC8056719", pmid="1", title="t",
                      sections={"Discussion": "VEXAS was considered in the differential"})

    def fake_extract(unit, complete=None):
        raise AssertionError("a non-case-report must not be extracted")

    verdict = RelevanceVerdict(
        RelevanceLabel.MENTION_ONLY, "VEXAS was considered in the differential",
        "differential only")
    records, _, _, out = process_article(
        article, "PMC8056719", classify=lambda a: verdict,
        extract=fake_extract, segment=_one_unit,
    )
    assert records == []
    assert out.label is RelevanceLabel.MENTION_ONLY


def test_uncertain_article_skips_extraction():
    article = Article(pmc_id="PMC1", pmid="1", title="t",
                      sections={"S": "ambiguous text"})

    def fake_extract(unit, complete=None):
        raise AssertionError("an uncertain paper must not be extracted")

    records, _, _, out = process_article(
        article, "PMC1",
        classify=lambda a: RelevanceVerdict(RelevanceLabel.UNCERTAIN, "", "unclear"),
        extract=fake_extract, segment=_one_unit,
    )
    assert records == []
    assert out.label is RelevanceLabel.UNCERTAIN


def test_triage_report_counts_and_lists():
    report = TriageReport()
    report.record("PMC1", RelevanceVerdict(RelevanceLabel.CASE_REPORT, "q", "r"))
    report.record("PMC2", RelevanceVerdict(RelevanceLabel.MENTION_ONLY, "q", "r"))
    report.record("PMC3", RelevanceVerdict(RelevanceLabel.REVIEW, "q", "r"))
    report.record("PMC4", RelevanceVerdict(RelevanceLabel.UNCERTAIN, "", "r"))
    assert report.case_report == 1
    assert report.mention_only == 1
    assert report.review == 1
    assert report.uncertain == 1
    assert report.total == 4
    # Excluded = mention_only + review; flagged = uncertain.
    assert [e.pmc_id for e in report.excluded] == ["PMC2", "PMC3"]
    assert [e.pmc_id for e in report.flagged] == ["PMC4"]


def test_classify_cached_calls_model_once_across_runs(tmp_path):
    article = Article(pmc_id="PMC1", pmid="1", title="t", sections={"S": "x"})
    cache = DiskCache("relevance", root=tmp_path)
    calls = {"n": 0}

    def counting_classify(a):
        calls["n"] += 1
        return _case_verdict()

    first = classify_cached(article, "PMC1", cache, classify=counting_classify)
    second = classify_cached(article, "PMC1", cache, classify=counting_classify)
    assert calls["n"] == 1
    assert first.label is second.label is RelevanceLabel.CASE_REPORT
    assert second.quote == "diagnosed with VEXAS"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL (`ImportError: cannot import name 'TriageReport'`).

- [ ] **Step 3: Add types, helpers, and the triage field**

In `src/vexas/pipeline.py`, add imports near the top:

```python
from vexas.relevance import RelevanceLabel, RelevanceVerdict, classify_relevance
```

Add a second cache alongside the existing record caches (after line 22):

```python
_relevance_cache = DiskCache("relevance")
```

Add the triage types (place them above `PipelineResult`):

```python
@dataclass
class TriageEntry:
    pmc_id: str
    label: str
    quote: str
    reason: str


@dataclass
class TriageReport:
    """Per-paper relevance breakdown -- the denominator refinement itself.

    Counts every full-text paper by label. Excluded lists the papers removed
    from extraction (mention_only + review); flagged lists uncertain papers
    surfaced for human review. Both carry the verbatim quote behind the call.
    """
    case_report: int = 0
    mention_only: int = 0
    review: int = 0
    uncertain: int = 0
    excluded: list[TriageEntry] = field(default_factory=list)
    flagged: list[TriageEntry] = field(default_factory=list)

    def record(self, pmc_id: str, verdict: RelevanceVerdict) -> None:
        entry = TriageEntry(pmc_id, verdict.label.value, verdict.quote,
                            verdict.reason)
        if verdict.label is RelevanceLabel.CASE_REPORT:
            self.case_report += 1
        elif verdict.label is RelevanceLabel.MENTION_ONLY:
            self.mention_only += 1
            self.excluded.append(entry)
        elif verdict.label is RelevanceLabel.REVIEW:
            self.review += 1
            self.excluded.append(entry)
        else:
            self.uncertain += 1
            self.flagged.append(entry)

    @property
    def total(self) -> int:
        return (self.case_report + self.mention_only
                + self.review + self.uncertain)
```

Add `triage` to `PipelineResult`:

```python
@dataclass
class PipelineResult:
    records: list[PatientRecord] = field(default_factory=list)
    gate_stats: GateStats | None = None
    verify_stats: VerifyStats | None = None
    corpus_size: int = 0
    full_text_count: int = 0
    failed_pmc_ids: list[str] = field(default_factory=list)
    triage: "TriageReport | None" = None
```

Add the two helpers (after `records_from_article`):

```python
def classify_cached(article, pmc_id, cache, classify=classify_relevance):
    """Return the relevance verdict for one paper, cached by PMC id.

    Resume-not-restart, like every stage: a re-run never re-calls the gate.
    """
    cached = cache.get_json(pmc_id)
    if cached is not None:
        return RelevanceVerdict(
            RelevanceLabel(cached["label"]), cached["quote"], cached["reason"]
        )
    verdict = classify(article)
    cache.put_json(pmc_id, {"label": verdict.label.value,
                            "quote": verdict.quote, "reason": verdict.reason})
    return verdict


def process_article(article, pmc_id, classify=classify_relevance,
                    extract=extract_unit, segment=segment_texts,
                    verify: bool = False, verify_fn=verify_record):
    """Gate one article, then extract only if it is a VEXAS case report.

    Returns (records, gate_stats, verify_stats, verdict). A non-case-report
    returns no records and empty stats -- extraction is never called.
    """
    verdict = classify(article)
    if verdict.label is not RelevanceLabel.CASE_REPORT:
        return [], GateStats(), VerifyStats(), verdict
    records, gate_stats, verify_stats = records_from_article(
        article, extract=extract, segment=segment,
        verify=verify, verify_fn=verify_fn,
    )
    return records, gate_stats, verify_stats, verdict
```

Note: `process_article` takes `classify` and `extract`/`segment`/`verify_fn` injectable so it is testable offline exactly like `records_from_article`. `run()` (Task 4) supplies a cache-wrapping classify.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: PASS (existing 6 + 5 new = 11).

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (259 + 5 = 264).

- [ ] **Step 6: Commit**

```bash
git add src/vexas/pipeline.py tests/test_pipeline.py
git commit -m "feat: gate each article on its relevance verdict before extraction"
```

---

### Task 4: Wire `run()`, CLI summary, and cost estimate

**Files:**
- Modify: `src/vexas/pipeline.py` (`run()` -- add `gate` param + triage loop)
- Modify: `src/vexas/cli.py` (`estimate_cost` gate term; `format_triage_summary`; print in `main()`)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `TriageReport`, `classify_cached`, `RelevanceLabel`, `_relevance_cache` (Task 3).
- Produces:
  - `run(limit=None, verify=True, gate=True) -> PipelineResult` with `result.triage` populated when `gate` is set.
  - `format_triage_summary(triage: TriageReport) -> str` in `cli.py` (counts only, no bare percentage).
  - `estimate_cost` includes one gate call per report (priced at the existing rates as an upper bound).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
from vexas.cli import estimate_cost, format_triage_summary
from vexas.pipeline import TriageReport
from vexas.relevance import RelevanceLabel, RelevanceVerdict


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'format_triage_summary'` and `AttributeError: GATE_CALLS`).

- [ ] **Step 3: Add gate pricing constants and term to `estimate_cost`**

In `src/vexas/cli.py`, add constants near `SEGMENT_CALLS` (after line 42):

```python
#: Relevance gate: one call per report before extraction (relevance.py). Priced
#: at the extraction rates here as an upper bound -- it actually runs on a
#: cheaper model, and excluded papers skip extraction entirely, so the real
#: run costs less than this estimate. Modeled as a single uncached call.
GATE_CALLS = 1
GATE_OUTPUT = 60
```

In `estimate_cost`, inside the `for chars in char_counts:` loop, add the gate as a single uncached call (mirroring segmentation). Change the segmentation/uncached lines:

```python
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
```

- [ ] **Step 4: Add `format_triage_summary`**

In `src/vexas/cli.py`, add above `main()`:

```python
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
```

- [ ] **Step 5: Print the summary in `main()`**

In `src/vexas/cli.py`, after the records line in `main()` (after line 255), add:

```python
    if result.triage:
        print(format_triage_summary(result.triage))
```

- [ ] **Step 6: Add the `gate` param to `run()` and wire the triage loop**

In `src/vexas/pipeline.py`, replace `run()` (lines 62-101) with:

```python
def run(limit: int | None = None, verify: bool = True,
        gate: bool = True) -> PipelineResult:
    """Fetch, triage, extract, gate and (by default) verify the whole corpus.

    When `gate` is set (default), each full-text paper is classified before
    extraction and only VEXAS case reports are extracted; the rest are recorded
    in result.triage. Set gate=False to extract every paper (pre-gate behavior).
    """
    pmc_ids = search_pmc_ids()
    result = PipelineResult(corpus_size=len(pmc_ids))
    gate_stats = GateStats()
    verify_stats = VerifyStats()
    triage = TriageReport()
    cache = _verified_cache if verify else _record_cache

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

        if gate:
            verdict = classify_cached(article, pmc_id, _relevance_cache)
            triage.record(pmc_id, verdict)
            if verdict.label is not RelevanceLabel.CASE_REPORT:
                continue

        cached = cache.get_json(pmc_id)
        if cached is not None:
            result.records.extend(PatientRecord(**item) for item in cached)
            continue

        records, article_gate, article_verify = records_from_article(
            article, verify=verify
        )
        gate_stats.merge(article_gate)
        verify_stats.merge(article_verify)
        cache.put_json(pmc_id, [r.model_dump(mode="json") for r in records])
        result.records.extend(records)

        if index % 10 == 0:
            print(f"{index}/{len(pmc_ids)} processed, "
                  f"{len(result.records)} records")

    result.gate_stats = gate_stats
    result.verify_stats = verify_stats
    result.triage = triage if gate else None
    return result
```

- [ ] **Step 7: Run the CLI tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (264 + 2 = 266).

- [ ] **Step 9: Commit**

```bash
git add src/vexas/pipeline.py src/vexas/cli.py tests/test_cli.py
git commit -m "feat: run the relevance gate in the pipeline and report triage"
```

---

### Task 5: Docs + manual judgment validation (no-spend)

**Files:**
- Modify: `README.md` (document the triage stage)
- Modify: `docs/superpowers/specs/2026-07-21-vexas-relevance-gate-design.md` (record the manual validation outcome)

**Interfaces:** none (documentation + recorded manual result).

- [ ] **Step 1: Document the stage in the README**

In `README.md`, under "How it stays honest", add a bullet after the "Blind verification" bullet:

```markdown
- **Relevance triage.** Before extraction, each paper is classified as a VEXAS
  case report or not. Papers that only mention VEXAS in a differential or a
  review are excluded and listed with the verbatim quote behind the call, so a
  false-positive paper never contributes a record to the denominators.
```

In the "How it works" eight-stage list, insert after stage 1 (fetch) and renumber:

```markdown
2. **triage** each paper as a VEXAS case report vs a mention-only / review paper
```

- [ ] **Step 2: Validate real judgment by hand (no-spend, creator pattern)**

This is the manual step the design's testing section calls for -- the offline
suite proves routing; a human (or a real-key run) confirms the model's judgment.
Acting as the relevance model by hand (see memory `creator-runs-app-via-claude`),
read each eval paper's cached source text and author its verdict:

Run to print each paper's source text (no API cost):

```bash
.venv/bin/python -c "
from vexas.fetch.pmc import fetch_full_text, parse_article
for pmc_id in ['PMC8056719', 'PMC8313797', 'PMC11582098']:
    a = parse_article(fetch_full_text(pmc_id))
    print('===', pmc_id, '===')
    print(a.full_text[:1500])
"
```

Expected judgments (confirm against the text; each quote must be a verbatim
substring):
- `PMC8056719` -> `mention_only` (Adult-onset Still's disease; VEXAS only in the differential).
- `PMC8313797` -> `case_report` (single VEXAS patient, UBA1 M41L).
- `PMC11582098` -> `case_report` (8-patient VEXAS series).

- [ ] **Step 3: Record the outcome in the design doc**

Append a short "Manual validation" section to the design doc noting the three
verdicts above were confirmed by hand against the cached source text, the date,
and that the live `classify_relevance` HTTP path remains unexercised until a
real-key run (same status as the extraction smoke tests).

- [ ] **Step 4: Run the full suite (docs change should not affect it)**

Run: `.venv/bin/pytest -q`
Expected: all pass (266).

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-07-21-vexas-relevance-gate-design.md
git commit -m "docs: document the relevance triage stage and record manual validation"
```

---

## Self-Review

**Spec coverage:**
- Placement / stage 2.5 in `run()` before extraction -> Task 4 (`run()` loop).
- `relevance.py` module (labels, verdict, prompt, classify) -> Task 2.
- Fail-closed to UNCERTAIN (exception / unparseable / unknown label / ungrounded quote) -> Task 2 tests + `classify_relevance`.
- Precedence rule (review + case = case_report) -> encoded in `RELEVANCE_INSTRUCTIONS` (Task 2) and confirmed by the manual validation on PMC11582098 (Task 5).
- Neutral preamble (not extraction's SHARED_RULES) -> Task 2 `RELEVANCE_INSTRUCTIONS`.
- Haiku model + `model` override on `complete_json` -> Task 1.
- No cache breakpoint on the gate prompt -> Task 2 `build_relevance_prompt` + test.
- `gate=True` toggle, symmetric with verify -> Task 4 `run()`.
- Relevance cache (resume-not-restart) -> Task 3 `classify_cached` + Task 4 `_relevance_cache` wiring.
- CASE_REPORT extracts / others skip extraction -> Task 3 `process_article` tests. (Note: `run()` inlines the same branch directly for cache-ordering; `process_article` is the offline-tested proof of the routing rule.)
- `PipelineResult.triage` breakdown with quotes -> Task 3.
- Run-summary print, counts only, no bare percentage -> Task 4 `format_triage_summary` + test.
- Estimate adds the gate call as an upper bound -> Task 4 `estimate_cost` + test.
- Offline abstention proof (mention_only verdict excludes PMC8056719) -> Task 3 `test_mention_only_article_skips_extraction`.
- Manual/live judgment validation -> Task 5.
- README documents the stage -> Task 5.

**Note on `run()` vs `process_article`:** `run()` inlines the gate branch (classify_cached -> record -> continue) rather than calling `process_article`, because `run()` also threads the extraction cache lookup between the gate and extraction. `process_article` exists as the offline-testable statement of the same routing rule (extraction is skipped for non-case-reports). Both share `RelevanceLabel.CASE_REPORT` as the single keep condition, so they cannot drift on the decision. This duplication is deliberate and small; if `run()` grows, refactor it to call `process_article` with an injected cache-aware classify.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows complete code.

**Type consistency:** `RelevanceLabel` members and `.value` strings, `RelevanceVerdict(label, quote, reason)`, `TriageReport` field names (`case_report`/`mention_only`/`review`/`uncertain`/`excluded`/`flagged`/`total`/`record`), `classify_cached(article, pmc_id, cache, classify=)`, `process_article(... ) -> (records, gate_stats, verify_stats, verdict)`, and `format_triage_summary(triage)` are used identically across Tasks 2-5. `complete_json(..., model=None)` signature matches its call in `classify_relevance`. Cache JSON shape `{"label","quote","reason"}` is written and read identically in `classify_cached`.
