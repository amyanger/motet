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
from vexas.relevance import RelevanceLabel, RelevanceVerdict, classify_relevance
from vexas.schema import PatientRecord
from vexas.segment import segment_texts, split_patients
from vexas.verify import VerifyStats, verify_record

# Verified and unverified records are cached separately: the second pass
# prunes fields, so an unverified cache entry must never be served to a
# verified run (the key is only the PMC id, not the pipeline configuration).
_record_cache = DiskCache("records")
_verified_cache = DiskCache("records_verified")
_relevance_cache = DiskCache("relevance")


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


@dataclass
class PipelineResult:
    records: list[PatientRecord] = field(default_factory=list)
    gate_stats: GateStats | None = None
    verify_stats: VerifyStats | None = None
    corpus_size: int = 0
    full_text_count: int = 0
    failed_pmc_ids: list[str] = field(default_factory=list)
    triage: "TriageReport | None" = None


def records_from_article(
    article: Article, extract=extract_unit, segment=segment_texts,
    verify: bool = False, verify_fn=verify_record,
) -> tuple[list[PatientRecord], GateStats, VerifyStats]:
    """Extract every patient record from one article.

    Case series are split into per-patient units first, so no prompt ever
    sees two patients at once. When `verify` is set, each record is checked
    field-by-field against its own unit text by a blind second pass.
    """
    gate_stats = GateStats()
    verify_stats = VerifyStats()
    if not article.sections:
        return [], gate_stats, verify_stats

    units = split_patients(article, segment(article))
    records = []
    for unit in units:
        record, unit_gate = extract(unit)
        gate_stats.merge(unit_gate)
        if verify:
            record, unit_verify = verify_fn(record, unit.text)
            verify_stats.merge(unit_verify)
        records.append(record)
    return records, gate_stats, verify_stats


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


def run(limit: int | None = None, verify: bool = True) -> PipelineResult:
    """Fetch, extract, gate and (by default) verify the whole corpus."""
    pmc_ids = search_pmc_ids()
    result = PipelineResult(corpus_size=len(pmc_ids))
    gate_stats = GateStats()
    verify_stats = VerifyStats()
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
    return result
