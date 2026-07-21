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
    # is enabled -- segmentation itself already supports N units.
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
