from vexas.cache import DiskCache
from vexas.extract.runner import GateStats
from vexas.fetch.pmc import Article
from vexas.pipeline import (
    PipelineResult,
    TriageEntry,
    TriageReport,
    classify_cached,
    process_article,
    records_from_article,
)
from vexas.relevance import RelevanceLabel, RelevanceVerdict
from vexas.schema import GroundedField, PatientRecord
from vexas.verify import VerifyStats


def _one_unit(article):
    """A segment stub that never calls the API: one unit = full text."""
    return [article.full_text]


def test_records_from_article_uses_stubbed_extraction():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t",
                      sections={"Case": "A 68-year-old man with fevers."})

    def fake_extract(unit, complete=None):
        return (PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id),
                GateStats(kept=3, rejected=1))

    records, stats, verify_stats = records_from_article(
        article, extract=fake_extract, segment=_one_unit
    )
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

    records, _, _ = records_from_article(
        article, extract=fake_extract, segment=_one_unit
    )
    assert records == []


def test_records_from_article_makes_one_record_per_segmented_unit():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t",
                      sections={"Cases": "first patient. second patient."})

    def fake_extract(unit, complete=None):
        return (PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id),
                GateStats())

    records, _, _ = records_from_article(
        article, extract=fake_extract,
        segment=lambda a: ["first patient.", "second patient."],
    )
    assert [r.patient_id for r in records] == ["39012345-p1", "39012345-p2"]


def test_verification_runs_when_enabled_and_prunes_records():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t",
                      sections={"Case": "some text"})

    def fake_extract(unit, complete=None):
        record = PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id)
        record.outcome = GroundedField(value="died", quote="patient died")
        return record, GateStats(kept=1)

    def fake_verify(record, source, complete=None):
        record.outcome = None  # verifier rejects the field
        return record, VerifyStats(verified=0, rejected=1)

    records, _, verify_stats = records_from_article(
        article, extract=fake_extract, segment=_one_unit,
        verify=True, verify_fn=fake_verify,
    )
    assert records[0].outcome is None
    assert verify_stats.rejected == 1


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


def test_verification_is_skipped_when_disabled():
    article = Article(pmc_id="PMC1", pmid="39012345", title="t",
                      sections={"Case": "some text"})

    def fake_extract(unit, complete=None):
        return (PatientRecord(pmid=unit.pmid, patient_id=unit.patient_id),
                GateStats(kept=1))

    def fake_verify(record, source, complete=None):
        raise AssertionError("verify must not run when disabled")

    records, _, verify_stats = records_from_article(
        article, extract=fake_extract, segment=_one_unit,
        verify=False, verify_fn=fake_verify,
    )
    assert len(records) == 1
    assert verify_stats.verified == 0
    assert verify_stats.rejected == 0
