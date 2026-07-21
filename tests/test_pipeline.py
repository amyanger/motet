from vexas.extract.runner import GateStats
from vexas.fetch.pmc import Article
from vexas.pipeline import PipelineResult, records_from_article
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
