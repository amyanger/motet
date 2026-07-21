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
