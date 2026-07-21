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
