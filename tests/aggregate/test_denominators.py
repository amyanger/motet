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
