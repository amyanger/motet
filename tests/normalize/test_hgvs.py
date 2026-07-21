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
