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
