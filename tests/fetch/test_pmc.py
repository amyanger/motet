from pathlib import Path

import pytest

from vexas.fetch.pmc import parse_article, parse_sections

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_pmc.xml"


@pytest.fixture
def xml():
    return FIXTURE.read_text(encoding="utf-8")


def test_parse_sections_keys_by_section_title(xml):
    sections = parse_sections(xml)
    assert "Introduction" in sections
    assert "Case presentation" in sections


def test_section_text_joins_paragraphs(xml):
    sections = parse_sections(xml)
    case = sections["Case presentation"]
    assert "recurrent fevers for six months" in case
    assert "ruxolitinib" in case


def test_inline_markup_is_flattened(xml):
    """UBA1 is wrapped in <italic> and must not break the text."""
    sections = parse_sections(xml)
    assert "somatic UBA1 mutations" in sections["Introduction"]


def test_parse_article_extracts_identifiers(xml):
    article = parse_article(xml)
    assert article.pmc_id == "PMC9999999"
    assert article.pmid == "39012345"
    assert article.title == "VEXAS syndrome presenting with chondritis"


def test_full_text_concatenates_all_sections(xml):
    article = parse_article(xml)
    assert "recurrent fevers" in article.full_text
    assert "somatic UBA1 mutations" in article.full_text


def test_quotes_from_sections_survive_the_gate(xml):
    """Guards the seam between fetch and gate: text extracted here must
    still match quotes the model will produce from it."""
    from vexas.gate import is_grounded

    article = parse_article(xml)
    assert is_grounded("recurrent fevers for six months", article.full_text)
