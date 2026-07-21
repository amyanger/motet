from vexas.fetch.pmc import Article
from vexas.segment import PatientUnit, split_patients


def _article():
    return Article(
        pmc_id="PMC9999999",
        pmid="39012345",
        title="Two cases of VEXAS",
        sections={"Case presentation": "Patient 1 ... Patient 2 ..."},
    )


def test_single_unit_gets_suffix_p1():
    units = split_patients(_article(), ["A 68-year-old man ..."])
    assert len(units) == 1
    assert units[0].patient_id == "39012345-p1"


def test_multiple_units_get_sequential_ids():
    units = split_patients(_article(), ["first case text", "second case text"])
    assert [u.patient_id for u in units] == ["39012345-p1", "39012345-p2"]


def test_unit_text_is_preserved_verbatim():
    """Segmentation must not alter text, or the gate breaks downstream."""
    text = "A 68-year-old man  with\nrecurrent fevers."
    units = split_patients(_article(), [text])
    assert units[0].text == text


def test_empty_units_are_dropped():
    units = split_patients(_article(), ["real text", "   ", ""])
    assert len(units) == 1


def test_pmid_falls_back_to_pmc_id_when_absent():
    article = Article(pmc_id="PMC9999999", pmid=None, title="t",
                      sections={"Body": "x"})
    units = split_patients(article, ["text"])
    assert units[0].patient_id == "PMC9999999-p1"
