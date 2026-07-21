from vexas.fetch.pmc import Article
from vexas.segment import (
    PatientUnit,
    detect_patient_count,
    segment_texts,
    split_patients,
)


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


# --- detect_patient_count and segment_texts (model-driven) ---

def _series_article():
    return Article(
        pmc_id="PMC9999999",
        pmid="39012345",
        title="Two cases of VEXAS",
        sections={
            "Cases": "Two patients are described. A 68-year-old man had "
                     "recurrent fevers. A 55-year-old woman had a rash."
        },
    )


def _anchors_stub(payload):
    return lambda prompt, **kwargs: payload


def test_detect_patient_count_returns_number_of_patients():
    stub = _anchors_stub({"patients": [{"anchor": "a"}, {"anchor": "b"},
                                       {"anchor": "c"}]})
    assert detect_patient_count(_series_article(), complete=stub) == 3


def test_detect_patient_count_is_at_least_one():
    stub = _anchors_stub({"patients": []})
    assert detect_patient_count(_series_article(), complete=stub) == 1


def test_single_patient_returns_the_whole_text():
    article = _series_article()
    stub = _anchors_stub({"patients": [{"anchor": "A 68-year-old man"}]})
    blocks = segment_texts(article, complete=stub)
    assert blocks == [article.full_text]


def test_two_patients_split_at_verbatim_anchors():
    article = _series_article()
    stub = _anchors_stub({"patients": [{"anchor": "A 68-year-old man"},
                                       {"anchor": "A 55-year-old woman"}]})
    blocks = segment_texts(article, complete=stub)
    assert len(blocks) == 2
    assert "68-year-old man had recurrent fevers" in blocks[0]
    assert "55-year-old" not in blocks[0]
    assert "55-year-old woman had a rash" in blocks[1]


def test_segmentation_preserves_every_byte():
    """Concatenating the units must reproduce the source exactly, or the
    gate rejects quotes downstream."""
    article = _series_article()
    stub = _anchors_stub({"patients": [{"anchor": "A 68-year-old man"},
                                       {"anchor": "A 55-year-old woman"}]})
    blocks = segment_texts(article, complete=stub)
    assert "".join(blocks) == article.full_text


def test_anchor_list_order_does_not_matter():
    article = _series_article()
    stub = _anchors_stub({"patients": [{"anchor": "A 55-year-old woman"},
                                       {"anchor": "A 68-year-old man"}]})
    blocks = segment_texts(article, complete=stub)
    assert len(blocks) == 2
    assert "68-year-old man had recurrent fevers" in blocks[0]


def test_falls_back_to_single_unit_when_an_anchor_is_not_verbatim():
    article = _series_article()
    stub = _anchors_stub({"patients": [{"anchor": "A 68-year-old man"},
                                       {"anchor": "a 55 year old lady"}]})
    blocks = segment_texts(article, complete=stub)
    assert blocks == [article.full_text]


def test_falls_back_when_anchors_are_ambiguous():
    """Two anchors resolving to the same offset is unreliable -- do not
    split."""
    article = _series_article()
    stub = _anchors_stub({"patients": [{"anchor": "A 68-year-old man"},
                                       {"anchor": "A 68-year-old man"}]})
    blocks = segment_texts(article, complete=stub)
    assert blocks == [article.full_text]
