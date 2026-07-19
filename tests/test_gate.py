from vexas.gate import is_grounded, normalize_for_match


def test_exact_substring_is_grounded():
    source = "The patient developed recurrent fevers and ear chondritis."
    assert is_grounded("recurrent fevers", source) is True


def test_absent_text_is_not_grounded():
    source = "The patient developed recurrent fevers."
    assert is_grounded("ear chondritis", source) is False


def test_whitespace_differences_are_tolerated():
    source = "The patient developed\n  recurrent   fevers\tand chondritis."
    assert is_grounded("recurrent fevers and chondritis", source) is True


def test_case_differences_are_tolerated():
    source = "Macrocytic Anemia was noted."
    assert is_grounded("macrocytic anemia", source) is True


def test_unicode_dashes_are_normalized():
    source = "Treatment was steroid–sparing throughout."
    assert is_grounded("steroid-sparing", source) is True


def test_curly_quotes_are_normalized():
    source = "The authors described a “classic” presentation."
    assert is_grounded('"classic" presentation', source) is True


def test_paraphrase_is_rejected():
    """The core failure mode: model infers instead of quoting."""
    source = "The patient was treated with ruxolitinib 10 mg twice daily."
    assert is_grounded("patient received a JAK inhibitor", source) is False


def test_empty_quote_is_not_grounded():
    assert is_grounded("", "some source text") is False


def test_whitespace_only_quote_is_not_grounded():
    assert is_grounded("   \n  ", "some source text") is False


def test_normalize_collapses_runs_of_whitespace():
    assert normalize_for_match("a  \n\t b") == "a b"
