import pytest

from vexas.extract.client import extract_json_block


def test_parses_bare_json():
    assert extract_json_block('{"a": 1}') == {"a": 1}


def test_parses_json_inside_fenced_block():
    raw = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
    assert extract_json_block(raw) == {"a": 1}


def test_parses_json_with_leading_prose():
    raw = 'I found the following:\n{"a": 1}'
    assert extract_json_block(raw) == {"a": 1}


def test_raises_on_unparseable_output():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_block("I could not find anything.")


def test_handles_nested_braces():
    raw = '{"outer": {"inner": [1, 2]}}'
    assert extract_json_block(raw) == {"outer": {"inner": [1, 2]}}
