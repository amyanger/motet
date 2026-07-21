import pytest

from vexas.extract import client
from vexas.extract.client import extract_json_block
from vexas.config import settings


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


def test_complete_json_defaults_to_settings_model(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            class Msg:
                content = [type("B", (), {"text": '{"ok": true}'})()]
                usage = type("U", (), {"input_tokens": 1})()
            return Msg()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(client, "_client", FakeClient())
    result = client.complete_json([{"type": "text", "text": "hi"}])
    assert result == {"ok": True}
    assert captured["model"] == settings.model


def test_complete_json_uses_explicit_model(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            class Msg:
                content = [type("B", (), {"text": '{"ok": true}'})()]
                usage = type("U", (), {"input_tokens": 1})()
            return Msg()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(client, "_client", FakeClient())
    client.complete_json([{"type": "text", "text": "hi"}], model="claude-haiku-4-5-20251001")
    assert captured["model"] == "claude-haiku-4-5-20251001"
