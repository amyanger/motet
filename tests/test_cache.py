import pytest

from vexas.cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache("test_ns", root=tmp_path)


def test_get_returns_none_when_missing(cache):
    assert cache.get("39012345") is None


def test_put_then_get_roundtrips(cache):
    cache.put("39012345", "<article>body</article>")
    assert cache.get("39012345") == "<article>body</article>"


def test_has_reflects_presence(cache):
    assert cache.has("39012345") is False
    cache.put("39012345", "x")
    assert cache.has("39012345") is True


def test_json_roundtrips(cache):
    payload = {"pmid": "39012345", "patients": 2}
    cache.put_json("39012345", payload)
    assert cache.get_json("39012345") == payload


def test_namespaces_are_isolated(tmp_path):
    a = DiskCache("raw", root=tmp_path)
    b = DiskCache("extracted", root=tmp_path)
    a.put("39012345", "raw text")
    assert b.get("39012345") is None


def test_unsafe_keys_are_sanitized(cache):
    cache.put("../../etc/passwd", "value")
    assert cache.get("../../etc/passwd") == "value"


def test_unicode_survives_roundtrip(cache):
    cache.put("39012345", "chondritis – M41T “classic”")
    assert cache.get("39012345") == "chondritis – M41T “classic”"
